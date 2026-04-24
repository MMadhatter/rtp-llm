"""Unit tests for DeepSeek-V4 MoE primitives (M5):

  * sqrt(softplus) scoring vs naive math
  * noaux_tc topk without n_group constraint
  * Hash routing determinism + uniformity
  * Clamped SwiGLU bounds + V3-style fallback
"""

from collections import Counter
from unittest import TestCase, main

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.moe.clamped_swiglu import (
    clamped_swiglu,
    clamped_swiglu_split,
)
from rtp_llm.models_py.modules.moe.hash_router import (
    HASH_PRIMES,
    HASH_VERSION,
    hash_route_topk,
)
from rtp_llm.models_py.modules.moe.v4_gating import (
    noaux_tc_topk_v4,
    sqrt_softplus_score,
)


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------
class SqrtSoftplusTest(TestCase):
    """Match a closed-form reference."""

    def test_matches_naive(self):
        torch.manual_seed(0)
        x = torch.randn(8, 16) * 5
        ref = torch.sqrt(torch.log1p(torch.exp(x.float()).clamp(max=1e30)))
        torch.testing.assert_close(
            sqrt_softplus_score(x).float(), ref, rtol=1e-3, atol=1e-4
        )

    def test_non_negative(self):
        x = torch.randn(64) * 100
        s = sqrt_softplus_score(x)
        self.assertTrue((s >= 0).all())
        self.assertTrue(torch.isfinite(s).all())

    def test_monotonic(self):
        # sqrt(softplus) is strictly monotone increasing.
        x = torch.linspace(-10, 10, steps=200)
        s = sqrt_softplus_score(x)
        self.assertTrue((s[1:] >= s[:-1]).all())

    def test_dtype_preserved(self):
        for dt in (torch.float32, torch.bfloat16):
            self.assertEqual(sqrt_softplus_score(torch.zeros(4, dtype=dt)).dtype, dt)


# ---------------------------------------------------------------------------
class NoauxTcTopkV4Test(TestCase):
    """V4 routing — global topk, no group filtering."""

    def setUp(self):
        torch.manual_seed(7)
        self.E, self.K = 256, 6
        self.bias = torch.randn(self.E) * 0.01
        self.scaling = 1.5  # Flash-Base's routed_scaling_factor

    def test_shape_and_dtype(self):
        logits = torch.randn(13, self.E)
        idx, gate = noaux_tc_topk_v4(logits, self.bias, self.K, self.scaling)
        self.assertEqual(idx.shape, (13, self.K))
        self.assertEqual(gate.shape, (13, self.K))
        self.assertEqual(idx.dtype, torch.int64)
        self.assertEqual(gate.dtype, logits.dtype)

    def test_gate_sum_equals_scaling(self):
        logits = torch.randn(11, self.E)
        _, gate = noaux_tc_topk_v4(
            logits, self.bias, self.K, self.scaling, norm_topk_prob=True
        )
        torch.testing.assert_close(
            gate.sum(dim=-1),
            torch.full((11,), self.scaling, dtype=gate.dtype),
            rtol=1e-3, atol=1e-3,
        )

    def test_no_group_filtering(self):
        """V4 must consider all experts, not be restricted to n_group nodes."""
        # Construct logits where the global top-K all live in distant 'groups'
        # to verify we aren't accidentally clipping by group.
        E = 256
        logits = torch.full((1, E), -1e6)
        # Place top scores at expert ids 0, 64, 128, 192, 255, 1.
        # In V3 routing with n_group=8 (32 experts each) these would span
        # 5 different groups; the V3 'topk_group=4' filter would drop one.
        # V4 must keep all 6 since it has no such filter.
        chosen = [0, 64, 128, 192, 255, 1]
        for j, e in enumerate(chosen):
            logits[0, e] = 100 + j
        idx, _ = noaux_tc_topk_v4(
            logits, torch.zeros(E), top_k=6, routed_scaling_factor=1.0
        )
        self.assertEqual(set(idx[0].tolist()), set(chosen))

    def test_bias_steers_selection_only(self):
        """The bias affects which experts are chosen but not the gate value."""
        logits = torch.randn(1, self.E)
        scores_no_bias = sqrt_softplus_score(logits)
        # Make a bias that strongly prefers expert 0.
        bias = torch.zeros(self.E)
        bias[0] = 10.0
        idx, gate = noaux_tc_topk_v4(
            logits, bias, top_k=1, routed_scaling_factor=1.0,
            norm_topk_prob=False,
        )
        self.assertEqual(idx[0, 0].item(), 0)
        # The gate value at expert 0 must equal the *unbiased* score.
        torch.testing.assert_close(gate[0, 0], scores_no_bias[0, 0])

    def test_bias_can_be_none(self):
        logits = torch.randn(2, self.E)
        idx, gate = noaux_tc_topk_v4(logits, None, self.K, self.scaling)
        self.assertEqual(idx.shape, (2, self.K))


# ---------------------------------------------------------------------------
class HashRouterTest(TestCase):
    """Determinism + uniformity properties of the deterministic hash router."""

    def test_determinism_across_calls(self):
        ids = torch.tensor([0, 1, 2, 100, 1000, 10000])
        a, _ = hash_route_topk(ids, num_routed_experts=256, top_k=6)
        b, _ = hash_route_topk(ids, num_routed_experts=256, top_k=6)
        torch.testing.assert_close(a, b)

    def test_uniformity_over_vocab(self):
        # Roll all 129280 vocab token ids through the router; each of the 256
        # experts should receive close-to-uniform mass.
        E = 256
        K = 6
        ids = torch.arange(129280)
        idx, _ = hash_route_topk(ids, num_routed_experts=E, top_k=K)
        flat = idx.view(-1).tolist()
        counts = Counter(flat)
        expected = ids.numel() * K / E
        # Loose chi-square style: max deviation < 30% of expected. Multiplicative
        # hashing isn't a uniform PRF so we don't insist on tighter than that.
        for e in range(E):
            self.assertLess(abs(counts[e] - expected), 0.3 * expected)

    def test_gate_values_uniform_and_sum_to_scaling(self):
        ids = torch.arange(10)
        K = 6
        scaling = 1.5
        _, gate = hash_route_topk(
            ids, num_routed_experts=256, top_k=K, routed_scaling_factor=scaling
        )
        torch.testing.assert_close(
            gate, torch.full_like(gate, scaling / K)
        )
        torch.testing.assert_close(
            gate.sum(dim=-1),
            torch.full((10,), scaling, dtype=gate.dtype),
        )

    def test_top_k_must_not_exceed_prime_table(self):
        with self.assertRaises(ValueError):
            hash_route_topk(torch.tensor([0]), num_routed_experts=8,
                            top_k=len(HASH_PRIMES) + 1)

    def test_hash_version_is_pinned(self):
        # Bumping HASH_VERSION should be a deliberate change.
        self.assertEqual(HASH_VERSION, 1)

    def test_handles_2d_token_ids(self):
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        idx, gate = hash_route_topk(ids, num_routed_experts=64, top_k=4)
        self.assertEqual(idx.shape, (2, 3, 4))
        self.assertEqual(gate.shape, (2, 3, 4))

    def test_rejects_float_token_ids(self):
        with self.assertRaises(TypeError):
            hash_route_topk(torch.tensor([1.0, 2.0]), 64, 4)


# ---------------------------------------------------------------------------
class ClampedSwigluTest(TestCase):
    def test_clamps_branches(self):
        # gate >> limit (must clamp on top), linear < -limit (must clamp on bottom)
        gate_and_linear = torch.cat(
            [torch.full((1, 4), 50.0), torch.full((1, 4), -50.0)], dim=-1
        )
        out = clamped_swiglu(gate_and_linear, swiglu_limit=10.0)
        # silu(10) ~= 10.0, linear clamped to -10 -> out = silu(10) * -10
        expected = F.silu(torch.full((1, 4), 10.0)) * torch.full((1, 4), -10.0)
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

    def test_unbounded_when_limit_disabled(self):
        gate_and_linear = torch.cat(
            [torch.full((1, 2), 5.0), torch.full((1, 2), -5.0)], dim=-1
        )
        out_clamp = clamped_swiglu(gate_and_linear, swiglu_limit=10.0)
        out_no = clamped_swiglu(gate_and_linear, swiglu_limit=None)
        # Within the limit, clamping is a no-op.
        torch.testing.assert_close(out_clamp, out_no)

    def test_fallback_to_plain_swiglu_when_limit_zero(self):
        gate_and_linear = torch.cat(
            [torch.full((1, 2), 100.0), torch.full((1, 2), 100.0)], dim=-1
        )
        out = clamped_swiglu(gate_and_linear, swiglu_limit=0)
        plain = F.silu(torch.full((1, 2), 100.0)) * torch.full((1, 2), 100.0)
        torch.testing.assert_close(out, plain)

    def test_split_form_matches_fused_form(self):
        torch.manual_seed(0)
        x = torch.randn(3, 8)
        gate, linear = torch.chunk(x, 2, dim=-1)
        out_fused = clamped_swiglu(x, swiglu_limit=10.0)
        out_split = clamped_swiglu_split(gate, linear, swiglu_limit=10.0)
        torch.testing.assert_close(out_fused, out_split)


if __name__ == "__main__":
    main()
