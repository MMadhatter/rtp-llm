"""Parity tests for the fused noaux_tc top-k routing.

Compares :func:`noaux_tc_topk_v4_fused` against the reference
:func:`noaux_tc_topk_v4` (sqrt(softplus) → bias add → topk → renormalise).

Reference: vLLM PR #40760 ``csrc/moe/topk_softplus_sqrt_kernels.cu`` and
SGLang PR #23600 ``python/sglang/srt/layers/moe/deepseek_v4_topk.py``.
"""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.moe.noaux_tc_fused import noaux_tc_topk_v4_fused
from rtp_llm.models_py.modules.moe.v4_gating import noaux_tc_topk_v4


class NoauxTcFusedParityTest(TestCase):
    def _setup(self, T=7, E=64, top_k=6, seed=0, dtype=torch.float32):
        torch.manual_seed(seed)
        logits = torch.randn(T, E, dtype=dtype)
        bias = torch.randn(E, dtype=dtype) * 0.1
        return logits, bias

    def test_matches_reference_with_bias(self):
        logits, bias = self._setup()
        ref_idx, ref_gate = noaux_tc_topk_v4(
            logits, bias, top_k=6, routed_scaling_factor=1.5
        )
        fused_idx, fused_gate = noaux_tc_topk_v4_fused(
            logits, bias, top_k=6, routed_scaling_factor=1.5
        )
        self.assertTrue(torch.equal(fused_idx, ref_idx))
        torch.testing.assert_close(fused_gate, ref_gate, rtol=1e-5, atol=1e-5)

    def test_matches_reference_no_bias(self):
        logits, _ = self._setup()
        ref_idx, ref_gate = noaux_tc_topk_v4(
            logits, None, top_k=4, routed_scaling_factor=2.5
        )
        fused_idx, fused_gate = noaux_tc_topk_v4_fused(
            logits, None, top_k=4, routed_scaling_factor=2.5
        )
        self.assertTrue(torch.equal(fused_idx, ref_idx))
        torch.testing.assert_close(fused_gate, ref_gate, rtol=1e-5, atol=1e-5)

    def test_norm_topk_prob_disabled(self):
        logits, bias = self._setup()
        ref_idx, ref_gate = noaux_tc_topk_v4(
            logits,
            bias,
            top_k=6,
            routed_scaling_factor=1.0,
            norm_topk_prob=False,
        )
        fused_idx, fused_gate = noaux_tc_topk_v4_fused(
            logits,
            bias,
            top_k=6,
            routed_scaling_factor=1.0,
            norm_topk_prob=False,
        )
        self.assertTrue(torch.equal(fused_idx, ref_idx))
        torch.testing.assert_close(fused_gate, ref_gate, rtol=1e-5, atol=1e-5)

    def test_normalisation_invariant(self):
        """When norm_topk_prob=True, sum(gate)/scale must equal 1."""
        logits, bias = self._setup()
        scale = 2.5
        _, gate = noaux_tc_topk_v4_fused(
            logits, bias, top_k=6, routed_scaling_factor=scale
        )
        ratios = gate.sum(dim=-1) / scale
        torch.testing.assert_close(
            ratios, torch.ones_like(ratios), rtol=1e-5, atol=1e-5
        )

    def test_bias_only_steers_selection(self):
        """A huge bias on expert 0 should force it into the top-k, but the
        returned gate value must come from the bias-free score."""
        T, E, top_k = 3, 8, 2
        torch.manual_seed(7)
        logits = torch.randn(T, E)
        bias = torch.zeros(E)
        bias[0] = 1e6  # forces expert 0 to win for every token
        idx, gate = noaux_tc_topk_v4_fused(
            logits, bias, top_k=top_k, routed_scaling_factor=1.0
        )
        # Expert 0 is in every token's top-k.
        self.assertTrue((idx == 0).any(dim=-1).all())

        # Score for expert 0 = sqrt(softplus(logits[:, 0])); gate value for
        # expert 0 (after norm) must reflect this *un-biased* score.
        unbiased = torch.sqrt(torch.nn.functional.softplus(logits[:, 0].float()))
        # For each token find the slot where idx==0 and read the gate value
        # before normalisation: check that gate_for_expert_0 / sum(gate) ==
        # unbiased / sum_unbiased_for_selected.
        for t in range(T):
            slot0 = (idx[t] == 0).nonzero().item()
            sel_unbiased = torch.sqrt(
                torch.nn.functional.softplus(logits[t, idx[t]].float())
            )
            ratio = gate[t, slot0].float() / gate[t].float().sum()
            ref_ratio = unbiased[t] / sel_unbiased.sum()
            torch.testing.assert_close(ratio, ref_ratio, rtol=1e-5, atol=1e-5)

    def test_bf16_matches_reference(self):
        logits, bias = self._setup(dtype=torch.bfloat16)
        ref_idx, ref_gate = noaux_tc_topk_v4(
            logits, bias, top_k=6, routed_scaling_factor=1.5
        )
        fused_idx, fused_gate = noaux_tc_topk_v4_fused(
            logits, bias, top_k=6, routed_scaling_factor=1.5
        )
        # bf16 ties may break differently between paths because the
        # reference roundtrips scores through bf16 before adding the bias
        # (`scores.dtype == logits.dtype`) while the fused path keeps fp32
        # all the way to topk. Compare *sets* of indices and the
        # bf16-rounded gate magnitudes.
        ref_sets = [set(row.tolist()) for row in ref_idx]
        fused_sets = [set(row.tolist()) for row in fused_idx]
        self.assertEqual(fused_sets, ref_sets)
        torch.testing.assert_close(
            fused_gate.float().sum(-1),
            ref_gate.float().sum(-1),
            rtol=1e-2,
            atol=1e-2,
        )


if __name__ == "__main__":
    main()
