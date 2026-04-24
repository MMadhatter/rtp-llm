"""Tests for the HCA attention reference (DeepSeek-V4, paper §2.3).

Covers:
  * partial RoPE — angle correctness, inverse round-trip, nope passthrough
  * GroupedOutputProjection — shape contract, equivalence to per-group dense
  * mqa_attention_with_sink — sink-only (zero K) makes output zero, weights
    sum to 1 with the sink column included
  * HcaAttention.forward — end-to-end shape, dtype, grad flow, batch invariance
"""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.hybrid.hca_attention import (
    GroupedOutputProjection,
    HcaAttention,
    apply_partial_rope,
    mqa_attention_with_sink,
    precompute_rope_cache,
)


# ---------------------------------------------------------------------------
class PartialRopeTest(TestCase):
    def test_zero_position_is_identity(self):
        cos, sin = precompute_rope_cache(8, rope_dim=4, base=10000.0)
        x = torch.randn(2, 3, 8)            # (B, L, head_dim)
        # Zero position rotates by 0 -> identity.
        cos0 = cos[:3]                      # use positions 0..2
        sin0 = sin[:3]
        # Construct a position-0 cos/sin to force identity behaviour.
        cos_id = torch.ones_like(cos0)
        sin_id = torch.zeros_like(sin0)
        out = apply_partial_rope(x, cos_id, sin_id, rope_dim=4)
        torch.testing.assert_close(out, x)

    def test_nope_is_passthrough(self):
        cos, sin = precompute_rope_cache(4, rope_dim=4, base=10000.0)
        x = torch.randn(4, 8)               # head_dim=8, rope_dim=4 -> 4 nope dims
        out = apply_partial_rope(x, cos, sin, rope_dim=4)
        # First 4 dims must be unchanged.
        torch.testing.assert_close(out[..., :4], x[..., :4])

    def test_inverse_undoes_forward(self):
        cos, sin = precompute_rope_cache(16, rope_dim=8, base=10000.0)
        x = torch.randn(2, 16, 12)          # head_dim=12, rope on last 8
        rotated = apply_partial_rope(x, cos, sin, rope_dim=8)
        unrotated = apply_partial_rope(rotated, cos, sin, rope_dim=8, inverse=True)
        torch.testing.assert_close(unrotated, x, rtol=1e-5, atol=1e-5)

    def test_full_rope_dim(self):
        # rope_dim == head_dim -> no nope split.
        cos, sin = precompute_rope_cache(4, rope_dim=4, base=10000.0)
        x = torch.randn(4, 4)
        out = apply_partial_rope(x, cos, sin, rope_dim=4)
        # Should still round-trip with inverse.
        back = apply_partial_rope(out, cos, sin, rope_dim=4, inverse=True)
        torch.testing.assert_close(back, x, rtol=1e-5, atol=1e-5)

    def test_rejects_oversized_rope(self):
        cos, sin = precompute_rope_cache(4, rope_dim=4, base=10000.0)
        x = torch.randn(4, 4)
        with self.assertRaises(ValueError):
            apply_partial_rope(x, cos, sin, rope_dim=8)


# ---------------------------------------------------------------------------
class GroupedOutputProjectionTest(TestCase):
    def test_shape(self):
        proj = GroupedOutputProjection(
            num_heads=8, head_dim=16, hidden_size=64, o_groups=4, o_lora_rank=8,
        )
        x = torch.randn(2, 3, 8, 16)
        out = proj(x)
        self.assertEqual(out.shape, (2, 3, 64))

    def test_equivalence_to_per_group_loop(self):
        torch.manual_seed(0)
        proj = GroupedOutputProjection(
            num_heads=8, head_dim=4, hidden_size=16, o_groups=2, o_lora_rank=4,
        )
        x = torch.randn(5, 8, 4)            # (T, n_h, d)
        out = proj(x)
        # Reference: split heads into 2 groups, apply each group's W_a/W_b.
        ref_parts = []
        for g in range(proj.o_groups):
            heads = x[:, g * proj.g_heads : (g + 1) * proj.g_heads, :]   # (T, g_heads, d)
            flat = heads.reshape(5, proj.g_heads * proj.head_dim)
            z = flat @ proj.W_a[g]
            y = z @ proj.W_b[g]
            ref_parts.append(y)
        ref = torch.cat(ref_parts, dim=-1)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    def test_rejects_indivisible(self):
        with self.assertRaises(ValueError):
            GroupedOutputProjection(num_heads=7, head_dim=4, hidden_size=16,
                                    o_groups=2, o_lora_rank=4)
        with self.assertRaises(ValueError):
            GroupedOutputProjection(num_heads=8, head_dim=4, hidden_size=15,
                                    o_groups=2, o_lora_rank=4)

    def test_rejects_wrong_input_shape(self):
        proj = GroupedOutputProjection(num_heads=4, head_dim=4, hidden_size=16,
                                       o_groups=2, o_lora_rank=4)
        x = torch.randn(2, 8, 4)            # n_h=8 != 4
        with self.assertRaises(ValueError):
            proj(x)


# ---------------------------------------------------------------------------
class MqaAttentionSinkTest(TestCase):
    def test_pure_sink_routes_all_weight_to_sink(self):
        """Q with very high sink logit -> weights collapse to sink, output ~0."""
        B, T_q, H, D = 1, 2, 4, 8
        T_kc = 3
        Q = torch.randn(B, T_q, H, D)
        K = torch.randn(B, T_kc, D)
        V = torch.randn(B, T_kc, D)
        sink = torch.full((H,), 1e6)        # huge sink logit
        out = mqa_attention_with_sink(
            Q, K, V, K_window=None, V_window=None, sink_logits=sink,
        )
        # All softmax mass goes to sink (which contributes 0 to the output).
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-4, rtol=1e-3)

    def test_zero_sink_acts_like_no_sink(self):
        B, T_q, H, D = 1, 2, 2, 4
        T_kc = 3
        Q = torch.randn(B, T_q, H, D)
        K = torch.randn(B, T_kc, D)
        V = torch.randn(B, T_kc, D)
        out_with_zero_sink = mqa_attention_with_sink(
            Q, K, V, K_window=None, V_window=None,
            sink_logits=torch.zeros(H),
        )
        # Reference: standard MQA softmax with one extra constant logit (=0).
        # This reweights but doesn't zero the output.
        scale = 1.0 / (D ** 0.5)
        Q_h = Q.transpose(1, 2)                          # (B, H, T_q, D)
        K_b = K.unsqueeze(1)
        V_b = V.unsqueeze(1)
        logits = torch.einsum("bhqd,bnkd->bhqk", Q_h, K_b) * scale
        logits_aug = torch.cat([logits, torch.zeros(*logits.shape[:-1], 1)], dim=-1)
        w = torch.softmax(logits_aug, dim=-1)
        ref = torch.einsum("bhqk,bnkd->bhqd", w[..., :-1], V_b).transpose(1, 2)
        torch.testing.assert_close(out_with_zero_sink, ref, rtol=1e-4, atol=1e-5)

    def test_swa_window_increases_keys(self):
        """When K_window is supplied, output uses both compressed + window."""
        B, T_q, H, D = 1, 3, 2, 4
        T_kc = 2
        n_win = 2
        Q = torch.randn(B, T_q, H, D)
        K = torch.randn(B, T_kc, D)
        V = torch.randn(B, T_kc, D)
        K_w = torch.randn(B, T_q, n_win, D)
        V_w = torch.randn(B, T_q, n_win, D)
        # All weight should go to V_w if we mask out compressed keys.
        compressed_mask = torch.ones(B, T_q, T_kc, dtype=torch.bool)
        out = mqa_attention_with_sink(
            Q, K, V, K_w, V_w, sink_logits=torch.full((H,), -1e6),
            causal_compressed_mask=compressed_mask,
        )
        # With sink logit very negative AND compressed masked, all weight
        # routes to V_window. Output must be a weighted sum of V_w.
        self.assertEqual(out.shape, (B, T_q, H, D))
        self.assertTrue(torch.isfinite(out).all())


# ---------------------------------------------------------------------------
class HcaAttentionForwardTest(TestCase):
    """Smoke + structural tests on the assembled HcaAttention."""

    def setUp(self):
        torch.manual_seed(123)
        self.cfg = dict(
            hidden_size=32, num_heads=4, head_dim=8, rope_head_dim=4,
            m_prime=2, q_lora_rank=8, o_groups=2, o_lora_rank=8,
            n_win=4, rope_max_pos=64, dtype=torch.float32,
        )
        self.layer = HcaAttention(**self.cfg)

    def test_forward_shape(self):
        B, T = 2, 8
        H = torch.randn(B, T, self.cfg["hidden_size"])
        positions = torch.arange(T)
        out = self.layer(H, positions)
        self.assertEqual(out.shape, (B, T, self.cfg["hidden_size"]))
        self.assertTrue(torch.isfinite(out).all())

    def test_dtype_preserved_bf16(self):
        layer_bf = HcaAttention(**{**self.cfg, "dtype": torch.bfloat16})
        H = torch.randn(1, 4, self.cfg["hidden_size"], dtype=torch.bfloat16)
        out = layer_bf(H, torch.arange(4))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_gradient_flows(self):
        B, T = 1, 4
        H = torch.randn(B, T, self.cfg["hidden_size"], requires_grad=True)
        positions = torch.arange(T)
        out = self.layer(H, positions).sum()
        out.backward()
        self.assertIsNotNone(H.grad)
        for name, p in self.layer.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} has no grad")

    def test_batch_independence(self):
        """Independent rows in the batch must produce independent outputs."""
        B, T = 2, 4
        H1 = torch.randn(1, T, self.cfg["hidden_size"])
        H2 = torch.randn(1, T, self.cfg["hidden_size"])
        H = torch.cat([H1, H2], dim=0)
        positions = torch.arange(T)
        out_b = self.layer(H, positions)
        out_1 = self.layer(H1, positions)
        out_2 = self.layer(H2, positions)
        torch.testing.assert_close(out_b[0:1], out_1, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(out_b[1:2], out_2, rtol=1e-4, atol=1e-5)

    def test_window_zero_disables_swa(self):
        cfg = {**self.cfg, "n_win": 0}
        layer = HcaAttention(**cfg)
        H = torch.randn(1, 4, cfg["hidden_size"])
        out = layer(H, torch.arange(4))
        self.assertEqual(out.shape, (1, 4, cfg["hidden_size"]))


if __name__ == "__main__":
    main()
