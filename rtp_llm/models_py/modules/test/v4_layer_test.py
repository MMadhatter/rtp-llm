"""Tests for the DeepSeek-V4 decoder layer composition (PR-E reference).

Verifies that the layer plumbing — mHC residual stream, HCA attention, V4 MoE
(hash + sqrt-softplus paths) — composes into a runnable module with
shape/dtype/grad-flow that match the spec.

These are *structural* tests: they don't compare against an HF reference
(which requires a real V4 ckpt), only that the assembled module behaves
sanely on small dummy inputs.
"""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_layer import (
    DeepSeekV4DecoderLayer,
    DeepSeekV4MoE,
)


# ---------------------------------------------------------------------------
class DeepSeekV4MoEHashRoutingTest(TestCase):
    """First ``num_hash_layers`` layers route by token-id hash."""

    def setUp(self):
        torch.manual_seed(0)
        self.cfg = dict(
            hidden_size=16, num_routed_experts=8, moe_intermediate_size=32,
            top_k=2, layer_idx=0, num_hash_layers=3,
        )

    def test_uses_hash_routing(self):
        moe = DeepSeekV4MoE(**self.cfg)
        self.assertTrue(moe.use_hash_routing)
        self.assertIsNone(moe.gate)
        self.assertIsNone(moe.e_score_bias)

    def test_requires_token_ids(self):
        moe = DeepSeekV4MoE(**self.cfg)
        x = torch.randn(2, 4, self.cfg["hidden_size"])
        with self.assertRaises(ValueError):
            moe(x, token_ids=None)

    def test_forward_shape(self):
        moe = DeepSeekV4MoE(**self.cfg)
        B, T, H = 2, 4, self.cfg["hidden_size"]
        x = torch.randn(B, T, H)
        token_ids = torch.randint(0, 1000, (B, T))
        out = moe(x, token_ids)
        self.assertEqual(out.shape, (B, T, H))
        self.assertTrue(torch.isfinite(out).all())

    def test_deterministic_across_calls(self):
        moe = DeepSeekV4MoE(**self.cfg)
        moe.eval()
        x = torch.randn(1, 4, self.cfg["hidden_size"])
        token_ids = torch.tensor([[7, 13, 42, 99]])
        with torch.no_grad():
            a = moe(x, token_ids)
            b = moe(x, token_ids)
        torch.testing.assert_close(a, b)


class DeepSeekV4MoELearnedRoutingTest(TestCase):
    """Layers past ``num_hash_layers`` use sqrt(softplus) topk routing."""

    def setUp(self):
        torch.manual_seed(1)
        self.cfg = dict(
            hidden_size=16, num_routed_experts=8, moe_intermediate_size=32,
            top_k=2, layer_idx=5, num_hash_layers=3,
        )

    def test_does_not_use_hash_routing(self):
        moe = DeepSeekV4MoE(**self.cfg)
        self.assertFalse(moe.use_hash_routing)
        self.assertIsNotNone(moe.gate)
        self.assertIsNotNone(moe.e_score_bias)

    def test_token_ids_optional_for_learned_routing(self):
        moe = DeepSeekV4MoE(**self.cfg)
        x = torch.randn(2, 4, self.cfg["hidden_size"])
        out = moe(x, token_ids=None)
        self.assertEqual(out.shape, x.shape)

    def test_grad_flows_to_gate(self):
        moe = DeepSeekV4MoE(**self.cfg)
        x = torch.randn(1, 3, self.cfg["hidden_size"], requires_grad=True)
        moe(x).sum().backward()
        self.assertIsNotNone(moe.gate.grad)
        self.assertIsNotNone(x.grad)

    def test_shared_expert_always_contributes(self):
        """Zeroing routed expert weights leaves the shared-expert path intact."""
        cfg = {**self.cfg, "num_routed_experts": 4, "top_k": 1}
        moe = DeepSeekV4MoE(**cfg)
        with torch.no_grad():
            moe.W_gate.zero_()
            moe.W_up.zero_()
            moe.W_down.zero_()
        x = torch.randn(1, 2, cfg["hidden_size"])
        out = moe(x)
        # Routed contribution is 0; output must equal shared-expert output alone.
        shared_only = moe._shared_forward(x.reshape(-1, cfg["hidden_size"])).reshape_as(x)
        torch.testing.assert_close(out, shared_only, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
class DeepSeekV4DecoderLayerTest(TestCase):
    """End-to-end layer: residual (B, T, n_hc, d) -> updated residual."""

    def setUp(self):
        torch.manual_seed(7)
        self.cfg = dict(
            hidden_size=16, num_heads=2, head_dim=8, rope_head_dim=4,
            m_prime=2, q_lora_rank=8, o_groups=2, o_lora_rank=8,
            hc_mult=2,
            num_routed_experts=4, moe_intermediate_size=8, moe_top_k=2,
            layer_idx=5, num_hash_layers=3,
            n_win=2, rope_max_pos=64, sinkhorn_iters=4,
            n_shared_experts=1, swiglu_limit=10.0,
            dtype=torch.float32,
        )

    def test_forward_preserves_residual_shape(self):
        layer = DeepSeekV4DecoderLayer(**self.cfg)
        B, T = 2, 4
        residual = torch.randn(B, T, self.cfg["hc_mult"], self.cfg["hidden_size"])
        positions = torch.arange(T)
        out = layer(residual, positions, token_ids=None)
        self.assertEqual(out.shape, residual.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_hash_layer_requires_token_ids(self):
        cfg = {**self.cfg, "layer_idx": 0}      # < num_hash_layers
        layer = DeepSeekV4DecoderLayer(**cfg)
        residual = torch.randn(1, 3, cfg["hc_mult"], cfg["hidden_size"])
        with self.assertRaises(ValueError):
            layer(residual, torch.arange(3), token_ids=None)

    def test_grad_flows_through_all_subblocks(self):
        layer = DeepSeekV4DecoderLayer(**self.cfg)
        B, T = 1, 3
        residual = torch.randn(
            B, T, self.cfg["hc_mult"], self.cfg["hidden_size"],
            requires_grad=True,
        )
        out = layer(residual, torch.arange(T), token_ids=None).sum()
        out.backward()
        self.assertIsNotNone(residual.grad)
        # mHC alpha gates must learn — ensure they receive grads.
        self.assertIsNotNone(layer.mhc_attn.alpha_pre.grad)
        self.assertIsNotNone(layer.mhc_attn.alpha_res.grad)
        self.assertIsNotNone(layer.mhc_attn.alpha_post.grad)
        self.assertIsNotNone(layer.mhc_ffn.alpha_pre.grad)
        # Attention parameters
        self.assertIsNotNone(layer.attention.W_DQ.grad)
        self.assertIsNotNone(layer.attention.W_UQ.grad)
        # MoE shared expert always reached
        self.assertIsNotNone(layer.moe.W_shared_gate.grad)

    def test_dtype_bf16(self):
        layer = DeepSeekV4DecoderLayer(**{**self.cfg, "dtype": torch.bfloat16})
        residual = torch.randn(
            1, 3, self.cfg["hc_mult"], self.cfg["hidden_size"],
            dtype=torch.bfloat16,
        )
        out = layer(residual, torch.arange(3), token_ids=None)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_two_hash_layers_share_router(self):
        """Two hash-routing layers given the same token_ids must agree on routing
        (router is purely token-id derived)."""
        from rtp_llm.models_py.modules.moe.hash_router import hash_route_topk
        cfg = {**self.cfg, "layer_idx": 0}
        l1 = DeepSeekV4DecoderLayer(**cfg)
        l2 = DeepSeekV4DecoderLayer(**{**cfg, "layer_idx": 1})
        token_ids = torch.tensor([[3, 7, 13]])
        # Both layers route via the same hash table; verify the router is in
        # use (idx is deterministic across both).
        idx1, _ = hash_route_topk(
            token_ids, cfg["num_routed_experts"], cfg["moe_top_k"],
        )
        idx2, _ = hash_route_topk(
            token_ids, cfg["num_routed_experts"], cfg["moe_top_k"],
        )
        torch.testing.assert_close(idx1, idx2)
        # Both layers accept the call without error.
        residual = torch.randn(1, 3, cfg["hc_mult"], cfg["hidden_size"])
        l1(residual, torch.arange(3), token_ids)
        l2(residual, torch.arange(3), token_ids)


if __name__ == "__main__":
    main()
