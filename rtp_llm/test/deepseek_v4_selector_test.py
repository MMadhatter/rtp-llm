"""Tests for the production-path selector wiring inside :class:`_DeepSeekV4LayerWrap`.

Three things we want to lock down so future kernel additions don't regress:

  1. ``_pick_indexer_impl`` returns one of the three known names and a
     callable, and the order matches the documented preference (FP4 →
     DeepGEMM FP8 → bf16 fused).
  2. ``_CsaIndexerSelector`` exposes the same parameter names
     (``W_IUQ`` / ``w_heads``) as the reference :class:`CsaLightningIndexer`
     so the loader bind helpers still find them.
  3. ``DeepSeekV4MoE.forward`` routes through ``deepgemm_megamoe_or_batched``
     and falls back to the bf16 path when ``fp8_expert_weights`` is None,
     producing numerically identical output to the old direct
     ``batched_experts_forward`` call.
"""

from unittest import TestCase, main
from unittest.mock import patch

import torch


class PickIndexerImplTest(TestCase):
    def setUp(self):
        from rtp_llm.models_py.model_desc.deepseek_v4 import _pick_indexer_impl
        from rtp_llm.models_py.modules.hybrid.deepgemm_indexer import (
            deepgemm_indexer_score_topk,
        )
        from rtp_llm.models_py.modules.hybrid.fp4_indexer import fp4_indexer_score_topk
        from rtp_llm.models_py.modules.hybrid.fused_indexer import (
            fused_indexer_score_topk,
        )

        self.pick = _pick_indexer_impl
        self.fp4 = fp4_indexer_score_topk
        self.dg = deepgemm_indexer_score_topk
        self.bf16 = fused_indexer_score_topk

    def test_returns_callable_and_known_name(self):
        fn, name = self.pick()
        self.assertTrue(callable(fn))
        self.assertIn(name, ("fp4", "deepgemm_fp8", "fused_bf16"))

    def test_blackwell_prefers_fp4(self):
        # Force the gate functions into a known state.
        with patch(
            "rtp_llm.models_py.model_desc.deepseek_v4.has_fp4_kernels",
            return_value=True,
        ):
            fn, name = self.pick()
        self.assertEqual(name, "fp4")
        self.assertIs(fn, self.fp4)

    def test_hopper_with_deepgemm_picks_deepgemm(self):
        with patch(
            "rtp_llm.models_py.model_desc.deepseek_v4.has_fp4_kernels",
            return_value=False,
        ), patch(
            "rtp_llm.models_py.model_desc.deepseek_v4._HAS_DEEPGEMM_INDEXER",
            True,
        ):
            fn, name = self.pick()
        self.assertEqual(name, "deepgemm_fp8")
        self.assertIs(fn, self.dg)

    def test_falls_back_to_fused_bf16(self):
        with patch(
            "rtp_llm.models_py.model_desc.deepseek_v4.has_fp4_kernels",
            return_value=False,
        ), patch(
            "rtp_llm.models_py.model_desc.deepseek_v4._HAS_DEEPGEMM_INDEXER",
            False,
        ):
            fn, name = self.pick()
        self.assertEqual(name, "fused_bf16")
        self.assertIs(fn, self.bf16)


class CsaIndexerSelectorParamsTest(TestCase):
    """Selector module must expose the same param shapes as the reference,
    so :func:`_bind_v4_attention` keeps working unchanged."""

    def setUp(self):
        from rtp_llm.models_py.model_desc.deepseek_v4 import _CsaIndexerSelector
        from rtp_llm.models_py.modules.hybrid.csa_attention import CsaLightningIndexer

        self.Sel = _CsaIndexerSelector
        self.Ref = CsaLightningIndexer

    def test_param_shapes_match_reference(self):
        kwargs = dict(
            q_lora_rank=64,
            num_indexer_heads=4,
            indexer_head_dim=32,
            top_k=8,
        )
        sel = self.Sel(**kwargs)
        ref = self.Ref(**kwargs)
        self.assertEqual(sel.W_IUQ.shape, ref.W_IUQ.shape)
        self.assertEqual(sel.w_heads.shape, ref.w_heads.shape)
        self.assertEqual(sel.num_indexer_heads, ref.num_indexer_heads)
        self.assertEqual(sel.indexer_head_dim, ref.indexer_head_dim)
        self.assertEqual(sel.top_k, ref.top_k)

    def test_forward_returns_topk_idx_and_scores(self):
        torch.manual_seed(0)
        # Use float32 so the test runs on CPU without the bf16-only matmul
        # path (bf16 @ bf16 isn't always implemented on x86 reference torch).
        sel = self.Sel(
            q_lora_rank=64,
            num_indexer_heads=4,
            indexer_head_dim=32,
            top_k=4,
            dtype=torch.float32,
        )
        c_Q = torch.randn(1, 8, 64)
        K_IComp = torch.randn(1, 16, 32)
        idx, scores = sel(c_Q, K_IComp)
        self.assertEqual(idx.shape, (1, 8, 4))
        self.assertEqual(scores.shape, (1, 8, 4))
        self.assertTrue(torch.isfinite(scores).all())

    def test_impl_name_set(self):
        sel = self.Sel(
            q_lora_rank=32,
            num_indexer_heads=2,
            indexer_head_dim=16,
            top_k=4,
        )
        self.assertIn(sel.impl_name, ("fp4", "deepgemm_fp8", "fused_bf16"))


class MoEDeepGemmSelectorTest(TestCase):
    """``DeepSeekV4MoE.forward`` must still produce identical output when
    ``fp8_expert_weights is None`` (selector falls back to bf16)."""

    def setUp(self):
        from rtp_llm.models_py.model_desc.deepseek_v4_layer import DeepSeekV4MoE

        self.MoE = DeepSeekV4MoE
        torch.manual_seed(0)

    def test_fp8_weights_default_to_none(self):
        moe = self.MoE(
            hidden_size=16,
            num_routed_experts=8,
            moe_intermediate_size=24,
            top_k=2,
            layer_idx=4,  # past the hash-routed prefix
            num_hash_layers=3,
            n_shared_experts=1,
            routed_scaling_factor=1.0,
            swiglu_limit=10.0,
        )
        self.assertIsNone(moe.fp8_expert_weights)

    def test_forward_routes_through_selector(self):
        # bf16-only path must execute without error and return the right
        # shape — proves the selector's fallback works in the model.
        moe = self.MoE(
            hidden_size=16,
            num_routed_experts=8,
            moe_intermediate_size=24,
            top_k=2,
            layer_idx=4,
            num_hash_layers=3,
            n_shared_experts=1,
            routed_scaling_factor=1.0,
            swiglu_limit=10.0,
        )
        x = torch.randn(2, 4, 16)
        out = moe(x)
        self.assertEqual(out.shape, (2, 4, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_matches_direct_batched_call(self):
        """Selector-fallback output ≡ direct batched_experts_forward output.

        Locks down: wiring in the selector must not change numerics on the
        bf16 path. Any divergence means the dispatch logic accidentally
        rerouted to a different path.
        """
        from rtp_llm.models_py.modules.moe.batched_experts import (
            batched_experts_forward,
        )

        moe = self.MoE(
            hidden_size=16,
            num_routed_experts=8,
            moe_intermediate_size=24,
            top_k=2,
            layer_idx=4,
            num_hash_layers=3,
            n_shared_experts=1,
            routed_scaling_factor=1.0,
            swiglu_limit=10.0,
        )
        x = torch.randn(2, 4, 16)
        # The selector path:
        out_selector = moe(x)
        # The direct path: replicate the math without the selector.
        flat = x.reshape(-1, 16)
        from rtp_llm.models_py.modules.moe.v4_gating import noaux_tc_topk_v4

        logits = flat @ moe.gate
        topk_idx, gate_vals = noaux_tc_topk_v4(
            logits,
            moe.e_score_bias,
            moe.top_k,
            moe.routed_scaling_factor,
            norm_topk_prob=moe.norm_topk_prob,
        )
        out_direct_routed = batched_experts_forward(
            flat,
            topk_idx,
            gate_vals,
            moe.W_gate,
            moe.W_up,
            moe.W_down,
            moe.swiglu_limit,
        )
        out_direct = (out_direct_routed + moe._shared_forward(flat)).reshape(x.shape)
        torch.testing.assert_close(out_selector, out_direct, rtol=1e-5, atol=1e-5)


class PackV4MoeFp8Test(TestCase):
    """``_pack_v4_moe_fp8`` produces the exact layout DeepGEMM expects.

    The pack is a pure tensor reshape — we can verify it on CPU even when
    DeepGEMM isn't installed, by patching the import out and inspecting the
    resulting shapes.
    """

    def setUp(self):
        from rtp_llm.models_py.model_desc.deepseek_v4 import _pack_v4_moe_fp8
        from rtp_llm.models_py.model_desc.deepseek_v4_layer import DeepSeekV4MoE
        from rtp_llm.utils.model_weight import W

        self.pack = _pack_v4_moe_fp8
        self.MoE = DeepSeekV4MoE
        self.W = W

    def _make_moe(self, hidden=128, inter=256, E=4):
        return self.MoE(
            hidden_size=hidden,
            num_routed_experts=E,
            moe_intermediate_size=inter,
            top_k=2,
            layer_idx=0,
            num_hash_layers=3,
            n_shared_experts=1,
            routed_scaling_factor=1.0,
            swiglu_limit=10.0,
        )

    def _make_fp8_layer_w(self, hidden=128, inter=256, E=4):
        # Random e4m3 + ue8m0-style scale for w1/w2/w3.
        w1 = torch.randn(E, hidden, inter).to(torch.float8_e4m3fn)
        w3 = torch.randn(E, hidden, inter).to(torch.float8_e4m3fn)
        w2 = torch.randn(E, inter, hidden).to(torch.float8_e4m3fn)
        s1 = torch.rand(E, hidden // 128, inter // 128, dtype=torch.float32)
        s3 = torch.rand(E, hidden // 128, inter // 128, dtype=torch.float32)
        s2 = torch.rand(E, inter // 128, hidden // 128, dtype=torch.float32)
        return {
            self.W.v4_experts_w1: w1,
            self.W.v4_experts_w1_s: s1,
            self.W.v4_experts_w3: w3,
            self.W.v4_experts_w3_s: s3,
            self.W.v4_experts_w2: w2,
            self.W.v4_experts_w2_s: s2,
        }

    def test_returns_false_without_deepgemm(self):
        # Patch the import inside the helper to fail.
        moe = self._make_moe()
        layer_w = self._make_fp8_layer_w()
        with patch.dict("sys.modules", {"deep_gemm": None}):
            self.assertFalse(self.pack(moe, layer_w))
        self.assertIsNone(moe.fp8_expert_weights)

    def test_returns_false_for_bf16_ckpt(self):
        # Pure-bf16 ckpt: dtype check should reject.
        moe = self._make_moe()
        layer_w = self._make_fp8_layer_w()
        # Cast back to bf16 to simulate a non-quantised ckpt.
        layer_w[self.W.v4_experts_w1] = (
            layer_w[self.W.v4_experts_w1].to(torch.float32).to(torch.bfloat16)
        )
        # deep_gemm import: even if it succeeds, dtype check trips first.
        self.assertFalse(self.pack(moe, layer_w))
        self.assertIsNone(moe.fp8_expert_weights)

    def test_returns_false_when_keys_missing(self):
        moe = self._make_moe()
        layer_w = self._make_fp8_layer_w()
        del layer_w[self.W.v4_experts_w2_s]
        self.assertFalse(self.pack(moe, layer_w))
        self.assertIsNone(moe.fp8_expert_weights)

    def test_packed_layout_matches_deepgemm_expectation(self):
        # Force the deep_gemm import gate true even on CPU.
        import sys
        import types

        fake_dg = types.ModuleType("deep_gemm")
        moe = self._make_moe(hidden=128, inter=256, E=4)
        layer_w = self._make_fp8_layer_w(hidden=128, inter=256, E=4)
        with patch.dict(sys.modules, {"deep_gemm": fake_dg}):
            ok = self.pack(moe, layer_w)
        self.assertTrue(ok)
        self.assertIsNotNone(moe.fp8_expert_weights)
        W_gu, W_gu_s, W_d, W_d_s = moe.fp8_expert_weights
        # Shape contracts from deepgemm_megamoe.py docstring.
        self.assertEqual(W_gu.shape, (4, 2 * 256, 128))
        self.assertEqual(W_d.shape, (4, 128, 256))
        self.assertEqual(W_gu_s.shape, (4, 2 * (256 // 128), 128 // 128))
        self.assertEqual(W_d_s.shape, (4, 128 // 128, 256 // 128))
        self.assertEqual(W_gu.dtype, torch.float8_e4m3fn)
        self.assertEqual(W_d.dtype, torch.float8_e4m3fn)
        self.assertEqual(W_gu_s.dtype, torch.float32)

    def test_pack_preserves_values_via_uint8_cat(self):
        """The uint8-view cat must round-trip the FP8 bit pattern unchanged."""
        import sys
        import types

        fake_dg = types.ModuleType("deep_gemm")
        moe = self._make_moe(hidden=128, inter=256, E=2)
        layer_w = self._make_fp8_layer_w(hidden=128, inter=256, E=2)
        w1 = layer_w[self.W.v4_experts_w1]
        w3 = layer_w[self.W.v4_experts_w3]
        with patch.dict(sys.modules, {"deep_gemm": fake_dg}):
            self.assertTrue(self.pack(moe, layer_w))
        W_gu, _, _, _ = moe.fp8_expert_weights
        # First half of the inter axis must equal w1^T (bit-identical).
        gate_bytes = W_gu[:, :256, :].contiguous().view(torch.uint8)
        ref_bytes = w1.transpose(-1, -2).contiguous().view(torch.uint8)
        self.assertTrue(torch.equal(gate_bytes, ref_bytes))
        # Second half == w3^T.
        up_bytes = W_gu[:, 256:, :].contiguous().view(torch.uint8)
        ref3_bytes = w3.transpose(-1, -2).contiguous().view(torch.uint8)
        self.assertTrue(torch.equal(up_bytes, ref3_bytes))


if __name__ == "__main__":
    main()
