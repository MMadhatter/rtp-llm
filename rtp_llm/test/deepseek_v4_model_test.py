"""Unit tests for the DeepSeek-V4 engine model wiring.

Covers the structural contract of :mod:`rtp_llm.models_py.model_desc.deepseek_v4`:
the per-layer attention dispatcher (CSA / HCA / SWA-only), the FP8 ue8m0
dequant helper, the mHC parameter binding, and the per-layer module
construction off a small synthetic config.

The full :class:`DeepSeekV4Model` wraps the Embedding / KV cache plumbing
provided by :class:`GptModelBase`, both of which require GPU side ops in
production. Tests below focus on the parts that can be exercised end-to-end
on CPU (layer construction + a forward pass on a single
:class:`_DeepSeekV4LayerWrap`) and skip the model-level wiring when CUDA
is not present.
"""

from typing import Dict
from unittest import TestCase, main

import torch

from rtp_llm.models.deepseek_v4 import DeepSeekV4
from rtp_llm.models_py.model_desc.deepseek_v4 import (
    _bind_mhc,
    _bind_v4_attention,
    _bind_v4_moe,
    _DeepSeekV4LayerWrap,
    _dequant_fp8_block,
    _layer_kind,
)
from rtp_llm.models_py.model_desc.deepseek_v4_layer import DeepSeekV4MoE
from rtp_llm.models_py.modules.hybrid.csa_attention import CsaAttention
from rtp_llm.models_py.modules.hybrid.hca_attention import HcaAttention
from rtp_llm.models_py.modules.mhc import MhcLayer
from rtp_llm.test.deepseek_v4_infra_test import (
    FLASH_BASE_CONFIG,
    _make_blank_v4_model_config,
)
from rtp_llm.utils.model_weight import W


# ---------------------------------------------------------------------------
class LayerKindDispatcherTest(TestCase):
    """``_layer_kind`` must map compress_ratios entries to the right class
    of attention block — the per-layer wrap depends on this mapping."""

    def test_swa_only_for_zero(self):
        self.assertEqual(_layer_kind(0), "swa_only")

    def test_csa_for_four(self):
        self.assertEqual(_layer_kind(4), "csa")

    def test_hca_for_one_twenty_eight(self):
        self.assertEqual(_layer_kind(128), "hca")

    def test_unknown_ratio_raises(self):
        with self.assertRaises(ValueError):
            _layer_kind(7)


# ---------------------------------------------------------------------------
class DequantFp8BlockTest(TestCase):
    """ue8m0-scaled FP8 dequant — sanity round-trip."""

    def test_uniform_scale_round_trip(self):
        # 256x256 fake-FP8 weight (uint8) with all blocks scaled by 2^0 = 1.
        # uint8 raw value should round-trip to bf16 of the same magnitude.
        w = torch.full((256, 256), 7, dtype=torch.uint8)
        s = torch.full((2, 2), 127, dtype=torch.float32)  # 127 → 2^0 = 1.0
        out = _dequant_fp8_block(w, s, block=128)
        self.assertEqual(out.dtype, torch.bfloat16)
        # Every entry should be ≈ 7 (scaled by 1.0).
        torch.testing.assert_close(out, torch.full_like(out, 7.0), rtol=0, atol=1e-2)

    def test_per_block_scaling(self):
        # Two blocks, top-left scaled by 2^0=1, bottom-right by 2^1=2.
        w = torch.full((256, 256), 4, dtype=torch.uint8)
        s = torch.tensor([[127.0, 127.0], [127.0, 128.0]], dtype=torch.float32)
        out = _dequant_fp8_block(w, s, block=128).float()
        self.assertAlmostEqual(out[0, 0].item(), 4.0, places=2)
        self.assertAlmostEqual(out[200, 200].item(), 8.0, places=2)
        # Off-diagonal blocks stay at 4.
        self.assertAlmostEqual(out[0, 200].item(), 4.0, places=2)


# ---------------------------------------------------------------------------
def _make_layer_weights(
    config, layer_id: int = 0, num_routed_experts: int = 4
) -> Dict[str, torch.Tensor]:
    """Build a synthetic per-layer weight dict that's small enough to
    instantiate every reference module on CPU. Uses bf16 throughout, no
    FP8 scales (so the dequant fast-path is skipped)."""
    h = config.hidden_size
    head_dim = config.attn_config.size_per_head
    num_heads = config.attn_config.head_num
    q_lora = config.attn_config.q_lora_rank
    o_groups = config.attn_config.o_groups
    o_lora = config.attn_config.o_lora_rank
    inter = config.moe_inter_size
    n_hc = config.attn_config.hc_mult

    f = {"dtype": torch.bfloat16}
    layer_w: Dict[str, torch.Tensor] = {
        W.v4_attn_norm: torch.ones(h, **f),
        W.v4_ffn_norm: torch.ones(h, **f),
        W.v4_q_norm: torch.ones(head_dim, **f),
        W.v4_kv_norm: torch.ones(head_dim, **f),
        W.v4_attn_sink: torch.zeros(num_heads, **f),
        W.v4_wq_a: torch.randn(h, q_lora, **f) * 0.02,
        W.v4_wq_b: torch.randn(q_lora, num_heads * head_dim, **f) * 0.02,
        W.v4_wkv: torch.randn(h, head_dim, **f) * 0.02,
        W.v4_wo_a: torch.randn(
            o_groups, (num_heads // o_groups) * head_dim, o_lora, **f
        )
        * 0.02,
        W.v4_wo_b: torch.randn(o_groups, o_lora, h // o_groups, **f) * 0.02,
        # mHC packing — see _bind_mhc docstring.
        W.v4_hc_attn_fn: torch.randn(n_hc * h, n_hc + n_hc * n_hc + n_hc, **f) * 0.02,
        W.v4_hc_attn_base: torch.zeros(n_hc + n_hc * n_hc + n_hc + n_hc * h, **f),
        W.v4_hc_attn_scale: torch.zeros(3, **f),
        W.v4_hc_ffn_fn: torch.randn(n_hc * h, n_hc + n_hc * n_hc + n_hc, **f) * 0.02,
        W.v4_hc_ffn_base: torch.zeros(n_hc + n_hc * n_hc + n_hc + n_hc * h, **f),
        W.v4_hc_ffn_scale: torch.zeros(3, **f),
        # MoE — stacked routed experts.
        W.v4_experts_w1: torch.randn(num_routed_experts, h, inter, **f) * 0.02,
        W.v4_experts_w2: torch.randn(num_routed_experts, inter, h, **f) * 0.02,
        W.v4_experts_w3: torch.randn(num_routed_experts, h, inter, **f) * 0.02,
        W.v4_shared_w1: torch.randn(h, inter, **f) * 0.02,
        W.v4_shared_w2: torch.randn(inter, h, **f) * 0.02,
        W.v4_shared_w3: torch.randn(h, inter, **f) * 0.02,
        W.v4_moe_gate_w: torch.randn(h, num_routed_experts, **f) * 0.02,
    }
    if layer_id < config.moe_hash_routing_layers:
        # Hash-routed layer ships tid2eid and never gate.bias.
        layer_w[W.v4_moe_gate_tid2eid] = torch.zeros(1, dtype=torch.int64)
    else:
        layer_w[W.v4_moe_gate_b] = torch.zeros(num_routed_experts, **f)
    return layer_w


def _small_v4_config():
    """Single-block-scale V4 config: tiny dims that still hit every code path."""
    config = _make_blank_v4_model_config()
    DeepSeekV4._populate_from_hf_dict(config, FLASH_BASE_CONFIG)
    config.hidden_size = 32
    config.attn_config.head_num = 4
    config.attn_config.size_per_head = 8
    config.attn_config.v_head_dim = 8
    config.attn_config.rope_head_dim = 4
    config.attn_config.nope_head_dim = 4
    config.attn_config.q_lora_rank = 16
    config.attn_config.o_groups = 2
    config.attn_config.o_lora_rank = 8
    config.attn_config.hc_mult = 4
    config.attn_config.hc_sinkhorn_iters = 5
    config.attn_config.indexer_head_num = 2
    config.attn_config.indexer_head_dim = 8
    config.attn_config.indexer_topk = 4
    config.attn_config.sliding_window = 4
    config.moe_inter_size = 16
    config.expert_num = 4
    config.moe_k = 2
    config.moe_hash_routing_layers = 2
    config.num_layers = 4
    config.attn_config.layer_compress_ratios = [0, 0, 4, 128]
    return config


# ---------------------------------------------------------------------------
class DeepSeekV4LayerWrapTest(TestCase):
    """Constructing _DeepSeekV4LayerWrap on each attention kind and running
    a forward — verifies wiring + shape contract on CPU."""

    def setUp(self):
        torch.manual_seed(0)
        self.config = _small_v4_config()

    def _build(self, layer_id: int):
        layer_w = _make_layer_weights(
            self.config,
            layer_id,
            num_routed_experts=self.config.expert_num,
        )
        compress_ratio = int(self.config.attn_config.layer_compress_ratios[layer_id])
        return _DeepSeekV4LayerWrap(
            self.config,
            layer_id,
            compress_ratio,
            layer_w,
            dtype=torch.bfloat16,
        )

    def test_swa_only_layer_uses_hca_with_m_eq_1(self):
        layer = self._build(0)  # compress_ratio=0
        self.assertIsInstance(layer.attention, HcaAttention)
        self.assertEqual(layer.attention.m_prime, 1)
        self.assertEqual(layer.kind, "swa_only")

    def test_csa_layer_builds_csa_attention(self):
        layer = self._build(2)  # compress_ratio=4
        self.assertIsInstance(layer.attention, CsaAttention)
        self.assertEqual(layer.kind, "csa")

    def test_hca_layer_builds_hca_attention(self):
        layer = self._build(3)  # compress_ratio=128
        self.assertIsInstance(layer.attention, HcaAttention)
        self.assertEqual(layer.attention.m_prime, 128)
        self.assertEqual(layer.kind, "hca")

    def test_layer_holds_two_mhc_blocks(self):
        layer = self._build(0)
        self.assertIsInstance(layer.mhc_attn, MhcLayer)
        self.assertIsInstance(layer.mhc_ffn, MhcLayer)
        # Independent params.
        self.assertIsNot(layer.mhc_attn.W_pre, layer.mhc_ffn.W_pre)

    def test_hash_layer_has_no_learned_gate(self):
        layer = self._build(0)  # < num_hash_layers=2
        self.assertTrue(layer.moe.use_hash_routing)
        self.assertIsNone(layer.moe.gate)

    def test_learned_layer_has_gate_and_bias(self):
        layer = self._build(2)  # >= num_hash_layers
        self.assertFalse(layer.moe.use_hash_routing)
        self.assertIsNotNone(layer.moe.gate)
        self.assertIsNotNone(layer.moe.e_score_bias)

    def test_swa_only_forward_runs(self):
        """Forward on a SWA-only layer with B=1, T=8: must produce a
        finite (B, T, n_hc, hidden) residual stream of the same shape."""
        layer = self._build(0)
        n_hc = self.config.attn_config.hc_mult
        T = 8
        residual = torch.randn(
            1, T, n_hc, self.config.hidden_size, dtype=torch.bfloat16
        )
        positions = torch.arange(T, dtype=torch.long)
        token_ids = torch.zeros(1, T, dtype=torch.long)
        out = layer(residual, positions, token_ids)
        self.assertEqual(out.shape, residual.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out.float()).all())

    def test_hca_forward_runs(self):
        layer = self._build(3)
        n_hc = self.config.attn_config.hc_mult
        T = 8
        residual = torch.randn(
            1, T, n_hc, self.config.hidden_size, dtype=torch.bfloat16
        )
        positions = torch.arange(T, dtype=torch.long)
        token_ids = torch.randint(0, 100, (1, T), dtype=torch.long)
        out = layer(residual, positions, token_ids)
        self.assertEqual(out.shape, residual.shape)
        self.assertTrue(torch.isfinite(out.float()).all())

    def test_csa_forward_runs(self):
        layer = self._build(2)
        n_hc = self.config.attn_config.hc_mult
        T = 16  # CSA m=4 needs at least m+1 tokens for a non-trivial compress
        residual = torch.randn(
            1, T, n_hc, self.config.hidden_size, dtype=torch.bfloat16
        )
        positions = torch.arange(T, dtype=torch.long)
        token_ids = torch.randint(0, 100, (1, T), dtype=torch.long)
        out = layer(residual, positions, token_ids)
        self.assertEqual(out.shape, residual.shape)
        self.assertTrue(torch.isfinite(out.float()).all())


# ---------------------------------------------------------------------------
class BindHelpersTest(TestCase):
    """The three ``_bind_*`` helpers: copy loaded tensors into reference
    module parameters. Test they don't raise on the synthetic weight set
    (real coverage of binding correctness is the layer forward UTs above)."""

    def setUp(self):
        torch.manual_seed(0)
        self.config = _small_v4_config()
        self.layer_w = _make_layer_weights(self.config, layer_id=2)

    def test_bind_mhc_does_not_raise(self):
        mhc = MhcLayer(
            hidden_size=self.config.hidden_size,
            hc_mult=self.config.attn_config.hc_mult,
            sinkhorn_iters=2,
            dtype=torch.bfloat16,
        )
        _bind_mhc(mhc, self.layer_w, "attn")
        # alpha was zeroed → mHC becomes input-independent; behaviour
        # tested in mhc_test.py, here we just check binding succeeded.
        self.assertEqual(mhc.alpha_pre.item(), 0.0)
        self.assertEqual(mhc.alpha_res.item(), 0.0)
        self.assertEqual(mhc.alpha_post.item(), 0.0)

    def test_bind_attention_does_not_raise(self):
        attn = HcaAttention(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.attn_config.head_num,
            head_dim=self.config.attn_config.size_per_head,
            rope_head_dim=self.config.attn_config.rope_head_dim,
            m_prime=128,
            q_lora_rank=self.config.attn_config.q_lora_rank,
            o_groups=self.config.attn_config.o_groups,
            o_lora_rank=self.config.attn_config.o_lora_rank,
            dtype=torch.bfloat16,
        )
        _bind_v4_attention(attn, self.layer_w, "hca")
        # Sink stays zeroed (V4 default).
        torch.testing.assert_close(attn.sink_logits, torch.zeros_like(attn.sink_logits))

    def test_bind_moe_does_not_raise(self):
        moe = DeepSeekV4MoE(
            hidden_size=self.config.hidden_size,
            num_routed_experts=self.config.expert_num,
            moe_intermediate_size=self.config.moe_inter_size,
            top_k=self.config.moe_k,
            layer_idx=2,  # > num_hash_layers, so learned routing
            num_hash_layers=self.config.moe_hash_routing_layers,
            dtype=torch.bfloat16,
        )
        _bind_v4_moe(moe, self.layer_w, hash_routed=False)


# ---------------------------------------------------------------------------
class FullModelConfigSanityTest(TestCase):
    """Spot-check that ``_create_python_model`` returns the model class so
    the engine path uses it (was previously raising NotImplementedError)."""

    def test_create_python_model_imports_v4_model(self):
        # ``_create_python_model`` is the engine entry-point — it instantiates
        # the V4 PyTorch model with the loaded weights and stashes it on
        # ``self.py_model``. Full instantiation requires a real ModelWeights /
        # parallelism config so we just verify the import resolves cleanly
        # (was previously raising NotImplementedError).
        from rtp_llm.models_py.model_desc.deepseek_v4 import DeepSeekV4Model

        self.assertTrue(callable(DeepSeekV4Model))


# ---------------------------------------------------------------------------
class MultiLayerChainSmokeTest(TestCase):
    """End-to-end-ish CPU smoke: build all 3 attention kinds, chain them,
    then reduce residual + apply final norm + lm head. Mirrors the math in
    :meth:`DeepSeekV4Model.forward` but skips :class:`GptModelBase` (which
    pulls in CUDA-only Embedding / KV-cache plumbing).

    Catches construction-time bugs before the user runs E2E:
      * inter-layer shape contract (residual stays ``(B, T, n_hc, H)``)
      * expand_residual / reduce_residual round-trip
      * final RMSNorm + LM-head matmul shapes
      * MoE FP8 pack does NOT fire on bf16-only synthetic ckpts (so the
        forward selector still goes through the bf16 batched path)
    """

    def setUp(self):
        torch.manual_seed(0)
        self.config = _small_v4_config()
        # Force a 4-layer chain that hits each attention kind:
        #   0: SWA-only (compress_ratio=0)
        #   1: SWA-only
        #   2: CSA      (compress_ratio=4)
        #   3: HCA      (compress_ratio=128)
        self.config.attn_config.layer_compress_ratios = [0, 0, 4, 128]
        self.config.num_layers = 4

    def _build_chain(self):
        from rtp_llm.models_py.modules.mhc import expand_residual, reduce_residual

        layers = []
        for layer_id in range(self.config.num_layers):
            layer_w = _make_layer_weights(
                self.config,
                layer_id,
                num_routed_experts=self.config.expert_num,
            )
            ratio = int(self.config.attn_config.layer_compress_ratios[layer_id])
            layer = _DeepSeekV4LayerWrap(
                self.config,
                layer_id,
                ratio,
                layer_w,
                dtype=torch.bfloat16,
            )
            layers.append(layer)
        return layers, expand_residual, reduce_residual

    def test_construct_all_layer_kinds_in_one_chain(self):
        layers, _, _ = self._build_chain()
        kinds = [layer.kind for layer in layers]
        self.assertEqual(kinds, ["swa_only", "swa_only", "csa", "hca"])
        # Every layer must have its own mHC params (no aliasing).
        for i, layer in enumerate(layers):
            self.assertIsNot(
                layer.mhc_attn.W_pre,
                layer.mhc_ffn.W_pre,
                f"layer {i} attn/ffn mHC must not alias",
            )

    def test_forward_chain_preserves_residual_shape(self):
        """One forward pass through the whole chain — what the engine
        does in :meth:`DeepSeekV4Model.forward` minus embedding/lm-head."""
        layers, expand_residual, reduce_residual = self._build_chain()
        n_hc = self.config.attn_config.hc_mult
        B, T, H = 1, 16, self.config.hidden_size

        # Mimic embedding output: (B, T, H) bf16.
        embeds = torch.randn(B, T, H, dtype=torch.bfloat16)
        residual = expand_residual(embeds, n_hc)
        positions = torch.arange(T, dtype=torch.long)
        token_ids = torch.randint(0, 100, (B, T), dtype=torch.long)

        for layer in layers:
            residual = layer(residual, positions, token_ids)
            self.assertEqual(
                residual.shape,
                (B, T, n_hc, H),
                f"layer {layer.layer_idx} broke the residual contract",
            )
            self.assertTrue(torch.isfinite(residual.float()).all())

        # Final reduce + norm + lm_head shape contract.
        hidden = reduce_residual(residual)
        self.assertEqual(hidden.shape, (B, T, H))
        # Synthetic final norm + LM head — bf16 matmul.
        final_norm_w = torch.ones(H, dtype=torch.bfloat16)
        var = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normed = hidden * torch.rsqrt(var + 1e-6).to(hidden.dtype) * final_norm_w
        vocab = 256
        lm_head_w = torch.randn(H, vocab, dtype=torch.bfloat16) * 0.02
        logits = normed @ lm_head_w
        self.assertEqual(logits.shape, (B, T, vocab))
        self.assertTrue(torch.isfinite(logits.float()).all())

    def test_moe_fp8_pack_skipped_for_bf16_ckpt(self):
        """The synthetic weight set is bf16; the pack helper must decline so
        the forward path stays on the batched bf16 reference."""
        layers, _, _ = self._build_chain()
        # Pick the first non-hash-routed layer (layer 2, CSA).
        moe = layers[2].moe
        self.assertIsNone(
            moe.fp8_expert_weights,
            "bf16 synthetic ckpt must NOT trigger FP8 pack",
        )

    def test_csa_indexer_swapped_to_selector(self):
        """The post-construction indexer swap from the reference
        :class:`CsaLightningIndexer` to :class:`_CsaIndexerSelector`
        must apply to every CSA layer in the chain."""
        from rtp_llm.models_py.model_desc.deepseek_v4 import _CsaIndexerSelector

        layers, _, _ = self._build_chain()
        for layer in layers:
            if layer.kind == "csa":
                self.assertIsInstance(
                    layer.attention.indexer,
                    _CsaIndexerSelector,
                    f"CSA layer {layer.layer_idx} indexer must be the selector",
                )


if __name__ == "__main__":
    main()
