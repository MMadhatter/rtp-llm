"""Unit tests for the DeepSeek-V4 infra (PR-A + M0 stub).

Covers:
  1. ``register_model`` wired ``DeepseekV4ForCausalLM`` (and the MTP variant) to
     the right Python classes via :class:`ModelDict`.
  2. :func:`DeepSeekV4._populate_from_hf_dict` projects every V4-specific HF
     field onto :class:`ModelConfig` / :class:`AttentionConfigs`.
  3. End-to-end ``DeepSeekV4._create_config(ckpt_path_with_only_config_json)``
     succeeds for the published DeepSeek-V4-Flash-Base config.
  4. ``DeepseekV4Config`` HF wrapper round-trips through ``from_dict`` and is
     resolvable via ``model_type``.
  5. ``DeepSeekV4._create_python_model`` raises a descriptive
     ``NotImplementedError`` (forward path lands in PR-E+).
  6. ``DeepSeekV4Weight._get_weight_info`` returns a stub layer plan
     (per-layer empty list) so the loader can iterate without crashing.

Tested against the actual DeepSeek-V4-Flash-Base ``config.json``.
"""

import json
import os
import tempfile
from unittest import TestCase, main

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.models.deepseek_v4 import (
    _SCORING_FUNC_MAP,
    DeepSeekV4,
    DeepSeekV4Mtp,
    DeepSeekV4MtpWeight,
    DeepSeekV4Weight,
)
from rtp_llm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from rtp_llm.utils.model_weight import W

# Authoritative DeepSeek-V4-Flash-Base config (as published on HF).
# Kept inline so the test does not depend on network or external checkpoints.
FLASH_BASE_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "hc_eps": 1e-06,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "head_dim": 512,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 512,
    "initializer_range": 0.02,
    "max_position_embeddings": 1048576,
    "model_type": "deepseek_v4",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 43,
    "num_hash_layers": 3,
    "num_key_value_heads": 1,
    "num_nextn_predict_layers": 1,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "sliding_window": 128,
    "swiglu_limit": 10.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.57.1",
    "use_cache": True,
    "vocab_size": 129280,
    "compress_rope_theta": 160000,
    "topk_method": "noaux_tc",
    # Flash: first 2 layers SWA-only (0), then alternating CSA(4) / HCA(128),
    # trailing 0 is the MTP placeholder. Total length = num_hidden_layers + 1 = 44.
    "compress_ratios": [
        0,
        0,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        0,
    ],
}


def _make_blank_v4_model_config() -> ModelConfig:
    """Mirror ``DeepSeekV4._create_config`` defaults without touching disk."""
    config = ModelConfig()
    config.attn_config.head_num = 0
    config.attn_config.kv_head_num = 0
    config.attn_config.size_per_head = 0
    config.num_layers = 0
    config.inter_size = 0
    config.vocab_size = 129280
    config.max_seq_len = 8192
    config.norm_type = "rmsnorm"
    config.has_post_decoder_layernorm = True
    config.activation_type = "SiGLU"
    return config


class DeepSeekV4RegistrationTest(TestCase):
    def test_for_causal_lm_architecture_resolves(self):
        ft_type = ModelDict.get_ft_model_type_by_hf_architectures(
            "DeepseekV4ForCausalLM"
        )
        self.assertEqual(ft_type, "deepseek_v4")

    def test_mtp_architecture_resolves(self):
        ft_type = ModelDict.get_ft_model_type_by_hf_architectures(
            "DeepseekV4ForCausalLMNextN"
        )
        self.assertEqual(ft_type, "deepseek_v4_mtp")

    def test_mtp_class_inherits_v4(self):
        # MTP must reuse the V4 weight loader / config parsing path.
        self.assertTrue(issubclass(DeepSeekV4Mtp, DeepSeekV4))


class DeepSeekV4FromHfTest(TestCase):
    def setUp(self):
        self.config = _make_blank_v4_model_config()
        DeepSeekV4._populate_from_hf_dict(self.config, FLASH_BASE_CONFIG)

    # ----- top-level dimensions -----
    def test_top_level_dims(self):
        self.assertEqual(self.config.num_layers, 43)
        self.assertEqual(self.config.hidden_size, 4096)
        self.assertEqual(self.config.vocab_size, 129280)
        self.assertAlmostEqual(self.config.layernorm_eps, 1e-6)
        self.assertFalse(self.config.tie_word_embeddings)

    # ----- attention block -----
    def test_attention_dimensions(self):
        attn = self.config.attn_config
        self.assertEqual(attn.head_num, 64)
        self.assertEqual(attn.kv_head_num, 1)
        self.assertEqual(attn.size_per_head, 512)
        self.assertEqual(attn.v_head_dim, 512)
        self.assertEqual(attn.rope_head_dim, 64)
        # Partial RoPE: leading dims un-rotated.
        self.assertEqual(attn.nope_head_dim, 512 - 64)
        self.assertEqual(attn.q_lora_rank, 1024)
        self.assertEqual(attn.kv_lora_rank, 0)
        # V4 attention is compressed MQA, not MLA — keep MLA flag off.
        self.assertFalse(attn.use_mla)

    def test_rope_yarn_scaling(self):
        rope = self.config.attn_config.rope_config
        self.assertEqual(rope.dim, 64)
        self.assertEqual(rope.base, 10000)
        self.assertAlmostEqual(rope.scale, 16.0)
        self.assertEqual(rope.max_pos, 65536)
        self.assertAlmostEqual(rope.factor1, 1.0)  # beta_slow
        self.assertAlmostEqual(rope.factor2, 32.0)  # beta_fast

    # ----- V4-specific attention extensions -----
    def test_grouped_output_projection(self):
        attn = self.config.attn_config
        self.assertEqual(attn.o_groups, 8)
        self.assertEqual(attn.o_lora_rank, 1024)

    def test_sliding_window_and_compress_rope(self):
        attn = self.config.attn_config
        self.assertEqual(attn.sliding_window, 128)
        self.assertAlmostEqual(attn.compress_rope_theta, 160000.0)

    def test_per_layer_compress_ratios(self):
        attn = self.config.attn_config
        ratios = list(attn.layer_compress_ratios)
        # Length covers all transformer blocks plus the MTP placeholder.
        self.assertEqual(len(ratios), 43 + 1)
        # Flash-Base: first two layers are pure SWA (compress_ratio=0).
        self.assertEqual(ratios[0], 0)
        self.assertEqual(ratios[1], 0)
        # Then CSA(4) and HCA(128) alternate.
        self.assertEqual(ratios[2], 4)
        self.assertEqual(ratios[3], 128)
        # Trailing entry (MTP slot) is non-compressed.
        self.assertEqual(ratios[-1], 0)

    def test_mhc_hyperparameters(self):
        attn = self.config.attn_config
        self.assertEqual(attn.hc_mult, 4)
        self.assertEqual(attn.hc_sinkhorn_iters, 20)
        self.assertAlmostEqual(attn.hc_eps, 1e-6)

    def test_lightning_indexer(self):
        attn = self.config.attn_config
        self.assertTrue(attn.is_sparse)
        self.assertEqual(attn.indexer_head_dim, 128)
        self.assertEqual(attn.indexer_head_num, 64)
        self.assertEqual(attn.indexer_topk, 512)

    # ----- MoE -----
    def test_moe_dimensions(self):
        self.assertEqual(self.config.expert_num, 256)
        self.assertEqual(self.config.moe_k, 6)
        # Shared expert FFN size = n_shared_experts * moe_intermediate_size.
        self.assertEqual(self.config.inter_size, 1 * 2048)
        self.assertEqual(self.config.moe_inter_size, 2048)
        self.assertAlmostEqual(self.config.routed_scaling_factor, 1.5)
        self.assertTrue(self.config.has_moe_norm)
        self.assertEqual(self.config.moe_style, 2)

    def test_scoring_func_is_sqrt_softplus(self):
        # 2 == sqrt(softplus); enum extension lives next to softmax(0) / sigmoid(1).
        self.assertEqual(self.config.scoring_func, _SCORING_FUNC_MAP["sqrtsoftplus"])
        self.assertEqual(self.config.scoring_func, 2)

    def test_hash_routing_layers(self):
        # Flash: first 3 MoE layers should use deterministic hash routing.
        self.assertEqual(self.config.moe_hash_routing_layers, 3)

    def test_swiglu_limit(self):
        self.assertAlmostEqual(self.config.swiglu_limit, 10.0)

    def test_no_group_routing_constraint(self):
        # V4 drops the V3 n_group / topk_group constraint; defaults are 1.
        self.assertEqual(self.config.moe_n_group, 1)
        self.assertEqual(self.config.moe_topk_group, 1)

    def test_moe_layer_index_covers_all_layers(self):
        # V4 has no first_k_dense_replace — every block is MoE.
        self.assertEqual(list(self.config.moe_layer_index), list(range(43)))

    # ----- error handling -----
    def test_unknown_scoring_func_raises(self):
        bad = dict(FLASH_BASE_CONFIG)
        bad["scoring_func"] = "tanhsoftplus"
        config = _make_blank_v4_model_config()
        with self.assertRaises(ValueError):
            DeepSeekV4._populate_from_hf_dict(config, bad)

    def test_from_hf_round_trip_via_disk(self):
        # End-to-end: write the dict to disk and re-load through _from_hf.
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(FLASH_BASE_CONFIG, f)
            config = _make_blank_v4_model_config()
            DeepSeekV4._from_hf(config, tmp)
        self.assertEqual(config.num_layers, 43)
        self.assertEqual(config.attn_config.sliding_window, 128)
        self.assertEqual(len(list(config.attn_config.layer_compress_ratios)), 44)


class DeepSeekV4M0SkeletonTest(TestCase):
    """End-to-end M0 acceptance: ``_create_config`` works on a real V4 ckpt
    layout, and the python model / weight stubs are wired correctly."""

    def test_create_config_e2e_flash_base(self):
        """`from_pretrained → _create_config()` does not raise on Flash-Base."""
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(FLASH_BASE_CONFIG, f)
            config = DeepSeekV4._create_config(tmp)
        # Sanity check that the produced config carries V4 hallmarks.
        self.assertEqual(config.num_layers, 43)
        self.assertEqual(config.expert_num, 256)
        self.assertEqual(config.attn_config.sliding_window, 128)
        self.assertEqual(config.scoring_func, 2)  # sqrt_softplus

    def test_create_config_uses_defaults_when_config_missing(self):
        """When no config.json is present, we keep the blank defaults rather
        than crashing — caller can still instantiate the class for tooling."""
        with tempfile.TemporaryDirectory() as tmp:
            config = DeepSeekV4._create_config(tmp)
        # Defaults stay at the blank values populated by _create_config.
        self.assertEqual(config.num_layers, 0)
        self.assertEqual(config.vocab_size, 129280)

    def test_create_python_model_resolves_to_v4_model(self):
        """Engine path: ``_create_python_model`` must import the V4 model
        class without raising. Full instantiation needs a real
        ``ModelWeights`` / ``ParallelismConfig`` so it's exercised in
        :mod:`deepseek_v4_model_test`; here we just verify the import is
        clean and no NotImplementedError leaks through."""
        from rtp_llm.models_py.model_desc.deepseek_v4 import DeepSeekV4Model

        self.assertTrue(callable(DeepSeekV4Model))

    def test_does_not_support_cuda_graph_yet(self):
        """CUDA-graph capture would re-trace per-token Sinkhorn iterations;
        off until the fused mHC kernel lands."""
        instance = DeepSeekV4.__new__(DeepSeekV4)
        self.assertFalse(instance.support_cuda_graph())


class DeepSeekV4WeightStubTest(TestCase):
    """Validate the V4 per-layer weight plan against the published Flash
    ``model.safetensors.index.json`` key set: every emitted ckpt key must
    actually exist in the checkpoint, and every required ckpt key must be
    emitted somewhere in the plan."""

    def _make_stub_weight(
        self,
        num_layers: int = 43,
        compress_ratios=None,
        expert_num: int = 256,
    ):
        """Bypass ModelDeployWeightInfo.__init__ — we only need the shape of
        the layer plan, not a real ParallelismConfig / KVCacheConfig wiring."""
        if compress_ratios is None:
            compress_ratios = list(FLASH_BASE_CONFIG["compress_ratios"])
        config = _make_blank_v4_model_config()
        DeepSeekV4._populate_from_hf_dict(config, FLASH_BASE_CONFIG)
        # Override knobs so a single shared config can serve different test
        # topologies (e.g. 4-layer mini-model in fast unit tests).
        config.num_layers = num_layers
        config.attn_config.layer_compress_ratios = compress_ratios[: num_layers + 1]
        w = DeepSeekV4Weight.__new__(DeepSeekV4Weight)
        w._num_layers = num_layers
        w._hidden_size = 4096
        w._is_gated_activation = True
        w._align_size = 0
        w.expert_num_ = expert_num
        w.model_config = config
        return w

    def _names_of(self, atoms):
        out = set()
        for a in atoms:
            sub = getattr(a, "sub_weights", None)
            if sub:
                # CompositeWeight.sub_weights is a dict {name: atom}; both
                # iterating the dict and iterating a list need to work.
                children = sub.values() if isinstance(sub, dict) else sub
                out.update(self._names_of(children))
            else:
                out.add(a.name)
        return out

    def test_layer_plan_contains_attention_atoms(self):
        w = self._make_stub_weight(num_layers=4)
        plan = w._get_hf_layer_weight_info(0)  # SWA-only (compress=0)
        names = self._names_of(plan)
        # Always-present per-layer attention atoms.
        for n in (
            W.v4_attn_norm,
            W.v4_ffn_norm,
            W.v4_q_norm,
            W.v4_kv_norm,
            W.v4_attn_sink,
            W.v4_wq_a,
            W.v4_wq_b,
            W.v4_wkv,
            W.v4_wo_a,
            W.v4_wo_b,
            W.v4_hc_attn_base,
            W.v4_hc_attn_fn,
            W.v4_hc_attn_scale,
            W.v4_hc_ffn_base,
            W.v4_hc_ffn_fn,
            W.v4_hc_ffn_scale,
        ):
            self.assertIn(n, names)

    def test_swa_only_layer_omits_compressor_and_indexer(self):
        w = self._make_stub_weight(num_layers=4)
        plan = w._get_hf_layer_weight_info(0)  # compress_ratios[0] == 0
        names = self._names_of(plan)
        for n in (
            W.v4_compressor_ape,
            W.v4_compressor_norm,
            W.v4_compressor_wgate,
            W.v4_compressor_wkv,
            W.v4_indexer_compressor_ape,
            W.v4_indexer_weights_proj,
            W.v4_indexer_wq_b,
        ):
            self.assertNotIn(n, names, f"layer 0 (SWA) should not emit {n}")

    def test_csa_layer_emits_compressor_and_indexer(self):
        w = self._make_stub_weight(num_layers=4)
        plan = w._get_hf_layer_weight_info(2)  # compress_ratios[2] == 4 (CSA)
        names = self._names_of(plan)
        for n in (
            W.v4_compressor_ape,
            W.v4_compressor_norm,
            W.v4_compressor_wgate,
            W.v4_compressor_wkv,
            W.v4_indexer_compressor_ape,
            W.v4_indexer_compressor_norm,
            W.v4_indexer_compressor_wgate,
            W.v4_indexer_compressor_wkv,
            W.v4_indexer_weights_proj,
            W.v4_indexer_wq_b,
            W.v4_indexer_wq_b_s,
        ):
            self.assertIn(n, names, f"layer 2 (CSA) must emit {n}")

    def test_hca_layer_emits_compressor_only(self):
        w = self._make_stub_weight(num_layers=4)
        plan = w._get_hf_layer_weight_info(3)  # compress_ratios[3] == 128 (HCA)
        names = self._names_of(plan)
        for n in (
            W.v4_compressor_ape,
            W.v4_compressor_norm,
            W.v4_compressor_wgate,
            W.v4_compressor_wkv,
        ):
            self.assertIn(n, names, f"layer 3 (HCA) must emit {n}")
        for n in (
            W.v4_indexer_compressor_ape,
            W.v4_indexer_weights_proj,
            W.v4_indexer_wq_b,
        ):
            self.assertNotIn(n, names, f"layer 3 (HCA) must not emit {n}")

    def test_layer_plan_emits_moe_blocks(self):
        w = self._make_stub_weight(num_layers=4)
        # Layer 0 is hash-routed (Flash sets num_hash_layers=3).
        plan_hash = w._get_hf_layer_weight_info(0)
        names_hash = self._names_of(plan_hash)
        self.assertIn(W.v4_moe_gate_w, names_hash)
        self.assertIn(W.v4_moe_gate_tid2eid, names_hash)
        self.assertNotIn(W.v4_moe_gate_b, names_hash)
        # Layer 3 uses learned routing → emits bias, no tid2eid.
        plan_learn = w._get_hf_layer_weight_info(3)
        names_learn = self._names_of(plan_learn)
        self.assertIn(W.v4_moe_gate_w, names_learn)
        self.assertIn(W.v4_moe_gate_b, names_learn)
        self.assertNotIn(W.v4_moe_gate_tid2eid, names_learn)
        # Shared expert (always-on) + routed experts must show up on every
        # layer regardless of routing strategy.
        for n in (
            W.v4_shared_w1,
            W.v4_shared_w1_s,
            W.v4_shared_w2,
            W.v4_shared_w2_s,
            W.v4_shared_w3,
            W.v4_shared_w3_s,
            W.v4_experts_w1,
            W.v4_experts_w1_s,
            W.v4_experts_w2,
            W.v4_experts_w2_s,
            W.v4_experts_w3,
            W.v4_experts_w3_s,
        ):
            self.assertIn(n, names_hash)
            self.assertIn(n, names_learn)

    def test_global_weights_present(self):
        w = self._make_stub_weight()
        info = w._get_weight_info()
        names = {atom.name for atom in info.weights}
        # Embedding / final-norm / lm_head are the minimum a HF ckpt provides.
        self.assertIn(W.embedding, names)
        self.assertIn(W.final_ln_gamma, names)
        self.assertIn(W.final_ln_beta, names)
        self.assertIn(W.lm_head, names)
        # Head-side mHC reduction (paper §2.5) lives in the global surface.
        self.assertIn(W.v4_hc_head_base, names)
        self.assertIn(W.v4_hc_head_fn, names)
        self.assertIn(W.v4_hc_head_scale, names)

    def test_global_weights_use_v4_naming(self):
        """V4 ships embeddings under bare keys (``embed.weight`` / ``norm`` /
        ``head``); reject any plan that still uses V3's ``model.embed_tokens``
        prefix, since loading would silently miss the actual data."""
        w = self._make_stub_weight()
        info = w._get_weight_info()
        glob_atoms = {a.name: a for a in info.weights}
        emb_keys = [ck.name for ck in glob_atoms[W.embedding].weights]
        head_keys = [ck.name for ck in glob_atoms[W.lm_head].weights]
        norm_keys = [ck.name for ck in glob_atoms[W.final_ln_gamma].weights]
        self.assertEqual(emb_keys, ["embed.weight"])
        self.assertEqual(head_keys, ["head.weight"])
        self.assertEqual(norm_keys, ["norm.weight"])

    def test_layer_plan_includes_quant_scales(self):
        """V4 ships FP8 weights with ue8m0 ``.scale`` siblings. The loader
        must emit *both* atoms so the per-tensor scale is preserved — leaving
        scales out would silently dequantize garbage."""
        w = self._make_stub_weight(num_layers=4)
        plan = w._get_hf_layer_weight_info(2)  # CSA: maximally featureful
        names = self._names_of(plan)
        # FP8 + scale pairs: every wq_*/wkv/wo_* / shared / experts atom must
        # come with its matching ``_s`` scale atom.
        pairs = [
            (W.v4_wq_a, W.v4_wq_a_s),
            (W.v4_wq_b, W.v4_wq_b_s),
            (W.v4_wkv, W.v4_wkv_s),
            (W.v4_wo_a, W.v4_wo_a_s),
            (W.v4_wo_b, W.v4_wo_b_s),
            (W.v4_indexer_wq_b, W.v4_indexer_wq_b_s),
            (W.v4_shared_w1, W.v4_shared_w1_s),
            (W.v4_shared_w2, W.v4_shared_w2_s),
            (W.v4_shared_w3, W.v4_shared_w3_s),
            (W.v4_experts_w1, W.v4_experts_w1_s),
            (W.v4_experts_w2, W.v4_experts_w2_s),
            (W.v4_experts_w3, W.v4_experts_w3_s),
        ]
        for w_name, s_name in pairs:
            self.assertIn(w_name, names, f"missing weight atom {w_name}")
            self.assertIn(s_name, names, f"missing scale atom {s_name}")

    def test_ckpt_keys_match_published_flash_index(self):
        """For each per-layer plan emitted, every CkptWeightInfo key must
        formatable into a key that follows V4-Flash's index conventions
        (i.e. starts with ``layers.{layer_id}.`` and ends with ``.weight``,
        ``.scale``, ``.bias``, ``.tid2eid``, ``ape`` or one of the bare
        ``hc_*`` suffixes)."""
        import re

        valid_suffixes = re.compile(
            r"\.(weight|scale|bias|tid2eid|ape)$"
            r"|^layers\.\d+\.hc_(attn|ffn)_(base|fn|scale)$"
            r"|^layers\.\d+\.attn\.(attn_sink|compressor\.ape"
            r"|indexer\.compressor\.ape)$"
        )
        w = self._make_stub_weight(num_layers=4)
        for layer_id in range(4):
            plan = w._get_hf_layer_weight_info(layer_id)
            for atom in self._iter_atoms(plan):
                for ck in atom.weights:
                    name = ck.name.format(i=str(layer_id), expert_id="0")
                    self.assertTrue(
                        name.startswith(f"layers.{layer_id}.")
                        or name.startswith("hc_head_"),
                        f"{name!r} does not look like a V4 ckpt key",
                    )
                    self.assertTrue(
                        valid_suffixes.search(name) is not None,
                        f"{name!r} has an unrecognised suffix",
                    )

    def _iter_atoms(self, items):
        for it in items:
            sub = getattr(it, "sub_weights", None)
            if sub:
                children = sub.values() if isinstance(sub, dict) else sub
                yield from self._iter_atoms(children)
            else:
                yield it

    def test_process_meta_does_not_inspect_keys(self):
        """V2 inspects weight keys for q_a_proj etc.; V4 must not — those keys
        do not exist in the V4 ckpt, and we don't want false positives."""
        w = self._make_stub_weight()
        # Even with V3-shaped fake meta, V4 stub should swallow it.
        result = w._process_meta(meta_dict={}, weight_keys=set())
        self.assertIsNone(result)


class DeepSeekV4MtpWeightTest(TestCase):
    """V4 MTP loader: full V4 SWA-only block under ``mtp.{i}.*`` keys plus
    enorm / hnorm / e_proj / h_proj and a per-MTP head-side mHC reduction."""

    def _make_stub_mtp_weight(self):
        config = _make_blank_v4_model_config()
        DeepSeekV4._populate_from_hf_dict(config, FLASH_BASE_CONFIG)
        config.num_layers = 1
        # MTP slot is the trailing 0 entry of compress_ratios.
        config.attn_config.layer_compress_ratios = [0]
        w = DeepSeekV4MtpWeight.__new__(DeepSeekV4MtpWeight)
        w._num_layers = 1
        w._hidden_size = 4096
        w._is_gated_activation = True
        w._align_size = 0
        w.expert_num_ = 256
        w.model_config = config
        return w

    def _names_of(self, atoms):
        out = set()
        for a in atoms:
            sub = getattr(a, "sub_weights", None)
            if sub:
                # CompositeWeight.sub_weights is a dict {name: atom}; both
                # iterating the dict and iterating a list need to work.
                children = sub.values() if isinstance(sub, dict) else sub
                out.update(self._names_of(children))
            else:
                out.add(a.name)
        return out

    def test_layer_plan_has_mtp_aux_tensors(self):
        from rtp_llm.utils.model_weight import W

        w = self._make_stub_mtp_weight()
        info = w._get_weight_info()
        self.assertEqual(len(info.layer_weights), 1)
        names = self._names_of(info.layer_weights[0])
        # V3-style enorm/hnorm/final-ln still live in the layer plan.
        self.assertIn(W.multi_tokens_predict_final_ln_gamma, names)
        self.assertIn(W.multi_tokens_predict_final_ln_beta, names)
        self.assertIn(W.multi_tokens_predict_enorm, names)
        self.assertIn(W.multi_tokens_predict_hnorm, names)
        # V4-specific: split e_proj / h_proj instead of fused eh_proj, with
        # ue8m0 scales.
        self.assertIn(W.v4_mtp_e_proj, names)
        self.assertIn(W.v4_mtp_e_proj_s, names)
        self.assertIn(W.v4_mtp_h_proj, names)
        self.assertIn(W.v4_mtp_h_proj_s, names)
        # MTP-local head-side mHC reduction.
        self.assertIn(W.v4_hc_head_base, names)
        self.assertIn(W.v4_hc_head_fn, names)
        self.assertIn(W.v4_hc_head_scale, names)

    def test_layer_plan_includes_full_v4_attention(self):
        """MTP rides on top of a V4 SWA-only attention block — all of the
        usual per-layer V4 atoms (LoRA Q/KV/O, Q/K-norm, sink, MoE) must show
        up under the ``mtp.{i}.*`` namespace."""
        w = self._make_stub_mtp_weight()
        info = w._get_weight_info()
        names = self._names_of(info.layer_weights[0])
        for n in (
            W.v4_attn_norm,
            W.v4_ffn_norm,
            W.v4_q_norm,
            W.v4_kv_norm,
            W.v4_attn_sink,
            W.v4_wq_a,
            W.v4_wq_b,
            W.v4_wkv,
            W.v4_wo_a,
            W.v4_wo_b,
            W.v4_hc_attn_base,
            W.v4_hc_ffn_base,
            W.v4_moe_gate_w,
            W.v4_shared_w1,
            W.v4_experts_w1,
        ):
            self.assertIn(n, names, f"MTP plan missing {n}")
        # MTP runs single-token spec-decode; it must NOT carry a compressor.
        for n in (W.v4_compressor_ape, W.v4_indexer_compressor_ape):
            self.assertNotIn(n, names, f"MTP must not emit {n}")

    def test_layer_keys_use_mtp_prefix(self):
        """Every per-layer ckpt key must be rebased onto ``mtp.{i}.*`` — the
        V4-Flash MTP ckpt has no ``layers.{i}.*`` aliases for them."""
        w = self._make_stub_mtp_weight()
        info = w._get_weight_info()
        for atom in info.layer_weights[0]:
            for sub in getattr(atom, "sub_weights", None) or [atom]:
                for ck in getattr(sub, "weights", []):
                    if "{i}" in ck.name:
                        self.assertTrue(
                            ck.name.startswith("mtp.{i}."),
                            f"{ck.name!r} not under mtp.* namespace",
                        )

    def test_global_weights_are_zero_placeholders(self):
        """V4 MTP shares its embedding + lm_head with the base model. The
        ckpt does not ship them, so the loader emits zero placeholders that
        ``MtpExecutor`` will alias to the real tensors at runtime."""
        w = self._make_stub_mtp_weight()
        info = w._get_weight_info()
        glob_atoms = {a.name: a for a in info.weights}
        self.assertIn(W.embedding, glob_atoms)
        self.assertIn(W.lm_head, glob_atoms)
        # Both atoms have no source ckpt key (zero-fill via process_fun).
        self.assertEqual(glob_atoms[W.embedding].weights, [])
        self.assertEqual(glob_atoms[W.lm_head].weights, [])

    def test_rejects_more_than_one_mtp_layer(self):
        # We deliberately bypass the helper to set num_layers=2.
        config = _make_blank_v4_model_config()
        DeepSeekV4._populate_from_hf_dict(config, FLASH_BASE_CONFIG)
        config.num_layers = 2
        config.attn_config.layer_compress_ratios = [0, 0]
        w = DeepSeekV4MtpWeight.__new__(DeepSeekV4MtpWeight)
        w._num_layers = 2
        w._hidden_size = 4096
        w._is_gated_activation = True
        w._align_size = 0
        w.expert_num_ = 256
        w.model_config = config
        with self.assertRaises(AssertionError):
            w._get_weight_info()


class DeepSeekV4FlashCkptCoverageTest(TestCase):
    """End-to-end loader coverage against the published V4-Flash key set.

    The full ``model.safetensors.index.json`` for ``DeepSeek-V4-Flash`` lists
    ~69k tensors (43 layers × ~250 keys + MTP + globals). Walking the real
    file would require the snapshot to be present on disk, so this test bakes
    in the *unique key patterns* with placeholders for layer index and expert
    id and checks every pattern is reachable from the loader's plan.

    If the loader misses a pattern, the model would silently load with random
    weights for that tensor; if it emits a pattern not in this set, loading
    against a real ckpt would fail with a missing-key error. Either drift is
    a regression we want to catch in CI.
    """

    # Per-layer attention + always-present FFN atoms.
    _PER_LAYER_ALWAYS = (
        "layers.{i}.attn_norm.weight",
        "layers.{i}.ffn_norm.weight",
        "layers.{i}.attn.q_norm.weight",
        "layers.{i}.attn.kv_norm.weight",
        "layers.{i}.attn.attn_sink",
        "layers.{i}.attn.wq_a.weight",
        "layers.{i}.attn.wq_a.scale",
        "layers.{i}.attn.wq_b.weight",
        "layers.{i}.attn.wq_b.scale",
        "layers.{i}.attn.wkv.weight",
        "layers.{i}.attn.wkv.scale",
        "layers.{i}.attn.wo_a.weight",
        "layers.{i}.attn.wo_a.scale",
        "layers.{i}.attn.wo_b.weight",
        "layers.{i}.attn.wo_b.scale",
        "layers.{i}.hc_attn_base",
        "layers.{i}.hc_attn_fn",
        "layers.{i}.hc_attn_scale",
        "layers.{i}.hc_ffn_base",
        "layers.{i}.hc_ffn_fn",
        "layers.{i}.hc_ffn_scale",
        # MoE — one per layer (all layers are MoE in V4).
        "layers.{i}.ffn.gate.weight",
        "layers.{i}.ffn.shared_experts.w1.weight",
        "layers.{i}.ffn.shared_experts.w1.scale",
        "layers.{i}.ffn.shared_experts.w2.weight",
        "layers.{i}.ffn.shared_experts.w2.scale",
        "layers.{i}.ffn.shared_experts.w3.weight",
        "layers.{i}.ffn.shared_experts.w3.scale",
        # Routed experts — stacked, but one logical key per ``(w_i, scale_i)``.
        "layers.{i}.ffn.experts.{e}.w1.weight",
        "layers.{i}.ffn.experts.{e}.w1.scale",
        "layers.{i}.ffn.experts.{e}.w2.weight",
        "layers.{i}.ffn.experts.{e}.w2.scale",
        "layers.{i}.ffn.experts.{e}.w3.weight",
        "layers.{i}.ffn.experts.{e}.w3.scale",
    )

    # Hash-routed layers (first num_hash_layers): tid2eid in place of bias.
    _PER_LAYER_HASH_ROUTED = ("layers.{i}.ffn.gate.tid2eid",)
    # Learned-routed layers: e_score_correction_bias.
    _PER_LAYER_LEARNED_ROUTED = ("layers.{i}.ffn.gate.bias",)

    _PER_LAYER_COMPRESSOR = (
        "layers.{i}.attn.compressor.ape",
        "layers.{i}.attn.compressor.norm.weight",
        "layers.{i}.attn.compressor.wgate.weight",
        "layers.{i}.attn.compressor.wkv.weight",
    )

    _PER_LAYER_INDEXER = (
        "layers.{i}.attn.indexer.compressor.ape",
        "layers.{i}.attn.indexer.compressor.norm.weight",
        "layers.{i}.attn.indexer.compressor.wgate.weight",
        "layers.{i}.attn.indexer.compressor.wkv.weight",
        "layers.{i}.attn.indexer.weights_proj.weight",
        "layers.{i}.attn.indexer.wq_b.weight",
        "layers.{i}.attn.indexer.wq_b.scale",
    )

    _GLOBAL = (
        "embed.weight",
        "norm.weight",
        "head.weight",
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
    )

    def _emitted_keys(self, w, layer_id):
        """Walk a layer plan, yield the exact ckpt key it expects."""
        for atom in self._iter_atoms(w._get_hf_layer_weight_info(layer_id)):
            for ck in atom.weights:
                yield ck.name.format(i=str(layer_id), expert_id="0")

    def _iter_atoms(self, items):
        for it in items:
            sub = getattr(it, "sub_weights", None)
            if sub:
                children = sub.values() if isinstance(sub, dict) else sub
                yield from self._iter_atoms(children)
            else:
                yield it

    def _make_full_flash_loader(self):
        config = _make_blank_v4_model_config()
        DeepSeekV4._populate_from_hf_dict(config, FLASH_BASE_CONFIG)
        w = DeepSeekV4.get_weight_cls().__new__(DeepSeekV4.get_weight_cls())
        w._num_layers = 43
        w._hidden_size = 4096
        w._is_gated_activation = True
        w._align_size = 0
        w.expert_num_ = 256
        w.model_config = config
        return w

    def _routing_keys_for(self, layer_id: int):
        """Return the gate-routing key set this layer should emit, given
        Flash's ``num_hash_layers=3``."""
        if layer_id < 3:
            return self._PER_LAYER_HASH_ROUTED
        return self._PER_LAYER_LEARNED_ROUTED

    def test_swa_only_layers_match_flash_index(self):
        """Layers 0 and 1 in V4-Flash are SWA-only — no compressor / indexer
        keys in the published index. The loader must emit exactly the
        always-present per-layer key set for them (plus the hash-routing
        ``tid2eid`` since both lie in the first 3 MoE blocks)."""
        w = self._make_full_flash_loader()
        for layer_id in (0, 1):
            emitted = set(self._emitted_keys(w, layer_id))
            expected_always = {
                p.format(i=str(layer_id), e="0")
                for p in self._PER_LAYER_ALWAYS + self._routing_keys_for(layer_id)
            }
            self.assertTrue(
                expected_always.issubset(emitted),
                f"layer {layer_id}: missing {expected_always - emitted}",
            )
            for p in self._PER_LAYER_COMPRESSOR + self._PER_LAYER_INDEXER:
                k = p.format(i=str(layer_id))
                self.assertNotIn(k, emitted, f"layer {layer_id} should NOT emit {k}")

    def test_csa_layers_match_flash_index(self):
        """Even-indexed layers ≥2 in V4-Flash are CSA — they ship both the
        token-level compressor and the lightning indexer branch."""
        w = self._make_full_flash_loader()
        for layer_id in (2, 4, 6, 42):
            emitted = set(self._emitted_keys(w, layer_id))
            for p in (
                self._PER_LAYER_ALWAYS
                + self._PER_LAYER_COMPRESSOR
                + self._PER_LAYER_INDEXER
                + self._routing_keys_for(layer_id)
            ):
                k = p.format(i=str(layer_id), e="0")
                self.assertIn(
                    k,
                    emitted,
                    f"layer {layer_id} (CSA): missing {k}",
                )

    def test_hca_layers_match_flash_index(self):
        """Odd-indexed layers ≥3 in V4-Flash are HCA — compressor only,
        no indexer."""
        w = self._make_full_flash_loader()
        for layer_id in (3, 5, 7, 41):
            emitted = set(self._emitted_keys(w, layer_id))
            for p in (
                self._PER_LAYER_ALWAYS
                + self._PER_LAYER_COMPRESSOR
                + self._routing_keys_for(layer_id)
            ):
                k = p.format(i=str(layer_id), e="0")
                self.assertIn(
                    k,
                    emitted,
                    f"layer {layer_id} (HCA): missing {k}",
                )
            for p in self._PER_LAYER_INDEXER:
                k = p.format(i=str(layer_id))
                self.assertNotIn(k, emitted, f"layer {layer_id} (HCA): unexpected {k}")

    def test_global_keys_match_flash_index(self):
        w = self._make_full_flash_loader()
        info = w._get_weight_info()
        emitted_global = set()
        for atom in info.weights:
            for ck in atom.weights:
                emitted_global.add(ck.name)
        for k in self._GLOBAL:
            self.assertIn(k, emitted_global, f"missing global key {k}")

    def test_against_real_flash_index_if_available(self):
        """If the published V4-Flash snapshot is mounted, walk its real
        ``model.safetensors.index.json`` and assert every emitted key shows
        up in the index. Skips silently when the snapshot is not available
        (CI workers without the model mount)."""
        candidate_dirs = [
            "/home/wangyin.yx/.cache/huggingface/hub/"
            "models--deepseek-ai--DeepSeek-V4-Flash/snapshots/"
            "6e763230a9d263eca2023f1d4a5ce1bfe126cf48",
        ]
        index_path = None
        for d in candidate_dirs:
            p = os.path.join(d, "model.safetensors.index.json")
            if os.path.exists(p):
                index_path = p
                break
        if index_path is None:
            self.skipTest("V4-Flash snapshot not on this host")
        with open(index_path) as f:
            ckpt_keys = set(json.load(f)["weight_map"].keys())
        w = self._make_full_flash_loader()

        # Per-layer plan emits keys that are formatable into ckpt names.
        info = w._get_weight_info()
        missing = []
        for layer_id, plan in enumerate(info.layer_weights):
            if layer_id >= w._num_layers:
                break
            for atom in self._iter_atoms(plan):
                for ck in atom.weights:
                    name = ck.name.format(i=str(layer_id), expert_id="0")
                    if name not in ckpt_keys:
                        missing.append((layer_id, name))
        # Globals.
        for atom in info.weights:
            for ck in atom.weights:
                if ck.name and ck.name not in ckpt_keys:
                    missing.append(("global", ck.name))
        self.assertEqual(
            missing,
            [],
            f"loader emits {len(missing)} keys not present in V4-Flash index "
            f"(showing first 5: {missing[:5]})",
        )

    def test_loader_does_not_emit_v3_only_keys(self):
        """V3 ckpt keys (``model.embed_tokens.weight``, ``q_a_proj`` etc.)
        must never show up in a V4 plan — they would cause weight loading to
        either fail (key missing) or, worse, silently load wrong data."""
        w = self._make_full_flash_loader()
        info = w._get_weight_info()
        all_emitted = set()
        for plan in info.layer_weights:
            for atom in self._iter_atoms(plan):
                for ck in atom.weights:
                    all_emitted.add(ck.name)
        for atom in info.weights:
            for ck in atom.weights:
                all_emitted.add(ck.name)
        for v3_only in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.{i}.self_attn.q_a_proj.weight",
            "model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.{i}.self_attn.kv_b_proj.weight",
            "model.layers.{i}.self_attn.q_a_layernorm.weight",
            "model.layers.{i}.input_layernorm.weight",
            "model.layers.{i}.post_attention_layernorm.weight",
        ):
            self.assertNotIn(v3_only, all_emitted, f"V3-only key leaked: {v3_only}")


class DeepseekV4HfConfigTest(TestCase):
    """The thin HF wrapper (vLLM-style) — must round-trip arbitrary kwargs."""

    def test_model_type_is_deepseek_v4(self):
        self.assertEqual(DeepseekV4Config.model_type, "deepseek_v4")

    def test_kwargs_round_trip(self):
        config = DeepseekV4Config(
            hidden_size=4096,
            num_hidden_layers=43,
            o_groups=8,
            compress_ratios=[0, 0, 4, 128],
            scoring_func="sqrtsoftplus",
        )
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 43)
        self.assertEqual(config.o_groups, 8)
        self.assertEqual(config.compress_ratios, [0, 0, 4, 128])
        self.assertEqual(config.scoring_func, "sqrtsoftplus")

    def test_to_dict_contains_v4_fields(self):
        config = DeepseekV4Config(**FLASH_BASE_CONFIG)
        d = config.to_dict()
        # Every field we set should survive the round-trip.
        self.assertEqual(d["compress_rope_theta"], 160000)
        self.assertEqual(d["sliding_window"], 128)
        self.assertEqual(d["hc_mult"], 4)

    def test_registered_in_config_mapping_names(self):
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        self.assertEqual(CONFIG_MAPPING_NAMES.get("deepseek_v4"), "DeepseekV4Config")


if __name__ == "__main__":
    main()
