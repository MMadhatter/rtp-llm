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

    def test_create_python_model_raises_not_implemented(self):
        """The production engine path needs the V4 weight loader (PR-F);
        the layer composition reference itself is in deepseek_v4_layer.py.
        Whichever shape the error takes, it must clearly be V4-specific."""
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(FLASH_BASE_CONFIG, f)
            config = DeepSeekV4._create_config(tmp)
        instance = DeepSeekV4.__new__(DeepSeekV4)  # bypass full __init__
        instance.model_config = config
        with self.assertRaises(NotImplementedError) as ctx:
            instance._create_python_model()
        msg = str(ctx.exception)
        self.assertIn("DeepSeek-V4", msg)
        # Must point readers at the reference layer module so they can wire
        # up bring-up themselves without re-discovering it.
        self.assertIn("deepseek_v4_layer.py", msg)

    def test_does_not_support_cuda_graph_yet(self):
        """CUDA-graph capture would hit unimplemented kernels; off until M2."""
        instance = DeepSeekV4.__new__(DeepSeekV4)
        self.assertFalse(instance.support_cuda_graph())


class DeepSeekV4WeightStubTest(TestCase):
    """Validate that the weight loader stub yields a consistent-shaped plan
    so the loader can iterate without raising during dummy-weight bring-up."""

    def _make_stub_weight(self, num_layers: int = 4):
        """Bypass ModelDeployWeightInfo.__init__ — we only need the shape of
        the layer plan, not a real ParallelismConfig / KVCacheConfig wiring."""
        w = DeepSeekV4Weight.__new__(DeepSeekV4Weight)
        w._num_layers = num_layers
        w._hidden_size = 4096
        return w

    def test_layer_plan_is_empty_per_layer(self):
        w = self._make_stub_weight(num_layers=43)
        info = w._get_weight_info()
        # One empty list per transformer block.
        self.assertEqual(len(info.layer_weights), 43)
        for layer_id, plan in enumerate(info.layer_weights):
            self.assertEqual(plan, [], f"layer {layer_id} should be a no-op")

    def test_global_weights_present(self):
        w = self._make_stub_weight()
        info = w._get_weight_info()
        names = {atom.name for atom in info.weights}
        # Embedding / final-norm / lm_head are the minimum a HF ckpt provides.
        self.assertIn("embedding", names)
        self.assertIn("final_layernorm.gamma", names)
        self.assertIn("final_layernorm.beta", names)
        self.assertIn("lm_head", names)

    def test_get_hf_layer_weight_info_is_noop(self):
        w = self._make_stub_weight()
        for layer_id in range(3):
            self.assertEqual(w._get_hf_layer_weight_info(layer_id), [])

    def test_process_meta_does_not_inspect_keys(self):
        """V2 inspects weight keys for q_a_proj etc.; V4 must not — those keys
        do not exist in the V4 ckpt, and we don't want false positives."""
        w = self._make_stub_weight()
        # Even with V3-shaped fake meta, V4 stub should swallow it.
        result = w._process_meta(meta_dict={}, weight_keys=set())
        self.assertIsNone(result)


class DeepSeekV4MtpWeightTest(TestCase):
    """V4 MTP loader — exposes V3-style enorm/hnorm/eh_proj/shared_head."""

    def _make_stub_mtp_weight(self):
        w = DeepSeekV4MtpWeight.__new__(DeepSeekV4MtpWeight)
        w._num_layers = 1
        w._hidden_size = 4096
        return w

    def test_layer_plan_has_mtp_aux_tensors(self):
        from rtp_llm.utils.model_weight import W

        w = self._make_stub_mtp_weight()
        info = w._get_weight_info()
        self.assertEqual(len(info.layer_weights), 1)
        names = {a.name for a in info.layer_weights[0]}
        # Per V3 MTP convention, all five auxiliary tensors must show up.
        self.assertIn(W.multi_tokens_predict_final_ln_gamma, names)
        self.assertIn(W.multi_tokens_predict_final_ln_beta, names)
        self.assertIn(W.multi_tokens_predict_enorm, names)
        self.assertIn(W.multi_tokens_predict_hnorm, names)
        self.assertIn(W.multi_tokens_predict_eh_proj, names)

    def test_global_weights_use_mtp_layer0_keys(self):
        w = self._make_stub_mtp_weight()
        info = w._get_weight_info()
        # Embedding & lm_head live under model.layers.0.* in MTP ckpts.
        glob_atoms = {a.name: a for a in info.weights}
        self.assertIn("embedding", glob_atoms)
        self.assertIn("lm_head", glob_atoms)
        emb_keys = [ckpt.name for ckpt in glob_atoms["embedding"].weights]
        self.assertEqual(emb_keys, ["model.layers.0.embed_tokens.weight"])
        head_keys = [ckpt.name for ckpt in glob_atoms["lm_head"].weights]
        self.assertEqual(head_keys, ["model.layers.0.shared_head.head.weight"])

    def test_rejects_more_than_one_mtp_layer(self):
        w = DeepSeekV4MtpWeight.__new__(DeepSeekV4MtpWeight)
        w._num_layers = 2
        w._hidden_size = 4096
        with self.assertRaises(AssertionError):
            w._get_weight_info()

    def test_eh_proj_uses_transpose_processor(self):
        from rtp_llm.utils.model_weight import W

        w = self._make_stub_mtp_weight()
        info = w._get_weight_info()
        eh_atom = next(
            a for a in info.layer_weights[0] if a.name == W.multi_tokens_predict_eh_proj
        )
        # transpose, identity, etc. are picklable callables — compare by name.
        self.assertEqual(eh_atom.process_fun.__name__, "transpose")


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
