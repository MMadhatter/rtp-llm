"""DeepSeek-V4 model registration & HF config parsing (PR-A / M0 skeleton).

Only the model-loading / config-parsing surface is implemented here. Forward
computation, weight remapping, KV-cache topology, mHC residual, CSA/HCA
attention and MoE changes land in subsequent PRs (see develop_ds_v4.md M0–M7).

M0 acceptance:
  * ``DeepSeekV4._create_config(ckpt_path)`` succeeds for any V4 config.json.
  * Model class is registered under the ``DeepseekV4ForCausalLM`` HF
    architecture (``DeepseekV4ForCausalLMNextN`` for the MTP module).
  * Weight loader returns a no-op layer plan, so weight loading runs to
    completion against ckpts containing only embedding / final_norm / lm_head.
  * ``_create_python_model`` raises a descriptive NotImplementedError that
    points at the milestone responsible for filling it in.
"""

import functools
import json
import logging
import os
from typing import List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.deepseek_v2 import DeepSeekV2, DeepSeekV2Weight, DeepSeekV3MtpWeight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    transpose,
    yarn_get_mscale,
    zeros,
)

# scoring_func enum: 0 = softmax, 1 = sigmoid, 2 = sqrt(softplus) (DeepSeek-V4)
_SCORING_FUNC_MAP = {
    "softmax": 0,
    "sigmoid": 1,
    "sqrtsoftplus": 2,
}


class DeepSeekV4Weight(DeepSeekV2Weight):
    """M0 stub weight loader.

    V4's per-layer weight key set is incompatible with V3 (no
    ``kv_a_proj_with_mqa`` / ``kv_b_proj``; new mHC, indexer, grouped-O, ...).
    Until PR-E lands the real layer plan, we expose only the global weights
    (embedding, final norm, lm_head) and an empty list per layer. This lets
    the loader run to completion against ckpts that only contain those keys
    (e.g. random/dummy weights for engine bring-up). Per-layer loading is a
    no-op; downstream forward will fail because ``_create_python_model``
    raises ``NotImplementedError``.
    """

    def _process_meta(self, meta_dict, weight_keys):
        # V2's auto-detection looks for q_a_proj / e_score_correction_bias keys
        # whose presence we cannot guarantee in a V4 ckpt; skip it for M0.
        return

    def _get_hf_layer_weight_info(self, layer_id: int):
        # No per-layer plan yet; PR-E (HCA-only model) will populate.
        return []

    def _get_weight_info(self):
        # DeepSeek-V4 ckpt key conventions (different from V2/V3):
        #   embed.weight   — token embedding (not model.embed_tokens.weight)
        #   norm.weight    — final RMSNorm  (not model.norm.weight)
        #   head.weight    — lm_head        (not lm_head.weight)
        #   hc_head_*      — head-side mHC params (loaded by the python model,
        #                    not part of the global ModelWeights surface)
        layer_weights: List[List[AtomicWeight]] = [[] for _ in range(self._num_layers)]
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("embed.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
            AtomicWeight(
                W.lm_head, [CkptWeightInfo("head.weight", identity)], identity
            ),
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class DeepSeekV4MtpWeight(DeepSeekV4Weight):
    """V4 MTP (next-N) weight loader.

    The MTP module's *non-attention* surface (enorm / hnorm / eh_proj /
    shared_head) is unchanged from V3 — it's a thin wrapper that combines the
    embedding of the next predicted token with the previous hidden state. We
    inherit the V4 base weight stub (so the per-layer attention plan stays
    empty until PR-E lands the real CSA/HCA loader) and bolt the MTP-specific
    auxiliary tensors onto each MTP layer.

    Layout matches :class:`DeepSeekV3MtpWeight` ckpt-key conventions:

      * ``model.layers.0.embed_tokens.weight`` → embedding table
      * ``model.layers.0.shared_head.head.weight`` → lm_head
      * ``model.layers.{i}.shared_head.norm.weight`` → final layernorm gamma
      * ``model.layers.{i}.enorm.weight`` → embedding-side RMSNorm gamma
      * ``model.layers.{i}.hnorm.weight`` → hidden-side RMSNorm gamma
      * ``model.layers.{i}.eh_proj.weight`` → fused (e ⊕ h) projection (transposed)
    """

    def _get_weight_info(self):
        layer_weights: List[List[WeightModule]] = []
        # MTP shares its embedding & lm_head with the MTP layer-0 ckpt.
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.layers.0.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("model.layers.0.shared_head.head.weight", identity)],
                identity,
            ),
        ]
        assert (
            self._num_layers == 1
        ), f"DeepSeekV4MtpWeight expects exactly 1 MTP layer, got {self._num_layers}"
        for layer in range(self._num_layers):
            # Per-layer attention plan still empty (filled by PR-E). The MTP
            # auxiliary tensors below are the *only* per-layer weights we need
            # in order for the loader to walk the ckpt without erroring.
            layer_plan = list(self._get_hf_layer_weight_info(layer))
            layer_plan.extend(
                [
                    AtomicWeight(
                        W.multi_tokens_predict_final_ln_gamma,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.shared_head.norm.weight",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_final_ln_beta,
                        [],
                        functools.partial(zeros, shape=[self._hidden_size]),
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_enorm,
                        [CkptWeightInfo("model.layers.{i}.enorm.weight", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_hnorm,
                        [CkptWeightInfo("model.layers.{i}.hnorm.weight", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_eh_proj,
                        [CkptWeightInfo("model.layers.{i}.eh_proj.weight", identity)],
                        transpose,
                    ),
                ]
            )
            layer_weights.append(layer_plan)
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class DeepSeekV4(DeepSeekV2):
    """DeepSeek-V4 base model.

    Inherits the V2 weight loader scaffolding but supplies its own HF parser
    because V4's architecture (MQA + CSA/HCA compression + mHC + sqrt-softplus
    routing + grouped output projection) does not share field names with V2/V3.
    """

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config.attn_config.head_num = 0
        config.attn_config.kv_head_num = 0
        config.attn_config.size_per_head = 0
        config.num_layers = 0
        config.inter_size = 0
        config.vocab_size = 129280  # V4 default
        config.max_seq_len = 8192
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        config.activation_type = "SiGLU"
        DeepSeekV4._from_hf(config, ckpt_path)
        return config

    def support_cuda_graph(self) -> bool:
        # CSA/HCA backends are not in place yet; CUDA-graph capture would hit
        # NotImplemented paths during warmup. Re-enable from M2 onwards.
        return False

    def _create_python_model(self):
        # PR-E reference layer composition lives in
        # ``rtp_llm.models_py.model_desc.deepseek_v4_layer``. Wiring it into
        # the production ``GenericMoeModel`` pipeline requires the V4-specific
        # weight-key loader (per-layer plan: W_DQ / W_UQ / W_KV / W_Z /
        # bias_pos / sink_logits / grouped-O / mHC params / hash router).
        # Until that loader lands (PR-F per develop_ds_v4.md §8), the engine
        # path raises so requests fail fast with a useful pointer instead of
        # silently using V3 weights.
        raise NotImplementedError(
            "DeepSeek-V4 production forward path needs the V4 per-layer "
            "weight loader. See "
            "rtp_llm/models_py/model_desc/deepseek_v4_layer.py for the "
            "reference layer composition (mHC + HCA attention + V4 MoE) "
            "and develop_ds_v4.md §6 for the milestone breakdown."
        )

    @staticmethod
    def get_weight_cls():
        return DeepSeekV4Weight

    @staticmethod
    def _from_hf(config: ModelConfig, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            logging.warning(
                "DeepSeekV4._from_hf: config.json not found at %s, skipping",
                config_path,
            )
            return
        with open(config_path) as reader:
            config_json = json.loads(reader.read())
        DeepSeekV4._populate_from_hf_dict(config, config_json)

    @staticmethod
    def _populate_from_hf_dict(config: ModelConfig, config_json: dict):
        # ---------- top-level model dimensions ----------
        config.num_layers = int(config_json["num_hidden_layers"])
        config.hidden_size = int(config_json["hidden_size"])
        config.vocab_size = int(config_json["vocab_size"])
        config.layernorm_eps = float(config_json.get("rms_norm_eps", 1e-6))
        config.tie_word_embeddings = bool(config_json.get("tie_word_embeddings", False))
        config.config_dtype = config_json.get("torch_dtype", None)

        # ---------- attention block ----------
        attn = config.attn_config
        attn.head_num = int(config_json["num_attention_heads"])
        # V4 uses MQA: a single shared KV head.
        attn.kv_head_num = int(config_json.get("num_key_value_heads", 1))
        head_dim = int(config_json["head_dim"])
        rope_head_dim = int(config_json["qk_rope_head_dim"])
        attn.size_per_head = head_dim
        attn.v_head_dim = head_dim
        attn.rope_head_dim = rope_head_dim
        # Partial RoPE: only the trailing rope_head_dim of Q/K is rotated.
        attn.nope_head_dim = head_dim - rope_head_dim

        # Q lora (V4 keeps q_lora_rank like V3, but no kv_lora; KV is single MQA head).
        q_lora_rank = config_json.get("q_lora_rank")
        attn.q_lora_rank = int(q_lora_rank) if q_lora_rank is not None else 0
        attn.kv_lora_rank = 0

        # V4 attention is "compressed MQA", not MLA — leave use_mla off.
        # The CSA/HCA backends consume the V4-specific fields below.
        attn.use_mla = False

        # ---------- RoPE (yarn, partial) ----------
        attn.rope_config.dim = rope_head_dim
        attn.rope_config.base = int(
            config_json.get("rope_theta", attn.rope_config.base)
        )
        attn.rope_config.offset = attn.nope_head_dim
        attn.rope_config.is_neox_style = not config_json.get("rope_interleave", True)

        rope_scaling = config_json.get("rope_scaling")
        if rope_scaling is not None:
            attn.rope_config.scale = float(rope_scaling["factor"])
            attn.rope_config.factor1 = float(rope_scaling.get("beta_slow", 1))
            attn.rope_config.factor2 = float(rope_scaling.get("beta_fast", 32))
            attn.rope_config.max_pos = int(
                rope_scaling["original_max_position_embeddings"]
            )
            # V4's HF config does not ship explicit mscale / mscale_all_dim;
            # default both to 1.0 so yarn_get_mscale falls back to 1.0 scaling.
            mscale = float(rope_scaling.get("mscale", 1.0))
            mscale_all_dim = float(rope_scaling.get("mscale_all_dim", 1.0))
            config.deepseek_rope_mscale = mscale
            config.deepseek_mscale_all_dim = mscale_all_dim
            scaling_factor = attn.rope_config.scale
            attn.rope_config.mscale = yarn_get_mscale(
                scaling_factor, mscale
            ) / yarn_get_mscale(scaling_factor, mscale_all_dim)
            softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            attn.softmax_extra_scale = softmax_mscale * softmax_mscale

        # ---------- V4-specific attention extensions ----------
        # Grouped Output Projection: n_h heads -> o_groups groups -> o_lora_rank.
        attn.o_groups = int(config_json.get("o_groups", 0))
        attn.o_lora_rank = int(config_json.get("o_lora_rank", 0))

        # SWA bypass window
        attn.sliding_window = int(config_json.get("sliding_window", 0))

        # Independent RoPE base for the compressed KV branch.
        attn.compress_rope_theta = float(config_json.get("compress_rope_theta", 0.0))

        # Per-layer compressed-attention type table.
        compress_ratios: List[int] = list(config_json.get("compress_ratios", []))
        attn.layer_compress_ratios = [int(x) for x in compress_ratios]

        # Manifold-Constrained Hyper-Connections (mHC) hyperparams.
        attn.hc_mult = int(config_json.get("hc_mult", 0))
        attn.hc_sinkhorn_iters = int(config_json.get("hc_sinkhorn_iters", 0))
        attn.hc_eps = float(config_json.get("hc_eps", 1e-6))

        # Lightning indexer (CSA only). Re-using V3.2 sparse indexer fields.
        if config_json.get("index_topk") is not None:
            attn.is_sparse = True
            attn.indexer_head_dim = int(config_json["index_head_dim"])
            attn.indexer_head_num = int(config_json["index_n_heads"])
            attn.indexer_topk = int(config_json["index_topk"])

        # ---------- MoE ----------
        scoring_func_str = config_json.get("scoring_func", "softmax")
        if scoring_func_str not in _SCORING_FUNC_MAP:
            raise ValueError(
                f"DeepSeek-V4: unsupported scoring_func '{scoring_func_str}', "
                f"expected one of {list(_SCORING_FUNC_MAP)}"
            )
        config.scoring_func = _SCORING_FUNC_MAP[scoring_func_str]

        config.routed_scaling_factor = float(config_json["routed_scaling_factor"])
        config.moe_k = int(config_json["num_experts_per_tok"])
        config.expert_num = int(config_json["n_routed_experts"])
        moe_intermediate_size = int(config_json["moe_intermediate_size"])
        config.moe_inter_size = moe_intermediate_size
        n_shared_experts = int(config_json.get("n_shared_experts", 0))
        # Shared-expert FFN size — mirrors DeepSeekV2._from_hf's convention.
        config.inter_size = n_shared_experts * moe_intermediate_size
        config.has_moe_norm = bool(config_json.get("norm_topk_prob", False))
        config.moe_style = 2  # shared + routed

        # V4 drops the V3 group-routing constraint (no n_group / topk_group).
        config.moe_n_group = int(config_json.get("n_group", 1))
        config.moe_topk_group = int(config_json.get("topk_group", 1))

        # V4 has no first_k_dense_replace: every transformer block is MoE.
        # Layers whose compress_ratios entry is 0 (e.g. MTP placeholder) are
        # filtered out at the layer-build stage, not here.
        config.moe_layer_index = list(range(config.num_layers))

        # First num_hash_layers MoE layers use deterministic hash routing.
        config.moe_hash_routing_layers = int(config_json.get("num_hash_layers", 0))

        # SwiGLU clamp bound (0 means disabled).
        config.swiglu_limit = float(config_json.get("swiglu_limit", 0.0))


class DeepSeekV4Mtp(DeepSeekV4):
    """DeepSeek-V4 next-N (MTP) module."""

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        # MTP is exactly one extra layer; mark all of its layers as MoE.
        config.moe_layer_index = list(range(config.num_layers))
        config.reverse_e_h_norm = True
        config.is_mtp = True
        return config

    @staticmethod
    def get_weight_cls():
        return DeepSeekV4MtpWeight


register_model("deepseek_v4", DeepSeekV4, ["DeepseekV4ForCausalLM"])
register_model("deepseek_v4_mtp", DeepSeekV4Mtp, ["DeepseekV4ForCausalLMNextN"])
