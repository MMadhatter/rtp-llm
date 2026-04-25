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
from typing import List, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
)
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.deepseek_v2 import DeepSeekV2, DeepSeekV2Weight, DeepSeekV3MtpWeight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    sp_moe_w1,
    stack_,
    transpose,
    yarn_get_mscale,
    zeros,
)

# Per-layer attention type, derived from config.attn_config.layer_compress_ratios.
# Mirrors :class:`models_py.modules.hybrid.cache_topology.LayerCacheKind` —
# kept here as plain ints to avoid pulling the model_desc dep into the loader.
_LAYER_TYPE_NON_CACHE = 0  # MTP placeholder slot (no compress entry generated)
_LAYER_TYPE_CSA = 4  # Compressed Sparse Attention (m = 4)
_LAYER_TYPE_HCA = 128  # Heavily Compressed Attention (m' = 128)
_LAYER_TYPE_SWA_ONLY = -1  # Pure sliding-window (no compressor); compress_ratio == 0


class _V4MoeWeight(MoeWeight):
    """V4-named MoE container.

    The base :class:`MoeWeight` constructor caches three convenience handles
    (``moe_w1`` / ``moe_w2`` / ``moe_gate``) by indexing into ``sub_weights``
    with V3's literal name strings — that hard-coded lookup explodes the
    moment we use V4's ``v4_experts_w*`` keys instead. Override ``__init__``
    to map those handles onto the V4-named atoms while leaving the rest of
    the dispatch / postprocess pipeline untouched.
    """

    def __init__(self, sub_weights, config: MoeConfig, **kwargs):
        # Defer to CompositeWeight directly to skip V3's name lookup. We still
        # enforce that everything is a MoeAtomicWeight so the load-time
        # ``StackSplitTensorSource`` machinery kicks in for stacked experts.
        from rtp_llm.model_loader.weight_module import CompositeWeight, QuantWeight

        self.config = config
        assert all(
            isinstance(sw, MoeAtomicWeight) or isinstance(sw, QuantWeight)
            for sw in sub_weights
        )
        kwargs["name"] = W.moe
        CompositeWeight.__init__(self, sub_weights, **kwargs)
        # V3-style handles re-pointed at V4 atoms. Down-projection (w2) and
        # gate-projection (w1) keep the same mathematical meaning as V3.
        self.moe_w1 = self.sub_weights.get(W.v4_experts_w1)
        self.moe_w2 = self.sub_weights.get(W.v4_experts_w2)
        self.moe_gate = self.sub_weights.get(W.v4_moe_gate_w)


def _classify_layer(compress_ratio: int, layer_id: int, num_layers: int) -> int:
    """Map a compress_ratios entry to one of the four V4 attention kinds.

    The trailing entry (index ``num_layers``) is the MTP slot — always
    NON_CACHE. Layers with compress_ratio==0 inside the transformer body are
    SWA-only (Flash uses this for the first 2 layers). 4 / 128 are CSA / HCA.
    Any other value is rejected — V4 only ships these three compression
    factors and silently mapping unknown ratios would cause hard-to-debug
    weight-key mismatches downstream.
    """
    if layer_id == num_layers:
        return _LAYER_TYPE_NON_CACHE
    if compress_ratio == 0:
        return _LAYER_TYPE_SWA_ONLY
    if compress_ratio == _LAYER_TYPE_CSA:
        return _LAYER_TYPE_CSA
    if compress_ratio == _LAYER_TYPE_HCA:
        return _LAYER_TYPE_HCA
    raise ValueError(
        f"DeepSeek-V4 layer {layer_id}: unsupported compress_ratio "
        f"{compress_ratio}; expected one of {{0, 4, 128}}"
    )


# scoring_func enum: 0 = softmax, 1 = sigmoid, 2 = sqrt(softplus) (DeepSeek-V4)
_SCORING_FUNC_MAP = {
    "softmax": 0,
    "sigmoid": 1,
    "sqrtsoftplus": 2,
}


class DeepSeekV4Weight(DeepSeekV2Weight):
    """V4 per-layer weight loader (PR-F).

    V4's checkpoint layout has *no* overlap with V3's at the per-layer level
    (no ``kv_a_proj_with_mqa``, ``kv_b_proj``, ``q_a_layernorm``, …) — every
    HF key is bespoke. The plan below mirrors the keys observed in
    DeepSeek-V4-Flash's ``model.safetensors.index.json``:

    * **Always per layer**: ``attn_norm`` / ``ffn_norm``, MQA Q-LoRA pair
      (``wq_a`` + ``wq_b``), single MQA KV (``wkv``), Q/K RMSNorm, attn sink,
      grouped-O LoRA (``wo_a`` + ``wo_b``), mHC params for both attn / ffn
      sub-blocks, MoE gate, shared expert, and the 256 routed experts.
    * **CSA + HCA layers only**: token-level compressor (``compressor.{ape,
      norm, wgate, wkv}``).
    * **CSA layers only** (``compress_ratio == 4``): the lightning indexer
      branch (``indexer.compressor.*`` + ``indexer.weights_proj`` +
      ``indexer.wq_b``).

    The branching is driven by ``model_config.attn_config.layer_compress_ratios``
    — the same array PR-E uses to decide which sub-module to instantiate.

    Globals (``embed`` / ``norm`` / ``head``) live under bare keys
    (no ``model.`` prefix) and the head-side mHC reduction is stored as
    ``hc_head_{base, fn, scale}``.
    """

    def _process_meta(self, meta_dict, weight_keys):
        # V2's auto-detection looks for q_a_proj / e_score_correction_bias keys
        # which simply don't exist in a V4 ckpt — running it on V4 data would
        # silently leave the V2 "is this V3?" flags in their default state.
        return

    # ------------------------------------------------------------------
    # Per-layer plan, branching on compress_ratios.
    # ------------------------------------------------------------------
    def _layer_kind(self, layer_id: int) -> int:
        ratios = list(self.model_config.attn_config.layer_compress_ratios)
        if not ratios:
            # Defensive: no compress_ratios configured → treat as HCA (the
            # densest of the compressed variants, so its key set strictly
            # contains SWA-only's).
            return _LAYER_TYPE_HCA
        if layer_id >= len(ratios):
            return _LAYER_TYPE_HCA
        return _classify_layer(ratios[layer_id], layer_id, self._num_layers)

    def _is_hash_routed_layer(self, layer_id: int) -> bool:
        """Per-layer hash-routing predicate.

        The first ``num_hash_layers`` MoE blocks of the *base* model use a
        deterministic ``tid2eid`` lookup; later blocks use learned
        ``noaux_tc`` routing with an ``e_score_correction_bias``. Subclasses
        (e.g. :class:`DeepSeekV4MtpWeight`) override this to opt out — MTP is
        a single-token spec head that always uses learned routing.
        """
        num_hash = int(self.model_config.moe_hash_routing_layers)
        return layer_id < num_hash

    def _get_hf_attn_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        kind = self._layer_kind(layer_id)
        # Tensors emitted by every transformer block, regardless of attention
        # kind. The two LayerNorms straddling the block + the LoRA-decomposed
        # Q / KV / O paths + Q/K RMSNorm + per-head attention sink.
        layer: List[WeightModule] = [
            AtomicWeight(
                W.v4_attn_norm,
                [CkptWeightInfo("layers.{i}.attn_norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_ffn_norm,
                [CkptWeightInfo("layers.{i}.ffn_norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_q_norm,
                [CkptWeightInfo("layers.{i}.attn.q_norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_kv_norm,
                [CkptWeightInfo("layers.{i}.attn.kv_norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_attn_sink,
                [CkptWeightInfo("layers.{i}.attn.attn_sink", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_wq_a,
                [CkptWeightInfo("layers.{i}.attn.wq_a.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_wq_a_s,
                [CkptWeightInfo("layers.{i}.attn.wq_a.scale", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_wq_b,
                [CkptWeightInfo("layers.{i}.attn.wq_b.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_wq_b_s,
                [CkptWeightInfo("layers.{i}.attn.wq_b.scale", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_wkv,
                [CkptWeightInfo("layers.{i}.attn.wkv.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_wkv_s,
                [CkptWeightInfo("layers.{i}.attn.wkv.scale", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_wo_a,
                [CkptWeightInfo("layers.{i}.attn.wo_a.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_wo_a_s,
                [CkptWeightInfo("layers.{i}.attn.wo_a.scale", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_wo_b,
                [CkptWeightInfo("layers.{i}.attn.wo_b.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_wo_b_s,
                [CkptWeightInfo("layers.{i}.attn.wo_b.scale", identity)],
                identity,
            ),
        ]

        # Token-level compressor — only present on layers that actually
        # compress (CSA m=4 or HCA m'=128). SWA-only layers omit it.
        if kind in (_LAYER_TYPE_CSA, _LAYER_TYPE_HCA):
            layer.extend(
                [
                    AtomicWeight(
                        W.v4_compressor_ape,
                        [CkptWeightInfo("layers.{i}.attn.compressor.ape", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.v4_compressor_norm,
                        [
                            CkptWeightInfo(
                                "layers.{i}.attn.compressor.norm.weight",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.v4_compressor_wgate,
                        [
                            CkptWeightInfo(
                                "layers.{i}.attn.compressor.wgate.weight",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.v4_compressor_wkv,
                        [
                            CkptWeightInfo(
                                "layers.{i}.attn.compressor.wkv.weight",
                                identity,
                            )
                        ],
                        identity,
                    ),
                ]
            )

        # Lightning indexer — CSA only.
        if kind == _LAYER_TYPE_CSA:
            layer.extend(
                [
                    AtomicWeight(
                        W.v4_indexer_compressor_ape,
                        [
                            CkptWeightInfo(
                                "layers.{i}.attn.indexer.compressor.ape",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.v4_indexer_compressor_norm,
                        [
                            CkptWeightInfo(
                                "layers.{i}.attn.indexer.compressor.norm.weight",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.v4_indexer_compressor_wgate,
                        [
                            CkptWeightInfo(
                                "layers.{i}.attn.indexer.compressor.wgate.weight",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.v4_indexer_compressor_wkv,
                        [
                            CkptWeightInfo(
                                "layers.{i}.attn.indexer.compressor.wkv.weight",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.v4_indexer_weights_proj,
                        [
                            CkptWeightInfo(
                                "layers.{i}.attn.indexer.weights_proj.weight",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.v4_indexer_wq_b,
                        [
                            CkptWeightInfo(
                                "layers.{i}.attn.indexer.wq_b.weight", identity
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.v4_indexer_wq_b_s,
                        [
                            CkptWeightInfo(
                                "layers.{i}.attn.indexer.wq_b.scale", identity
                            )
                        ],
                        identity,
                    ),
                ]
            )

        # mHC — A/B/C generators for the attention sub-block, then the FFN
        # sub-block. Same shape on every layer regardless of attention kind.
        layer.extend(
            [
                AtomicWeight(
                    W.v4_hc_attn_base,
                    [CkptWeightInfo("layers.{i}.hc_attn_base", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.v4_hc_attn_fn,
                    [CkptWeightInfo("layers.{i}.hc_attn_fn", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.v4_hc_attn_scale,
                    [CkptWeightInfo("layers.{i}.hc_attn_scale", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.v4_hc_ffn_base,
                    [CkptWeightInfo("layers.{i}.hc_ffn_base", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.v4_hc_ffn_fn,
                    [CkptWeightInfo("layers.{i}.hc_ffn_fn", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.v4_hc_ffn_scale,
                    [CkptWeightInfo("layers.{i}.hc_ffn_scale", identity)],
                    identity,
                ),
            ]
        )
        return layer

    def _get_hf_moe_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        # V4 has no first_k_dense_replace — every transformer block is MoE.
        moe_config = MoeConfig(
            align_size=self._align_size,
            expert_num=self.expert_num_,
        )
        ffn_config = FfnConfig(
            align_size=self._align_size,
            is_gated_activation=self._is_gated_activation,
            is_moe=False,
        )

        # Routing gate is split across two checkpoint conventions: the first
        # ``num_hash_layers`` MoE blocks ship a static ``tid2eid`` lookup
        # (deterministic hash routing) and **no** ``e_score_correction_bias``;
        # the rest ship the bias and **no** lookup. Always emit ``gate.weight``.
        is_hash_layer = self._is_hash_routed_layer(layer_id)
        gate_atoms: List[WeightModule] = [
            FfnAtomicWeight(
                W.v4_moe_gate_w,
                [CkptWeightInfo("layers.{i}.ffn.gate.weight", identity)],
                identity,
                config=ffn_config,
            ),
        ]
        if is_hash_layer:
            gate_atoms.append(
                FfnAtomicWeight(
                    W.v4_moe_gate_tid2eid,
                    [CkptWeightInfo("layers.{i}.ffn.gate.tid2eid", identity)],
                    identity,
                    config=ffn_config,
                )
            )
        else:
            gate_atoms.append(
                FfnAtomicWeight(
                    W.v4_moe_gate_b,
                    [CkptWeightInfo("layers.{i}.ffn.gate.bias", identity)],
                    identity,
                    config=ffn_config,
                )
            )

        return [
            FfnWeight(sub_weights=gate_atoms, config=ffn_config),
            # ---- Shared expert (always-on, ungated). Three matrices in the
            # SwiGLU split: w1 = gate, w3 = up, w2 = down. -----------------
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.v4_shared_w1,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.shared_experts.w1.weight",
                                identity,
                            )
                        ],
                        identity,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.v4_shared_w1_s,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.shared_experts.w1.scale",
                                identity,
                            )
                        ],
                        identity,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.v4_shared_w2,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.shared_experts.w2.weight",
                                identity,
                            )
                        ],
                        identity,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.v4_shared_w2_s,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.shared_experts.w2.scale",
                                identity,
                            )
                        ],
                        identity,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.v4_shared_w3,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.shared_experts.w3.weight",
                                identity,
                            )
                        ],
                        identity,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.v4_shared_w3_s,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.shared_experts.w3.scale",
                                identity,
                            )
                        ],
                        identity,
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            ),
            # ---- Routed experts. Each weight stacks all 256 experts along
            # an extra leading dim — MoeAtomicWeight resolves ``{expert_id}``
            # to the per-expert ckpt key at load time and merges them. -----
            _V4MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.v4_experts_w1,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.experts.{expert_id}.w1.weight",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.v4_experts_w1_s,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.experts.{expert_id}.w1.scale",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.v4_experts_w2,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.experts.{expert_id}.w2.weight",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.v4_experts_w2_s,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.experts.{expert_id}.w2.scale",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.v4_experts_w3,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.experts.{expert_id}.w3.weight",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.v4_experts_w3_s,
                        [
                            CkptWeightInfo(
                                "layers.{i}.ffn.experts.{expert_id}.w3.scale",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                ],
                config=moe_config,
            ),
        ]

    def _get_hf_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        plan: List[WeightModule] = []
        plan.extend(self._get_hf_attn_layer_weight_info(layer_id))
        plan.extend(self._get_hf_moe_layer_weight_info(layer_id))
        return plan

    def _get_weight_info(self):
        # DeepSeek-V4 ckpt key conventions (different from V2/V3):
        #   embed.weight   — token embedding (not model.embed_tokens.weight)
        #   norm.weight    — final RMSNorm  (not model.norm.weight)
        #   head.weight    — lm_head        (not lm_head.weight)
        #   hc_head_*      — head-side mHC reduction (paper §2.5; folds the
        #                    n_hc residual streams back into a single hidden
        #                    state before lm_head).
        layer_weights: List[List[WeightModule]] = [
            self._get_hf_layer_weight_info(layer_id)
            for layer_id in range(self._num_layers)
        ]
        weights: List[WeightModule] = [
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
            AtomicWeight(
                W.v4_hc_head_base,
                [CkptWeightInfo("hc_head_base", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_hc_head_fn,
                [CkptWeightInfo("hc_head_fn", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_hc_head_scale,
                [CkptWeightInfo("hc_head_scale", identity)],
                identity,
            ),
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class DeepSeekV4MtpWeight(DeepSeekV4Weight):
    """V4 MTP (next-N) weight loader.

    V4 MTP is a *full* V4 decoder block (attn + MoE + mHC for both sub-blocks,
    plus a head-side mHC reduction local to this MTP layer) wrapped with the
    V3-style ``enorm`` / ``hnorm`` / projection pair that combines the next
    token's embedding with the previous hidden state. Unlike V3 the embedding
    and hidden projections are **separate** (``e_proj`` + ``h_proj``) instead
    of fused into a single ``eh_proj``, so we load them independently.

    V4-Flash MTP-only ckpt key prefix is ``mtp.{i}.*`` (not
    ``model.layers.{i}.*`` like V3). MTP layers do not run a compressor —
    spec-decode operates on a single emitted token at a time.

    The published Flash MTP block does *not* ship an embedding table or
    ``head.weight`` — they are tied to the base model. The loader synthesises
    zero-shaped placeholders so :class:`ModelWeights` does not raise; the
    consumer (``MtpExecutor``) is expected to alias the base model's tables.
    """

    def _layer_kind(self, layer_id: int) -> int:
        # MTP is single-token: no compressor regardless of compress_ratios.
        return _LAYER_TYPE_SWA_ONLY

    def _is_hash_routed_layer(self, layer_id: int) -> bool:
        # MTP never uses hash routing — it ships ``ffn.gate.bias`` instead
        # of ``ffn.gate.tid2eid``.
        return False

    def _format_layer_keys(
        self, items: List[WeightModule], prefix: str
    ) -> List[WeightModule]:
        """Rewrite each per-layer ckpt key to live under ``{prefix}.{{i}}.*``
        instead of the base ``layers.{i}.*`` namespace V4 base layers use.

        ``sub_weights`` may be either a list (raw plan) or the dict that
        ``CompositeWeight.__init__`` builds from one — we walk both shapes
        so the rewrite covers FFN / MoE composites as well.
        """

        def visit(node):
            ck_list = getattr(node, "weights", None)
            if ck_list:
                for ck in ck_list:
                    if ck.name.startswith("layers.{i}."):
                        ck.name = prefix + "." + ck.name[len("layers.") :]
            sub = getattr(node, "sub_weights", None)
            if sub:
                children = sub.values() if isinstance(sub, dict) else sub
                for child in children:
                    visit(child)

        for item in items:
            visit(item)
        return items

    def _get_hf_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        # Re-use the SWA-only branch of the base layer plan (no compressor)
        # then rebase every key onto ``mtp.{i}.*`` and add MTP-specific atoms.
        plan = list(super()._get_hf_layer_weight_info(layer_id))
        plan = self._format_layer_keys(plan, "mtp")
        plan.extend(
            [
                # Final RMSNorm of the MTP block.
                AtomicWeight(
                    W.multi_tokens_predict_final_ln_gamma,
                    [CkptWeightInfo("mtp.{i}.norm.weight", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.multi_tokens_predict_final_ln_beta,
                    [],
                    functools.partial(zeros, shape=[self._hidden_size]),
                ),
                # V3-style embedding / hidden RMSNorms applied before the
                # projection that fuses next-token embedding with prev hidden.
                AtomicWeight(
                    W.multi_tokens_predict_enorm,
                    [CkptWeightInfo("mtp.{i}.enorm.weight", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.multi_tokens_predict_hnorm,
                    [CkptWeightInfo("mtp.{i}.hnorm.weight", identity)],
                    identity,
                ),
                # V4-specific: separate e_proj / h_proj instead of V3's fused
                # eh_proj. Each is FP8 with a ue8m0 scale.
                AtomicWeight(
                    W.v4_mtp_e_proj,
                    [CkptWeightInfo("mtp.{i}.e_proj.weight", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.v4_mtp_e_proj_s,
                    [CkptWeightInfo("mtp.{i}.e_proj.scale", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.v4_mtp_h_proj,
                    [CkptWeightInfo("mtp.{i}.h_proj.weight", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.v4_mtp_h_proj_s,
                    [CkptWeightInfo("mtp.{i}.h_proj.scale", identity)],
                    identity,
                ),
                # MTP-local mHC head-side reduction (the MTP layer is "spec
                # head" and folds the residual stream itself before lm_head).
                AtomicWeight(
                    W.v4_hc_head_base,
                    [CkptWeightInfo("mtp.{i}.hc_head_base", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.v4_hc_head_fn,
                    [CkptWeightInfo("mtp.{i}.hc_head_fn", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.v4_hc_head_scale,
                    [CkptWeightInfo("mtp.{i}.hc_head_scale", identity)],
                    identity,
                ),
            ]
        )
        return plan

    def _get_weight_info(self):
        assert self._num_layers == 1, (
            f"DeepSeekV4MtpWeight expects exactly 1 MTP layer, "
            f"got {self._num_layers}"
        )
        layer_weights: List[List[WeightModule]] = [
            self._get_hf_layer_weight_info(layer_id)
            for layer_id in range(self._num_layers)
        ]
        # MTP module shares its embedding + lm_head with the base model. The
        # ckpt does not ship them, so synthesise zero placeholders — consumers
        # (``MtpExecutor``) are responsible for aliasing the real tensors.
        vocab_size = int(self.model_config.vocab_size)
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [],
                functools.partial(zeros, shape=[vocab_size, self._hidden_size]),
            ),
            AtomicWeight(
                W.lm_head,
                [],
                functools.partial(zeros, shape=[vocab_size, self._hidden_size]),
            ),
        ]
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
        # Engine wiring for the V4 forward path. See
        # ``rtp_llm.models_py.model_desc.deepseek_v4`` for the model class —
        # it dispatches per layer on ``compress_ratios`` to either CSA or HCA
        # (with SWA-only collapsed onto HCA m'=1) and wraps each block in
        # mHC. KV cache integration is mocked (stateless re-prefill) until
        # the M4 heterogeneous KV cache lands; decode requests will work but
        # without prefix-cache savings.
        from rtp_llm.models_py.model_desc.deepseek_v4 import DeepSeekV4Model

        self.py_model = DeepSeekV4Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

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
