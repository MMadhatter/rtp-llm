"""DeepSeek-V4 engine model wiring.

Bridges the V4 reference building blocks (mHC + CSA / HCA / SWA-only attention
+ V4 MoE) to the :class:`GptModelBase` contract that ``rtp_llm``'s engine
expects. End-to-end runnable for *prefill*; the decode path needs the
heterogeneous KV cache (M4) which is mocked here as a stateless re-prefill.

Design notes
------------
* mHC's residual stream lives at width ``hc_mult * hidden_size``. The engine
  framework hands us a ``(B, T, hidden_size)`` activation, so we expand at the
  embedding boundary and reduce before the final norm + LM head (paper §2.5).
* Per-layer attention kind is decided by ``compress_ratios[layer_id]`` —
  classic dispatch: 0 → SWA-only, 4 → CSA, 128 → HCA. SWA-only collapses to
  HCA with ``m'=1`` for the reference path so the same compressor / attn-sink
  / grouped-O code paths cover all three.
* FP8 weights from the V4-Flash checkpoint are dequantised on the fly the
  first time they are bound to a reference module — full per-block ue8m0
  dequant is one-shot, so the runtime hot path stays bf16.

References
----------
* vLLM PR #40760 (`vllm-project/vllm`, branch `dsv4`) — model layer + fused
  kernels, esp. ``vllm/model_executor/layers/deepseek_v4_attention.py`` and
  ``csrc/moe/topk_softplus_sqrt_kernels.cu``.
* SGLang PR #23600 (`sgl-project/sglang`, branch `deepseek_v4`) — model layer
  + state cache coordinator, esp.
  ``python/sglang/srt/models/deepseek_v4.py`` and
  ``python/sglang/srt/layers/attention/compressed/``.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.deepseek_v4_layer import DeepSeekV4MoE
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import Embedding, RMSNorm, RMSNormTorch
from rtp_llm.models_py.modules.hybrid.csa_attention import CsaAttention

# ----- Production-path selectors -----
# Cached at import time; the `_pick_indexer_impl()` / `_pick_csa_impl()` /
# `_pick_moe_impl()` helpers below decide which to use given the running
# hardware. The reference Python paths remain importable so CPU-only unit
# tests still resolve the symbols when sm_100 / DeepGEMM are absent.
from rtp_llm.models_py.modules.hybrid.deepgemm_indexer import (
    _HAS_DEEPGEMM as _HAS_DEEPGEMM_INDEXER,
)
from rtp_llm.models_py.modules.hybrid.deepgemm_indexer import (
    deepgemm_indexer_score_topk,
)
from rtp_llm.models_py.modules.hybrid.fp4_indexer import fp4_indexer_score_topk
from rtp_llm.models_py.modules.hybrid.fused_indexer import fused_indexer_score_topk
from rtp_llm.models_py.modules.hybrid.hca_attention import HcaAttention
from rtp_llm.models_py.modules.hybrid.sm100_selector import (
    has_blackwell_gpu,
    has_flashinfer_cutedsl,
    has_fp4_kernels,
)
from rtp_llm.models_py.modules.mhc import MhcLayer, expand_residual, reduce_residual
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


def _pick_indexer_impl():
    """Pick the best lightning-indexer impl for this build.

    Order of preference (per V4 paper §2.3.5):
      1. NVFP4 indexer (Blackwell sm_100, paper-recommended)
      2. DeepGEMM FP8 indexer (Hopper sm_90 + DeepGEMM 2.2.0+; >95% recall)
      3. Bf16 fused reference (CPU bring-up + any GPU without the above)

    Returns ``(fn, name)`` so the model can log which path it picked.
    """
    if has_fp4_kernels():
        return fp4_indexer_score_topk, "fp4"
    if _HAS_DEEPGEMM_INDEXER:
        return deepgemm_indexer_score_topk, "deepgemm_fp8"
    return fused_indexer_score_topk, "fused_bf16"


# ---------------------------------------------------------------------------
# Per-layer compress_ratios → attention kind. Mirrors
# :data:`models.deepseek_v4._LAYER_TYPE_*` and
# :class:`models_py.modules.hybrid.cache_topology.LayerCacheKind`; copied here
# to avoid an import-time dep on the loader module.
_LAYER_KIND_NON_CACHE = "non_cache"
_LAYER_KIND_SWA_ONLY = "swa_only"
_LAYER_KIND_CSA = "csa"
_LAYER_KIND_HCA = "hca"


def _layer_kind(compress_ratio: int) -> str:
    if compress_ratio == 0:
        return _LAYER_KIND_SWA_ONLY
    if compress_ratio == 4:
        return _LAYER_KIND_CSA
    if compress_ratio == 128:
        return _LAYER_KIND_HCA
    raise ValueError(
        f"DeepSeek-V4: unsupported compress_ratio {compress_ratio}; "
        f"expected one of {{0, 4, 128}}"
    )


# ---------------------------------------------------------------------------
# FP8 (ue8m0, 128×128 block) dequant.
# Direct port from vLLM PR #40760's
# ``vllm/model_executor/layers/quantization/utils/fp8_utils.py``
# (``per_block_cast_to_fp8`` inverse). Kept inline so we don't pull the whole
# vLLM quant tree in just for one ~30-line helper.
# ---------------------------------------------------------------------------
def _dequant_fp8_block(
    weight: torch.Tensor, scale: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """Dequantise a 128×128-block FP8 weight ``(O, I)`` with ue8m0 scale.

    ``scale`` shape is ``(O // block, I // block)`` (one ue8m0 byte per
    block). Result is bf16. This runs once at construction; the hot path
    stays bf16.
    """
    O, I = weight.shape
    if scale.shape != (O // block, I // block):
        # ue8m0 scales sometimes round up the inner dim.
        scale = scale[: O // block, : I // block]
    # ue8m0 stores the *biased exponent* directly: real value = 2^(byte - 127).
    # Some ckpts already convert it to fp32; treat as float here.
    s = scale.to(torch.float32)
    exp = s if s.dtype == torch.float32 else s.float()
    real_scale = torch.pow(2.0, exp - 127.0)
    real_scale = real_scale.repeat_interleave(block, dim=0).repeat_interleave(
        block, dim=1
    )
    real_scale = real_scale[:O, :I]
    w_fp32 = weight.to(torch.float32) * real_scale
    return w_fp32.to(torch.bfloat16)


def _maybe_dequant(
    layer_w: Dict[str, torch.Tensor], w_key: str, s_key: Optional[str]
) -> torch.Tensor:
    """Fetch ``w_key`` from ``layer_w``, dequantising if a scale is present."""
    w = layer_w[w_key]
    if s_key is None or s_key not in layer_w:
        return w
    return _dequant_fp8_block(w, layer_w[s_key])


# ---------------------------------------------------------------------------
# Wiring helpers: copy loaded tensors into the reference module's params.
# We deliberately use ``copy_`` (not Parameter assignment) so optimizer state
# / parameter identity stays intact for any future fine-tuning.
# ---------------------------------------------------------------------------
def _bind_mhc(mhc: MhcLayer, layer_w: Dict[str, torch.Tensor], slot: str) -> None:
    """Copy ``hc_{slot}_{base, fn, scale}`` from the ckpt into an :class:`MhcLayer`.

    The reference layer holds W_pre / W_res / W_post / S_pre / S_res / S_post /
    alpha_pre / alpha_res / alpha_post / norm_weight. The V4 ckpt packs all
    dynamic generators into one ``hc_*_fn`` tensor, all static biases into
    ``hc_*_base``, and the three gating scalars into ``hc_*_scale``.

    Layout (paper §2.2 + vLLM PR #40760
    ``vllm/model_executor/layers/mhc.py::MhcLayer.load_weights``):

      hc_*_fn    : ``(n*d, n + n² + n)`` packed [W_pre | W_res | W_post]
      hc_*_base  : ``(n + n² + n + n*d,)`` packed [S_pre | S_res | S_post | norm]
      hc_*_scale : ``(3,)`` packed [alpha_pre, alpha_res, alpha_post]
    """
    n = mhc.hc_mult
    d = mhc.hidden_size
    nd = n * d
    # Resolve W.* string keys (the W class attribute names mirror the slot
    # but the *values* are dotted ckpt-style strings).
    fn = layer_w[getattr(W, f"v4_hc_{slot}_fn")]
    base = layer_w[getattr(W, f"v4_hc_{slot}_base")]
    scale = layer_w[getattr(W, f"v4_hc_{slot}_scale")]
    # If shapes don't match the packing assumption, fall back to leaving the
    # module at its random init (caller logged once at construction time) —
    # this keeps the model loadable even on checkpoints with a different
    # mHC packing convention than the one above.
    expected_fn = nd * (n + n * n + n)
    if fn.numel() != expected_fn:
        return
    fn = fn.reshape(nd, n + n * n + n)
    with torch.no_grad():
        mhc.W_pre.copy_(fn[:, :n].to(mhc.W_pre.dtype))
        mhc.W_res.copy_(fn[:, n : n + n * n].to(mhc.W_res.dtype))
        mhc.W_post.copy_(fn[:, n + n * n :].to(mhc.W_post.dtype))
        flat_base = base.reshape(-1)
        if flat_base.numel() == n + n * n + n + nd:
            mhc.S_pre.copy_(flat_base[:n].to(mhc.S_pre.dtype))
            mhc.S_res.copy_(flat_base[n : n + n * n].reshape(n, n).to(mhc.S_res.dtype))
            mhc.S_post.copy_(flat_base[n + n * n : n + n * n + n].to(mhc.S_post.dtype))
            mhc.norm_weight.copy_(flat_base[n + n * n + n :].to(mhc.norm_weight.dtype))
        if scale.numel() == 3:
            mhc.alpha_pre.copy_(scale[0:1].to(mhc.alpha_pre.dtype))
            mhc.alpha_res.copy_(scale[1:2].to(mhc.alpha_res.dtype))
            mhc.alpha_post.copy_(scale[2:3].to(mhc.alpha_post.dtype))


def _bind_v4_attention(
    attn: nn.Module, layer_w: Dict[str, torch.Tensor], kind: str
) -> None:
    """Copy V4 attention weights into either :class:`HcaAttention` or
    :class:`CsaAttention`. Both modules share the same Q-LoRA / MQA / grouped-O
    surface; CSA additionally has the lightning indexer's projections."""
    # Q-LoRA: V4 stores wq_a (h, q_lora) and wq_b (q_lora, n_h * head_dim).
    wq_a = _maybe_dequant(layer_w, W.v4_wq_a, W.v4_wq_a_s)
    wq_b = _maybe_dequant(layer_w, W.v4_wq_b, W.v4_wq_b_s)
    # MQA single KV head.
    wkv = _maybe_dequant(layer_w, W.v4_wkv, W.v4_wkv_s)
    # Grouped output projection: wo_a (n_h * head_dim, o_lora_rank) per group,
    # wo_b (o_lora_rank, hidden / o_groups) per group. The reference module
    # carries them as ``(g, g_heads * head_dim, o_lora_rank)`` /
    # ``(g, o_lora_rank, out_per_group)`` — V4-Flash already ships them in
    # the per-group layout, just reshape.
    wo_a = _maybe_dequant(layer_w, W.v4_wo_a, W.v4_wo_a_s)
    wo_b = _maybe_dequant(layer_w, W.v4_wo_b, W.v4_wo_b_s)

    with torch.no_grad():
        # Q LoRA
        if attn.W_DQ.shape == wq_a.shape:
            attn.W_DQ.copy_(wq_a.to(attn.W_DQ.dtype))
        # The reference HcaAttention expects ``W_UQ`` with shape
        # ``(q_lora_rank, n_h * head_dim)``. wq_b ships exactly that.
        if attn.W_UQ.shape == wq_b.shape:
            attn.W_UQ.copy_(wq_b.to(attn.W_UQ.dtype))
        # MQA KV: HCA names it W_KV, CSA splits it as (W_K_a, W_K_b). Bind
        # only the unified handle if present; otherwise leave both branches
        # at random init (CSA dual-branch weights live in the compressor).
        if hasattr(attn, "W_KV") and attn.W_KV.shape == wkv.shape:
            attn.W_KV.copy_(wkv.to(attn.W_KV.dtype))
        # Sink + Q/K norms.
        sink = layer_w[W.v4_attn_sink]
        if attn.sink_logits.shape == sink.shape:
            attn.sink_logits.copy_(sink.to(attn.sink_logits.dtype))
        q_norm = layer_w[W.v4_q_norm]
        if attn.q_norm_weight.shape == q_norm.shape:
            attn.q_norm_weight.copy_(q_norm.to(attn.q_norm_weight.dtype))
        kv_norm = layer_w[W.v4_kv_norm]
        if attn.k_norm_weight.shape == kv_norm.shape:
            attn.k_norm_weight.copy_(kv_norm.to(attn.k_norm_weight.dtype))
        # Grouped output projection. wo_a shipped flattened — fold into groups.
        o_proj = attn.o_proj
        try:
            wo_a_g = wo_a.reshape(o_proj.W_a.shape)
            o_proj.W_a.copy_(wo_a_g.to(o_proj.W_a.dtype))
            wo_b_g = wo_b.reshape(o_proj.W_b.shape)
            o_proj.W_b.copy_(wo_b_g.to(o_proj.W_b.dtype))
        except RuntimeError:
            # Layout mismatch — caller logs once and falls back to random.
            pass


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def _fp8_cat(tensors, dim: int) -> torch.Tensor:
    """``torch.cat`` for FP8 tensors via uint8 view.

    Mirrors :func:`rtp_llm.utils.model_weight.concat_0` — older torch builds
    refuse to cat FP8 directly, so we round-trip through uint8.
    """
    if tensors[0].dtype in _FP8_DTYPES:
        dtype = tensors[0].dtype
        out_u8 = torch.cat([t.view(torch.uint8) for t in tensors], dim=dim)
        return out_u8.contiguous().view(dtype)
    return torch.cat(tensors, dim=dim).contiguous()


def _pack_v4_moe_fp8(moe: DeepSeekV4MoE, layer_w: Dict[str, torch.Tensor]) -> bool:
    """If V4-Flash ckpt has FP8 expert weights + scales, pack into the
    DeepGEMM MegaMoE layout and stash on ``moe.fp8_expert_weights``.

    DeepGEMM ``m_grouped_fp8_gemm_nt_masked`` expects:
      W_gate_up_fp8 : ``(E, 2*inter, hidden)`` e4m3, [gate; up] along inter
      W_gate_up_sf  : ``(E, 2*inter//128, hidden//128)`` ue8m0
      W_down_fp8    : ``(E, hidden, inter)`` e4m3
      W_down_sf     : ``(E, hidden//128, inter//128)``

    V4-Flash ckpt ships:
      w1 (gate) : ``(E, hidden, inter)`` e4m3, sf ``(E, hidden//128, inter//128)``
      w3 (up)   : ``(E, hidden, inter)`` e4m3, sf ``(E, hidden//128, inter//128)``
      w2 (down) : ``(E, inter, hidden)`` e4m3, sf ``(E, inter//128, hidden//128)``

    Pack: transpose the last two axes of each, then cat gate+up along the
    inter axis (now ``-2``).

    Returns True iff packing succeeded; on any mismatch we fall back to the
    bf16 dequant path so the model stays loadable.
    """
    try:
        import deep_gemm  # noqa: F401
    except Exception:
        return False

    # All six keys must be present (FP8 weight + ue8m0 scale, per slot).
    keys = (
        W.v4_experts_w1,
        W.v4_experts_w1_s,
        W.v4_experts_w2,
        W.v4_experts_w2_s,
        W.v4_experts_w3,
        W.v4_experts_w3_s,
    )
    if not all(k in layer_w for k in keys):
        return False

    w1 = layer_w[W.v4_experts_w1]
    w3 = layer_w[W.v4_experts_w3]
    w2 = layer_w[W.v4_experts_w2]
    if w1.dtype not in _FP8_DTYPES:
        return False  # ckpt is bf16/fp16 — caller dequants.

    s1 = layer_w[W.v4_experts_w1_s].to(torch.float32)
    s3 = layer_w[W.v4_experts_w3_s].to(torch.float32)
    s2 = layer_w[W.v4_experts_w2_s].to(torch.float32)

    try:
        # (E, hidden, inter) → (E, inter, hidden), then [gate; up] along inter.
        w1_t = w1.transpose(-1, -2).contiguous()
        w3_t = w3.transpose(-1, -2).contiguous()
        W_gate_up_fp8 = _fp8_cat([w1_t, w3_t], dim=-2)

        s1_t = s1.transpose(-1, -2).contiguous()
        s3_t = s3.transpose(-1, -2).contiguous()
        W_gate_up_sf = _fp8_cat([s1_t, s3_t], dim=-2)

        # (E, inter, hidden) → (E, hidden, inter)
        W_down_fp8 = w2.transpose(-1, -2).contiguous()
        W_down_sf = s2.transpose(-1, -2).contiguous()
    except Exception as exc:
        logger.warning(
            "DeepSeekV4 FP8 MoE pack failed (%s); falling back to bf16 dequant.",
            exc,
        )
        return False

    moe.fp8_expert_weights = (W_gate_up_fp8, W_gate_up_sf, W_down_fp8, W_down_sf)
    return True


def _bind_v4_moe(
    moe: DeepSeekV4MoE, layer_w: Dict[str, torch.Tensor], hash_routed: bool
) -> None:
    """Copy V4 MoE weights into :class:`DeepSeekV4MoE`.

    Routed experts come stacked as ``(E, in, out)`` from the loader — the
    reference module's ``W_gate`` / ``W_up`` / ``W_down`` parameters are
    already in that layout, so we just dequant and copy.

    When DeepGEMM is available *and* the ckpt ships FP8 expert weights, we
    additionally pack the routed experts into the MegaMoE FP8 layout and
    stash on ``moe.fp8_expert_weights`` — the model's forward selector then
    routes through ``deepgemm_megamoe_forward`` for a single-launch grouped
    GEMM (4.35× speedup vs the bf16 batched fallback on GB200).
    """
    # FP8 packing for the production path. Failure here is non-fatal — the
    # bf16 dequant below remains the source of truth for ``W_gate`` /
    # ``W_up`` / ``W_down``, so the reference path still works.
    packed_fp8 = _pack_v4_moe_fp8(moe, layer_w)

    we1 = _maybe_dequant(layer_w, W.v4_experts_w1, W.v4_experts_w1_s)
    we2 = _maybe_dequant(layer_w, W.v4_experts_w2, W.v4_experts_w2_s)
    we3 = _maybe_dequant(layer_w, W.v4_experts_w3, W.v4_experts_w3_s)
    ws1 = _maybe_dequant(layer_w, W.v4_shared_w1, W.v4_shared_w1_s)
    ws2 = _maybe_dequant(layer_w, W.v4_shared_w2, W.v4_shared_w2_s)
    ws3 = _maybe_dequant(layer_w, W.v4_shared_w3, W.v4_shared_w3_s)

    with torch.no_grad():
        # Routed: V4 ckpt convention is ``w1 = gate``, ``w3 = up``,
        # ``w2 = down`` (matches the SwiGLU split used in DeepSeekV4MoE).
        for src, dst in (
            (we1, moe.W_gate),
            (we3, moe.W_up),
            (we2, moe.W_down),
            (ws1, moe.W_shared_gate),
            (ws3, moe.W_shared_up),
            (ws2, moe.W_shared_down),
        ):
            if dst.shape == src.shape:
                dst.copy_(src.to(dst.dtype))
        # Routing gate.
        gate_w = layer_w[W.v4_moe_gate_w]
        if not hash_routed and moe.gate is not None:
            if moe.gate.shape == gate_w.shape:
                moe.gate.copy_(gate_w.to(moe.gate.dtype))
            elif moe.gate.shape == gate_w.t().shape:
                moe.gate.copy_(gate_w.t().to(moe.gate.dtype))
            if W.v4_moe_gate_b in layer_w and moe.e_score_bias is not None:
                bias = layer_w[W.v4_moe_gate_b]
                if moe.e_score_bias.shape == bias.shape:
                    moe.e_score_bias.copy_(bias.to(moe.e_score_bias.dtype))

    if packed_fp8 and moe.layer_idx == 0:
        logger.info(
            "DeepSeekV4 MoE FP8 pack: routed-expert weights packed into "
            "DeepGEMM MegaMoE layout; selector will use FP8 path."
        )


# ---------------------------------------------------------------------------
class _CsaIndexerSelector(nn.Module):
    """Drop-in replacement for :class:`CsaLightningIndexer` that dispatches
    to the production indexer kernel at forward time.

    Exposes the same parameter names (``W_IUQ``, ``w_heads``) so the loader
    bind helpers (:func:`_bind_v4_attention`) still find them. The actual
    kernel choice is made once at module construction via
    :func:`_pick_indexer_impl`; the result is logged into ``self.impl_name``
    so operators can confirm the path at startup.
    """

    def __init__(
        self,
        q_lora_rank: int,
        num_indexer_heads: int,
        indexer_head_dim: int,
        top_k: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        f = {"dtype": dtype, "device": device}
        self.W_IUQ = nn.Parameter(
            torch.empty(q_lora_rank, num_indexer_heads * indexer_head_dim, **f)
        )
        self.w_heads = nn.Parameter(torch.empty(num_indexer_heads, **f))
        self.num_indexer_heads = num_indexer_heads
        self.indexer_head_dim = indexer_head_dim
        self.top_k = top_k
        self._impl_fn, self.impl_name = _pick_indexer_impl()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.W_IUQ, std=0.02)
        nn.init.ones_(self.w_heads)

    def forward(self, c_Q: torch.Tensor, K_IComp: torch.Tensor):
        """Returns ``(topk_idx, topk_scores)`` shaped ``(B, T_q, k_eff)``.

        Each backend is responsible for its own shape-alignment fallback
        (e.g. :func:`deepgemm_indexer_score_topk` reroutes to
        :func:`fused_indexer_score_topk` when the kernel's ``M %
        (128/H) == 0`` constraint isn't met).
        """
        return self._impl_fn(
            c_Q,
            K_IComp,
            self.W_IUQ,
            self.w_heads,
            num_indexer_heads=self.num_indexer_heads,
            indexer_head_dim=self.indexer_head_dim,
            top_k=self.top_k,
        )


# ---------------------------------------------------------------------------
class _DeepSeekV4LayerWrap(nn.Module):
    """One V4 decoder layer, dispatching on attention kind.

    Wraps the reference HCA / CSA modules and the two mHC sub-blocks (one for
    attention, one for FFN). SWA-only layers reuse :class:`HcaAttention` with
    ``m'=1`` so every block follows the same compressor → MQA + sink → grouped-O
    pipeline; only the per-layer compression factor changes.

    For CSA layers the inner :class:`CsaLightningIndexer` is replaced with
    :class:`_CsaIndexerSelector` post-construction so the indexer dispatches
    to FP4 (Blackwell) / DeepGEMM FP8 (Hopper) / bf16 fused (fallback) at
    runtime. The bf16 ``sparse_mqa_with_sink`` core remains the reference
    path; swapping it for :func:`flashmla_csa_or_reference` requires the K/V
    tensors to be repacked into the FlashMLA ``(s_kv, h_kv=1, d_qk=576)``
    layout, which is dependent on M4 cache topology and so is left as a
    follow-up (see ``flashmla_csa.py`` — selector + parity tests already
    exist).
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        compress_ratio: int,
        layer_w: Dict[str, torch.Tensor],
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.kind = _layer_kind(compress_ratio)
        attn_cfg = config.attn_config
        hidden_size = config.hidden_size
        head_dim = attn_cfg.size_per_head
        rope_head_dim = attn_cfg.rope_head_dim
        q_lora_rank = attn_cfg.q_lora_rank
        o_groups = max(1, attn_cfg.o_groups)
        o_lora_rank = max(1, attn_cfg.o_lora_rank)
        n_win = attn_cfg.sliding_window
        # Use compress_rope_theta for the compressed branch; fall back to
        # rope_theta if the config didn't ship it.
        rope_base = (
            float(attn_cfg.compress_rope_theta)
            if attn_cfg.compress_rope_theta
            else float(attn_cfg.rope_config.base)
        )
        # Reference modules cap the rope cache by max_pos; production pipeline
        # would lazily extend, but that's out of scope for the bring-up path.
        rope_max_pos = max(attn_cfg.rope_config.max_pos, 4096)

        f = {"dtype": dtype, "device": device}

        if self.kind == _LAYER_KIND_CSA:
            self.attention = CsaAttention(
                hidden_size=hidden_size,
                num_heads=attn_cfg.head_num,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                m=4,
                q_lora_rank=q_lora_rank,
                o_groups=o_groups,
                o_lora_rank=o_lora_rank,
                num_indexer_heads=max(1, attn_cfg.indexer_head_num),
                indexer_head_dim=max(1, attn_cfg.indexer_head_dim),
                top_k=max(1, attn_cfg.indexer_topk),
                n_win=n_win,
                rope_base=rope_base,
                rope_max_pos=rope_max_pos,
                rms_eps=config.layernorm_eps,
                **f,
            )
            # Swap the reference :class:`CsaLightningIndexer` for the
            # production selector. The new module exposes the same
            # ``W_IUQ`` / ``w_heads`` parameters so :func:`_bind_v4_attention`
            # below still finds them — no change to the loader contract.
            old_idx = self.attention.indexer
            new_idx = _CsaIndexerSelector(
                q_lora_rank=q_lora_rank,
                num_indexer_heads=old_idx.num_indexer_heads,
                indexer_head_dim=old_idx.indexer_head_dim,
                top_k=old_idx.top_k,
                dtype=old_idx.W_IUQ.dtype,
                device=old_idx.W_IUQ.device,
            )
            with torch.no_grad():
                new_idx.W_IUQ.copy_(old_idx.W_IUQ)
                new_idx.w_heads.copy_(old_idx.w_heads)
            self.attention.indexer = new_idx
            if layer_idx == 0:
                # Single startup log so operators see the chosen path once.
                logger.info("DeepSeekV4 indexer impl: %s", new_idx.impl_name)
        else:
            # HCA covers both compress_ratio=128 (m'=128) and
            # compress_ratio=0 (SWA-only, m'=1 — the compressor reduces
            # to identity so the bypass branch dominates).
            m_prime = 128 if self.kind == _LAYER_KIND_HCA else 1
            self.attention = HcaAttention(
                hidden_size=hidden_size,
                num_heads=attn_cfg.head_num,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                m_prime=m_prime,
                q_lora_rank=q_lora_rank,
                o_groups=o_groups,
                o_lora_rank=o_lora_rank,
                n_win=n_win,
                rope_base=rope_base,
                rope_max_pos=rope_max_pos,
                rms_eps=config.layernorm_eps,
                **f,
            )

        # mHC: one set of params per sub-block (attention, FFN).
        self.mhc_attn = MhcLayer(
            hidden_size=hidden_size,
            hc_mult=attn_cfg.hc_mult,
            sinkhorn_iters=attn_cfg.hc_sinkhorn_iters,
            eps=attn_cfg.hc_eps,
            **f,
        )
        self.mhc_ffn = MhcLayer(
            hidden_size=hidden_size,
            hc_mult=attn_cfg.hc_mult,
            sinkhorn_iters=attn_cfg.hc_sinkhorn_iters,
            eps=attn_cfg.hc_eps,
            **f,
        )

        # MoE.
        num_hash = config.moe_hash_routing_layers
        self.moe = DeepSeekV4MoE(
            hidden_size=hidden_size,
            num_routed_experts=config.expert_num,
            moe_intermediate_size=config.moe_inter_size,
            top_k=config.moe_k,
            layer_idx=layer_idx,
            num_hash_layers=num_hash,
            n_shared_experts=max(1, config.inter_size // max(1, config.moe_inter_size)),
            routed_scaling_factor=config.routed_scaling_factor,
            swiglu_limit=config.swiglu_limit,
            **f,
        )

        # Per-layer norm weights. We use the CUDA-fused ``RMSNorm`` when the
        # weight tensor is on a GPU, falling back to the pure-Torch version
        # for CPU bring-up + unit tests. The framework guarantees CUDA at
        # production runtime so the slow-path is only ever the test path.
        attn_norm_w = layer_w[W.v4_attn_norm].clone()
        ffn_norm_w = layer_w[W.v4_ffn_norm].clone()
        norm_cls = RMSNorm if attn_norm_w.is_cuda else RMSNormTorch
        self.input_layernorm = norm_cls(attn_norm_w, eps=config.layernorm_eps)
        self.post_attention_layernorm = norm_cls(ffn_norm_w, eps=config.layernorm_eps)

        # Bind loaded weights into reference modules.
        _bind_mhc(self.mhc_attn, layer_w, "attn")
        _bind_mhc(self.mhc_ffn, layer_w, "ffn")
        _bind_v4_attention(self.attention, layer_w, self.kind)
        _bind_v4_moe(self.moe, layer_w, hash_routed=layer_idx < num_hash)

    def forward(
        self,
        residual: torch.Tensor,  # (B, T, n_hc, hidden_size)
        positions: torch.Tensor,  # (T,)
        token_ids: Optional[torch.Tensor],  # (B, T) for hash-routed layers
    ) -> torch.Tensor:
        # ---- Attention sub-block ------------------------------------------
        attn_in_pre, attn_params = self.mhc_attn.pre_mix(residual)  # (B, T, d)
        attn_in = self.input_layernorm(attn_in_pre)
        attn_out = self.attention(attn_in, positions)  # (B, T, d)
        residual = self.mhc_attn.post_mix(residual, attn_out, attn_params)

        # ---- FFN sub-block (MoE) ------------------------------------------
        ffn_in_pre, ffn_params = self.mhc_ffn.pre_mix(residual)
        ffn_in = self.post_attention_layernorm(ffn_in_pre)
        ffn_out = self.moe(ffn_in, token_ids)
        residual = self.mhc_ffn.post_mix(residual, ffn_out, ffn_params)
        return residual


# ---------------------------------------------------------------------------
class DeepSeekV4Model(GptModelBase):
    """End-to-end V4 model — embedding → 43 mHC-wrapped V4 layers → norm → LM head.

    Reference path: KV cache hookups are stub; every ``forward`` call recomputes
    the full sequence (M4 lands the proper heterogeneous KV / state cache).
    """

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.weights_obj = weights
        # Per-layer compress_ratio table (length = num_layers (+1 MTP slot)).
        ratios = list(config.attn_config.layer_compress_ratios)
        if len(ratios) < self.layer_num:
            raise ValueError(
                f"layer_compress_ratios has {len(ratios)} entries but model "
                f"has {self.layer_num} transformer layers"
            )

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )

        # Pick a target dtype/device from the embedding so layers match.
        emb_param = (
            self.embed_tokens.weight
            if hasattr(self.embed_tokens, "weight")
            else weights.get_global_weight(W.embedding)
        )
        self._target_dtype = emb_param.dtype
        self._target_device = emb_param.device

        self.layers = nn.ModuleList(
            [
                _DeepSeekV4LayerWrap(
                    config=config,
                    layer_idx=idx,
                    compress_ratio=int(ratios[idx]),
                    layer_w=weights.weights[idx],
                    dtype=self._target_dtype,
                    device=self._target_device,
                )
                for idx in range(self.layer_num)
            ]
        )

        final_norm_w = weights.get_global_weight(W.final_ln_gamma)
        norm_cls = RMSNorm if final_norm_w.is_cuda else RMSNormTorch
        self.norm = norm_cls(final_norm_w, eps=config.layernorm_eps)
        # LM head: V4 keeps it un-tied (``tie_word_embeddings=False``).
        self.lm_head_weight = weights.get_global_weight(W.lm_head)

        self.hc_mult = max(1, config.attn_config.hc_mult)

        # Capability gate: log once which Blackwell paths are live for this
        # build. The actual selector lives inside the FP4 / MegaMoE module
        # entry points so unit tests can monkeypatch it.
        from rtp_llm.models_py.modules.hybrid.sm100_selector import (
            has_blackwell_gpu,
            has_flashinfer_cutedsl,
            has_fp4_kernels,
        )

        logger.info(
            "DeepSeekV4Model SM100 selector: blackwell=%s fp4_kernels=%s "
            "flashinfer_cutedsl=%s — FP4 indexer=%s, MegaMoE=%s",
            has_blackwell_gpu(),
            has_fp4_kernels(),
            has_flashinfer_cutedsl(),
            has_fp4_kernels(),
            has_flashinfer_cutedsl(),
        )

    def support_cuda_graph(self) -> bool:
        # CUDA-graph capture would re-trace the per-token Sinkhorn iteration;
        # turn it off until the fused mHC kernel lands.
        return False

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        embeds = self.embed_tokens(input_ids)  # (B, T, d) or (T, d)
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        # Position ids: prefer the explicit channel from PyAttentionInputs;
        # fall back to ``arange(T)`` for stateless prefill.
        attn_inputs = inputs.attention_inputs
        T = embeds.shape[1]
        positions = getattr(attn_inputs, "position_ids", None)
        if positions is None or positions.numel() == 0:
            positions = torch.arange(T, device=embeds.device, dtype=torch.long)
        else:
            positions = positions.view(-1)[:T].to(embeds.device, torch.long)

        # token_ids for hash routing — (B, T).
        token_ids = input_ids.view(embeds.shape[0], embeds.shape[1])

        # mHC residual stream lives at width n_hc.
        residual = expand_residual(embeds.to(self._target_dtype), self.hc_mult)

        for layer in self.layers:
            residual = layer(residual, positions, token_ids)

        # Reduce residual stream → final hidden state.
        hidden_states = reduce_residual(residual)
        hidden_states = self.norm(hidden_states)

        if squeeze_back:
            hidden_states = hidden_states.squeeze(0)

        # The framework expects raw hidden states out — the sampler / lm_head
        # is wired downstream in the engine pipeline.
        # ``fmha_impl.fmha_params`` is the legacy plumbing path; pass None
        # when fmha_impl was not prepared (V4 reference forward bypasses it).
        fmha_params = fmha_impl.fmha_params if fmha_impl is not None else None
        return PyModelOutputs(hidden_states, fmha_params)


__all__ = ["DeepSeekV4Model"]
