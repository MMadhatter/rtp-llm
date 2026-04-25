"""FlashMLA-backed sparse attention for DeepSeek-V4 CSA path.

flash_mla 1.0.0 ships exactly the two kernels V4 CSA needs:

  * :func:`flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512,
    attn_sink, topk_length)` — sparse one-shot prefill MQA with
    per-head ``attn_sink`` and per-query variable top-k length.
  * :func:`flash_mla_with_kvcache(q, k_cache, ..., indices, attn_sink,
    extra_k_cache, extra_indices_in_kvcache, topk_length,
    extra_topk_length, is_fp8_kvcache)` — paged decode that combines
    the same sparse + sink + variable-topk semantics with extra-K
    cache slots for the SWA tail (tokens not yet in the paged cache).

Both kernels enforce ``d_qk = 576`` (NoPE 512 + RoPE 64) and
``d_v = 512``, which matches V4-Flash's main attention exactly. They are
the *production* CSA path; :func:`online_sink_mqa` from
``online_sink_attention.py`` remains as the bf16 reference for parity.

This module wraps the kernel in two helpers:

  * :func:`flashmla_csa_sparse_fwd` — prefill / single-pass forward.
    Takes the indices already produced by the lightning indexer
    (:func:`fused_indexer_score_topk` or its FP8 / FP4 cousins) and runs
    the sparse MQA in one launch.
  * :func:`flashmla_csa_decode` — paged decode with optional extra-K
    SWA tail. Builds the schedule metadata once per call.

The :func:`flashmla_csa_or_reference` selector picks the FlashMLA path
when the package is importable and shapes satisfy
``d_qk == 576``/``d_v == 512``; otherwise it routes to the bf16
reference (``online_sink_mqa`` + a manual gather at the indices). The
reference is the same one used for parity tests, so the fall-back is
numerically equivalent up to FP8 quant noise.

Source:
* FlashMLA 1.0.0 — kernel + Python wrapper.
  https://github.com/deepseek-ai/FlashMLA
* SGLang PR #23600 — ``python/sglang/srt/layers/attention/compressed/
  paged_prefill.py`` uses the same ``flash_mla_sparse_fwd`` API for V4 CSA.
* vLLM PR #40760 — ``vllm/v1/attention/backends/mla/flashmla.py``
  ``FlashMLAImpl.forward`` for the equivalent decode wrapper.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _has_flashmla() -> bool:
    try:
        import flash_mla  # noqa: F401

        return True
    except Exception as e:
        logger.debug("flash_mla import failed: %s", e)
        return False


_HAS_FLASHMLA = _has_flashmla()
if _HAS_FLASHMLA:
    import flash_mla


# Production constants enforced by FlashMLA's V4 path.
_V4_D_QK = 576  # 512 (NoPE) + 64 (RoPE)
_V4_D_V = 512
# sm_100 sparse-prefill kernel requires h_q ∈ {64, 128} (V4-Flash / V4-Pro)
# and topk % 64 == 0 (the B_TOPK tile size). Outside these, fall back.
_FLASHMLA_SPARSE_HQ = (64, 128)
_FLASHMLA_SPARSE_BTOPK = 64


def flashmla_csa_sparse_fwd(
    q: torch.Tensor,  # (s_q, h_q, d_qk=576) bf16
    kv: torch.Tensor,  # (s_kv, h_kv=1, d_qk=576) bf16 — MQA single KV head
    indices: torch.Tensor,  # (s_q, h_kv=1, topk) int32; -1 for invalid
    sm_scale: float,
    *,
    attn_sink: Optional[torch.Tensor] = None,  # (h_q,) fp32
    topk_length: Optional[torch.Tensor] = None,  # (s_q,) int32
) -> torch.Tensor:
    """Sparse prefill / one-pass CSA forward via FlashMLA.

    Returns the attention output ``(s_q, h_q, d_v=512)`` in bf16. The
    kernel internally:
      1. gathers KV at ``indices`` (out-of-range entries skipped),
      2. runs MQA softmax with per-head ``attn_sink`` folded into the
         denominator,
      3. truncates to the leftmost ``topk_length[i]`` indices per query
         when provided (variable per-query top-k).
    """
    if not _HAS_FLASHMLA:
        raise RuntimeError("flashmla_csa_sparse_fwd called without flash_mla")
    if q.shape[-1] != _V4_D_QK or kv.shape[-1] != _V4_D_QK:
        raise ValueError(
            f"flash_mla_sparse_fwd requires d_qk={_V4_D_QK}; got q.d_qk="
            f"{q.shape[-1]}, kv.d_qk={kv.shape[-1]}"
        )
    if kv.shape[-2] != 1:
        raise ValueError(
            f"flash_mla_sparse_fwd is MQA: kv must have h_kv=1; got " f"{kv.shape[-2]}"
        )
    out, _max_logits, _lse = flash_mla.flash_mla_sparse_fwd(
        q.contiguous(),
        kv.contiguous(),
        indices.contiguous().to(torch.int32),
        sm_scale,
        d_v=_V4_D_V,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    return out


def flashmla_csa_decode(
    q: torch.Tensor,  # (B, seq_len_q, h_q, d_qk=576) bf16
    k_cache: torch.Tensor,  # (num_blocks, page_block_size, h_kv=1, head_dim) — fp8 or bf16
    block_table: Optional[
        torch.Tensor
    ],  # (B, max_blocks) int32 — None ok in sparse mode
    cache_seqlens: Optional[torch.Tensor],  # (B,) int32 — None ok in sparse mode
    indices: torch.Tensor,  # (B, seq_len_q, topk) int32
    *,
    attn_sink: Optional[torch.Tensor] = None,  # (h_q,) fp32
    topk_length: Optional[torch.Tensor] = None,  # (B,) int32
    extra_k_cache: Optional[torch.Tensor] = None,  # SWA tail K
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    is_fp8_kvcache: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Paged decode CSA forward via FlashMLA.

    Combines the sparse + sink + extra-K-tail semantics V4 needs. The
    extra-K tail covers SWA-recent tokens that haven't been compressed
    or paged into ``k_cache`` yet — they get attended to with a separate
    ``extra_indices_in_kvcache`` index list and an optional
    ``extra_topk_length`` for variable per-query length.

    Returns the attention output ``(B, seq_len_q, h_q, d_v=512)`` in bf16.
    The schedule metadata is built once per call; cache it externally if
    you call this in a tight inner loop with the same shape.
    """
    if not _HAS_FLASHMLA:
        raise RuntimeError("flashmla_csa_decode called without flash_mla")
    sched_meta, _ = flash_mla.get_mla_metadata(
        (
            cache_seqlens
            if cache_seqlens is not None
            else torch.zeros(
                q.shape[0],
                dtype=torch.int32,
                device=q.device,
            )
        ),
        q.shape[1] * q.shape[2],
        1,
    )
    out, _lse = flash_mla.flash_mla_with_kvcache(
        q.contiguous(),
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v=_V4_D_V,
        tile_scheduler_metadata=sched_meta,
        num_splits=None,
        softmax_scale=softmax_scale,
        causal=False,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )
    return out


def _reference_sparse_csa(
    q: torch.Tensor,  # (s_q, h_q, d_qk=576)
    kv: torch.Tensor,  # (s_kv, 1, d_qk=576)
    indices: torch.Tensor,  # (s_q, 1, topk) int32
    sm_scale: float,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
) -> torch.Tensor:
    """bf16 reference for :func:`flashmla_csa_sparse_fwd`.

    Inlined sparse MQA + sink + V-truncation; doesn't reuse
    ``online_sink_mqa`` because that helper requires a non-empty
    compressed branch and CSA routes everything through ``indices``.

    Math (per query token i, head h):

      gathered_kv[i, k, :] = kv[indices[i, 0, k], 0, :]      # (topk, d_qk)
      logits[i, h, k]      = (q[i, h, :] · gathered_kv[i, k, :]) * sm_scale
      mask out k where indices == -1, indices >= s_kv, or k >= topk_length[i]
      Z[i, h]              = Σ_k exp(logits[i, h, k]) + exp(attn_sink[h])
      attn_w[i, h, k]      = exp(logits[i, h, k]) / Z[i, h]
      output[i, h, :d_v]   = Σ_k attn_w[i, h, k] · gathered_kv[i, k, :d_v]

    Identical to FlashMLA up to FP16/BF16/FP8 quant noise.
    """
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]
    d_v = _V4_D_V

    flat_idx = indices.squeeze(-2)  # (s_q, topk)
    valid = (flat_idx >= 0) & (flat_idx < s_kv)
    safe_idx = flat_idx.clamp(min=0, max=max(s_kv - 1, 0))
    gathered = kv.squeeze(-2)[safe_idx]  # (s_q, topk, d_qk)

    if topk_length is not None:
        col = torch.arange(topk, device=q.device).unsqueeze(0)
        valid = valid & (col < topk_length.unsqueeze(-1))

    # logits: (s_q, h_q, topk)
    logits = torch.einsum("qhd,qkd->qhk", q.float(), gathered.float()) * sm_scale
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))

    # Sink folded into denominator (per-head).
    if attn_sink is not None:
        sink = attn_sink.float().view(1, h_q, 1)
    else:
        sink = torch.full(
            (1, h_q, 1), float("-inf"), dtype=torch.float32, device=q.device
        )

    # Numerically stable softmax with the sink term included in the max.
    max_l = logits.amax(dim=-1, keepdim=True)
    global_max = torch.maximum(max_l, sink)
    e = (logits - global_max).exp()
    Z = e.sum(dim=-1, keepdim=True) + (sink - global_max).exp()
    weights = e / Z  # (s_q, h_q, topk)

    # Gather V (first d_v lanes) and apply weights.
    v_slice = gathered[..., :d_v].float()  # (s_q, topk, d_v)
    out = torch.einsum("qhk,qkd->qhd", weights, v_slice)
    return out.to(q.dtype).contiguous()


def flashmla_csa_or_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-level CSA selector. Picks FlashMLA when shape + import OK.

    Drop-in for V4 CSA prefill: same input/output as
    :func:`flashmla_csa_sparse_fwd`, but transparently falls back to the
    bf16 reference :func:`online_sink_mqa` (via ``_reference_sparse_csa``)
    when:
      * ``flash_mla`` not importable, or
      * ``d_qk != 576`` (e.g. bring-up shapes), or
      * any GPU-only kernel constraint isn't met.
    """
    use_flashmla = (
        _HAS_FLASHMLA
        and q.is_cuda
        and q.shape[-1] == _V4_D_QK
        and kv.shape[-1] == _V4_D_QK
        and kv.shape[-2] == 1
        and q.shape[-2] in _FLASHMLA_SPARSE_HQ
        and indices.shape[-1] % _FLASHMLA_SPARSE_BTOPK == 0
    )
    if use_flashmla:
        return flashmla_csa_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
    return _reference_sparse_csa(
        q,
        kv,
        indices,
        sm_scale,
        attn_sink,
        topk_length,
    )


__all__ = [
    "flashmla_csa_sparse_fwd",
    "flashmla_csa_decode",
    "flashmla_csa_or_reference",
]
