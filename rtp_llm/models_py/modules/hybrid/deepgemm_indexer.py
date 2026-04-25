"""DeepGEMM-backed FP8 lightning indexer for DeepSeek-V4.

DeepGEMM 2.2.0 ships ``fp8_mqa_logits`` and ``fp8_paged_mqa_logits``,
which implement *exactly* V4's lightning-indexer math in a single fused
CUDA kernel:

    logits[m, n] = Σ_h  weights[m, h]  ·  ReLU( Q[m, h, :] · K[n, :] )

(see the in-tree reference :func:`indexer_ref._ref_fp8_mqa_logits`).
This is the entire job of :class:`CsaLightningIndexer` and the
Python-fused :func:`fused_indexer_score_topk` — the kernel does it in
one launch with FP8 inputs, no per-head Python loop, no
materialisation of the ``(B, T_q, H, T_kc)`` ReLU tensor.

For V4-Flash (``H=64``, long ctx ``T_kc → 1M``) the bf16 reference
materialises ~64 GiB of ReLU intermediates per attention block; the FP8
kernel keeps the accumulator on-chip and writes ``(T_q, T_kc)`` only.
On Hopper (sm_90) it uses TMA + WGMMA; on Blackwell (sm_100) the same
launcher dispatches to the CTA-pair MMA path.

This wrapper provides three entry points:

  * :func:`deepgemm_indexer_logits_ragged` — wraps :func:`fp8_mqa_logits`
    for prefill (ragged, no paged cache).
  * :func:`deepgemm_indexer_logits_paged` — wraps
    :func:`fp8_paged_mqa_logits` for decode (paged cache).
  * :func:`deepgemm_indexer_score_topk` — top-level: takes raw bf16 Q/K +
    head weights, FP8-quantises, calls the kernel, returns top-k. Falls
    back to the bf16 :func:`fused_indexer_score_topk` when DeepGEMM
    is unavailable.

Source: this is the same DeepGEMM kernel that the in-tree V3.2 indexer
(:mod:`rtp_llm.models_py.modules.base.cuda.indexer_op`) uses for the
``Indexer`` op (see lines 388-403 / 477-484). vLLM PR #40760 wraps the
same kernel as ``deepgemm_paged_indexer`` in
``vllm/v1/attention/ops/deepseek_v4_ops/fp8_indexer.py``; SGLang PR
#23600 ditto under ``compressed/indexer_fp8.py``.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from rtp_llm.models_py.modules.hybrid.fused_indexer import fused_indexer_score_topk

logger = logging.getLogger(__name__)


def _has_deepgemm() -> bool:
    try:
        import deep_gemm  # noqa: F401

        return True
    except Exception as e:
        logger.debug("deep_gemm import failed in deepgemm_indexer: %s", e)
        return False


_HAS_DEEPGEMM = _has_deepgemm()
if _HAS_DEEPGEMM:
    import deep_gemm


def _per_token_fp8_with_block_scale(
    x: torch.Tensor,  # (..., D) bf16/fp16
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``x`` per-token to e4m3, returning ``(x_fp8, x_scale_fp32)``.

    ``x_scale`` shape ``(..., D // block_size)`` stored as fp32. This is
    the layout :func:`deep_gemm.fp8_mqa_logits` expects for both Q and K.

    The DeepGEMM cast helpers handle the e4m3 amax + ue8m0 scale; we
    just reshape so the leading "tokens" dim lines up with what the
    kernel expects (a flat M dim).
    """
    if x.dim() < 2:
        raise ValueError(f"x must have at least 2 dims (M, D); got {x.shape}")
    *prefix, D = x.shape
    if D % block_size != 0:
        raise ValueError(f"last dim D={D} must be divisible by block_size={block_size}")
    flat = x.reshape(-1, D)
    fp8, sf = deep_gemm.per_token_cast_to_fp8(flat, True)
    fp8 = fp8.view(*prefix, D)
    sf = sf.view(*prefix, D // block_size)
    return fp8, sf


def deepgemm_indexer_logits_ragged(
    q_fp8: torch.Tensor,  # (M, H, D) e4m3
    q_scale: torch.Tensor,  # (M, H) fp32 — per-token scale (broadcast over D)
    k_fp8: torch.Tensor,  # (N, D) e4m3
    k_scale: torch.Tensor,  # (N, D//block) fp32
    weights: torch.Tensor,  # (M, H) fp32 — per-head ReLU multiplier
    cu_seq_len_k_start: torch.Tensor,  # (M,) int32
    cu_seq_len_k_end: torch.Tensor,  # (M,) int32
    *,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Wraps :func:`deep_gemm.fp8_mqa_logits` with the V4 indexer contract.

    Returns ``(M, N)`` fp32 logits, masked to ``[ks_i, ke_i)`` per query
    when ``clean_logits=True``. The math is::

        logits[m, n] = Σ_h weights[m, h] · ReLU( Q_q[m, h, :] · K_k[n, :] )

    The kernel internally applies the per-token / per-block FP8 scales;
    we simply hand it the (fp8, scale) pair the same way the V3.2
    Indexer op does.
    """
    if not _HAS_DEEPGEMM:
        raise RuntimeError("deepgemm_indexer_logits_ragged: deep_gemm missing")
    # The V3.2 indexer_op passes (k_fp8, k_scale) as a tuple — same here.
    return deep_gemm.fp8_mqa_logits(
        q_fp8,
        (k_fp8, k_scale),
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        clean_logits=clean_logits,
    )


def deepgemm_indexer_logits_paged(
    q_fp8: torch.Tensor,  # (B, T_next, H, D) e4m3
    kv_cache_fp8: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, D + sf)
    weights: torch.Tensor,  # (B*T_next, H) fp32
    context_lens: torch.Tensor,  # (B,) int32
    block_table: torch.Tensor,  # (B, max_blocks) int32
    max_context_len: int,
    *,
    block_size: int = 64,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Wraps :func:`deep_gemm.fp8_paged_mqa_logits`.

    Same math as the ragged variant but reads K from a paged cache via
    ``block_table`` — used for decode where K positions span multiple
    pages of the V4 KV cache.

    The wrapper builds the schedule metadata once per call; for the
    common decode-with-fixed-context case the caller can reuse the
    metadata across micro-batches by calling
    :func:`deep_gemm.get_paged_mqa_logits_metadata` directly.
    """
    if not _HAS_DEEPGEMM:
        raise RuntimeError("deepgemm_indexer_logits_paged: deep_gemm missing")
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens,
        block_size,
        deep_gemm.get_num_sms(),
    )
    return deep_gemm.fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits=clean_logits,
    )


def deepgemm_indexer_score_topk(
    c_Q: torch.Tensor,  # (B, T_q, q_lora_rank) bf16
    K_IComp: torch.Tensor,  # (B, T_kc, indexer_head_dim) bf16
    W_IUQ: torch.Tensor,  # (q_lora_rank, H * D) bf16
    w_heads: torch.Tensor,  # (H,) fp32
    num_indexer_heads: int,
    indexer_head_dim: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-level V4 indexer top-k using DeepGEMM FP8 MQA logits.

    Drop-in replacement for :func:`fused_indexer_score_topk`. Falls back
    to the bf16 reference when:
      * ``deep_gemm`` is unavailable, or
      * shapes don't satisfy the kernel's alignment (``T_kc`` must be a
        multiple of 128 / block_size; ``indexer_head_dim`` must be a
        multiple of 128).

    The math is identical to the bf16 reference up to FP8 quant noise.
    """
    if not _HAS_DEEPGEMM:
        return fused_indexer_score_topk(
            c_Q,
            K_IComp,
            W_IUQ,
            w_heads,
            num_indexer_heads=num_indexer_heads,
            indexer_head_dim=indexer_head_dim,
            top_k=top_k,
        )

    B, T_q, _ = c_Q.shape
    T_kc = K_IComp.shape[-2]
    H = num_indexer_heads
    D = indexer_head_dim
    block_size = 128
    # The fp8_mqa_logits kernel asserts:
    #   * D (indexer_head_dim) % 128 == 0  (per-block FP8 scale layout)
    #   * T_kc % 128 == 0                  (K-side tile)
    #   * M (= B * T_q) % (128 / H) == 0   (Q-side tile aligned to block_q)
    # The third constraint is the subtle one — it's `seq_len_alignment %
    # block_q == 0` in DeepGEMM's source. See
    # rtp_llm/models_py/modules/base/cuda/test/get_topk_ragged_cp_test.py
    # ("use 32 heads so block_q=4") for the in-tree note about it.
    block_q = max(128 // H, 1)
    M = B * T_q
    if D % block_size != 0 or T_kc % 128 != 0 or M % block_q != 0:
        return fused_indexer_score_topk(
            c_Q,
            K_IComp,
            W_IUQ,
            w_heads,
            num_indexer_heads=num_indexer_heads,
            indexer_head_dim=indexer_head_dim,
            top_k=top_k,
        )

    k_eff = min(top_k, T_kc)

    # Q projection in bf16 first (cheap; q_lora_rank is small).
    Q_proj = (c_Q @ W_IUQ).view(B, T_q, H, D)  # (B, T_q, H, D)

    # FP8 quant Q (per-token, per-block-of-D); kernel expects (M, H, D).
    # Stack across batches into a single M dim — DeepGEMM is sequence-aware
    # via the cu_seqlen tensors below.
    Q_flat = Q_proj.reshape(B * T_q, H, D)
    q_fp8, q_sf = _per_token_fp8_with_block_scale(Q_flat, block_size=block_size)

    # FP8 quant K (single MQA head, so (N, D)).
    K_flat = K_IComp.reshape(B * T_kc, D)
    k_fp8, k_sf = _per_token_fp8_with_block_scale(K_flat, block_size=block_size)

    # Per-head weight broadcast to (M, H). The reference passes the head
    # weights flat across tokens because they are per-head constants.
    weights = w_heads.detach().float().view(1, H).expand(B * T_q, H).contiguous()

    # cu_seq_len_k_{start,end}: each query attends only to its own batch's
    # K segment ([b * T_kc, b * T_kc + T_kc)) when stacked this way.
    bsz_idx = torch.arange(B, device=c_Q.device).unsqueeze(-1)
    starts = (bsz_idx * T_kc).expand(B, T_q).reshape(-1).to(torch.int32)
    ends = ((bsz_idx + 1) * T_kc).expand(B, T_q).reshape(-1).to(torch.int32)

    logits = deepgemm_indexer_logits_ragged(
        q_fp8,
        q_sf,  # q_sf is unused once q_fp8 is in (M, H, D)
        k_fp8,
        k_sf,
        weights,
        starts,
        ends,
        clean_logits=True,
    )  # (M, N) — but N = B*T_kc; logits outside the per-query window are -inf.

    logits = logits.view(B, T_q, B * T_kc)
    # Pull out each batch's own keys: gather columns [b*T_kc : (b+1)*T_kc].
    # All other entries are -inf so a per-batch top-k naturally restricts.
    scores = torch.empty(B, T_q, T_kc, dtype=logits.dtype, device=logits.device)
    for b in range(B):
        scores[b] = logits[b, :, b * T_kc : (b + 1) * T_kc]
    topk_scores, topk_idx = scores.topk(k_eff, dim=-1)
    return topk_idx, topk_scores


__all__ = [
    "deepgemm_indexer_logits_ragged",
    "deepgemm_indexer_logits_paged",
    "deepgemm_indexer_score_topk",
]
