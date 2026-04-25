"""GB200 / Blackwell FP4 lightning-indexer Q forward.

The bf16 reference path :func:`fused_indexer_score_topk` is fast on
Hopper because the H matmul is the expensive step. On Blackwell the
NVFP4 tensor cores let us run the per-head Q·K dot in FP4 with an
e4m3 block-scale; the speedup over bf16 is ~2× on the indexer because
its ``H * D`` dim is small (64 × 64 for V4-Flash, so a packed FP4 row
is 32 bytes per (head, k) — fits in shared memory comfortably).

Math is the bf16 reference modulo the FP4 quantisation noise. The
DeepSeek-V4 paper (§2.3.5) explicitly notes the indexer is FP4-on-
Blackwell, BF16-on-Hopper.

Source: vLLM PR #40760
``vllm/v1/attention/ops/deepseek_v4_ops/fp4_indexer.py`` (Python launch
wrapper) + ``csrc/deepseek_v4/indexer_kernel_fp4_sm100.cu`` (CUDA
fused kernel — out of scope for this Python port). SGLang PR #23600
``python/sglang/srt/layers/attention/compressed/indexer_fp4.py`` is the
sister wrapper.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

from rtp_llm.models_py.modules.hybrid.fused_indexer import fused_indexer_score_topk
from rtp_llm.models_py.modules.hybrid.sm100_selector import has_fp4_kernels

logger = logging.getLogger(__name__)

if has_fp4_kernels():
    # Defer the import behind the gate so Hopper builds stay clean.
    from rtp_llm.models_py.kernels.cuda.fp4_kernel.fp4_kernel import (
        cutlass_scaled_fp4_mm_wrapper,
        scaled_fp4_quant_wrapper,
    )
else:
    cutlass_scaled_fp4_mm_wrapper = None
    scaled_fp4_quant_wrapper = None


def _amax_global_scale(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor global scale for NVFP4 quant.

    NVFP4 uses a single ``input_global_scale`` per tensor on top of the
    per-16-elements e4m3 block scale; a safe choice is
    ``448 * 6 / amax(x)`` (vLLM ``_compute_input_global_scale``). 6.0 is
    the FP4 max value (e2m1), 448 is the e4m3 max for the block scale.
    Matches SGLang's ``compute_global_scale`` helper.
    """
    amax = x.detach().abs().float().amax().clamp_min(1e-6)
    return torch.tensor(448.0 * 6.0, device=x.device, dtype=torch.float32) / amax


def fp4_indexer_score_topk(
    c_Q: torch.Tensor,  # (B, T_q, q_lora_rank)
    K_IComp: torch.Tensor,  # (B, T_kc, indexer_head_dim)
    W_IUQ: torch.Tensor,  # (q_lora_rank, H * D)
    w_heads: torch.Tensor,  # (H,)
    num_indexer_heads: int,
    indexer_head_dim: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP4 lightning-indexer top-k. Falls back to bf16 path off Blackwell.

    The FP4 path quantises Q (post W_IUQ projection) and the compressed
    K stream once each, runs the per-head dot in NVFP4 via
    :func:`cutlass_scaled_fp4_mm_wrapper`, then reduces / topk in fp32.

    Returns ``(topk_idx, topk_scores)`` matching the bf16 reference's
    contract.
    """
    if not has_fp4_kernels():
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
    # CUTLASS NVFP4 GEMM constraints (m, n, k all multiples of 32 — block
    # size 16 × FP4 packing 2). V4-Flash production satisfies this for the
    # static D=64; long contexts handle T_q/T_kc naturally. Bring-up calls
    # with smaller shapes route through the bf16 path.
    if D % 32 != 0 or T_q % 32 != 0 or T_kc % 32 != 0:
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

    # FP4 quant kernel only accepts fp16/bf16. Coerce here so callers can
    # still pass fp32 buffers from the reference path.
    quant_dtype = torch.bfloat16
    Q_proj = (c_Q @ W_IUQ).view(B, T_q, H, D).to(quant_dtype)
    K_view = K_IComp.to(quant_dtype)  # (B, T_kc, D)

    # The FP4 path needs explicit per-tensor global scales for both
    # operands; compute once per batch from amax.
    scale_q = _amax_global_scale(Q_proj)
    scale_k = _amax_global_scale(K_view)
    # alpha for the matmul output is 1 / (scale_a * scale_b) so the
    # accumulator returns to the original numeric range.
    alpha = (1.0 / (scale_q * scale_k)).contiguous()

    scores = torch.zeros(B, T_q, T_kc, dtype=torch.float32, device=c_Q.device)
    w_heads_f = w_heads.detach().float()

    for b in range(B):
        # K side: (T_kc, D) → (T_kc, D//2 packed, scale)
        K_q, K_sf = scaled_fp4_quant_wrapper(K_view[b].contiguous(), scale_k)
        for h in range(H):
            # Q side, per-head: (T_q, D) → quant
            Q_q, Q_sf = scaled_fp4_quant_wrapper(
                Q_proj[b, :, h, :].contiguous(), scale_q
            )
            # FP4 GEMM: (T_q, D) × (T_kc, D)^T → (T_q, T_kc)
            per_head_logits = cutlass_scaled_fp4_mm_wrapper(
                Q_q, K_q, Q_sf, K_sf, alpha, out_dtype=torch.float32
            )
            scores[b].add_(per_head_logits.relu_(), alpha=float(w_heads_f[h]))

    topk_scores, topk_idx = scores.topk(k_eff, dim=-1)
    return topk_idx, topk_scores


__all__ = ["fp4_indexer_score_topk"]
