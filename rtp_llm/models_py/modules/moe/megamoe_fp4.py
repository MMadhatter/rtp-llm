"""GB200 / Blackwell NVFP4 MegaMoE expert GEMM wrapper.

DeepSeek-V4 MoE has ``E = 256`` routed experts × top-k 6. The bf16
:func:`batched_experts_forward` path is fine on Hopper but on
Blackwell the NVFP4 grouped GEMM (FlashInfer's
``grouped_gemm_nt_masked``) gives a ~3× win because it (a) packs two
FP4 weights per byte (so weights fit in L2 instead of HBM) and (b)
issues one masked grouped GEMM for *all* experts at once, regardless of
how many are active.

This module wraps the FlashInfer cute-DSL kernel into the same shape
contract as :func:`batched_experts_forward` so the model can swap them
behind a single ``use_megamoe()`` gate. Falls back to the bf16 path
when the FlashInfer cute-DSL package or the FP4 kernels are missing.

Source: vLLM PR #40760 ``vllm/model_executor/layers/fused_moe/cutlass_moe.py``
(``cutlass_moe_fp4`` launcher) + the ``flashinfer.cute_dsl.blockscaled_gemm``
backend used by SGLang PR #23600 ``moe/deepseek_v4_topk.py``. The CUDA
kernel itself ships in FlashInfer 0.4+ (``csrc/nv/cute_dsl_kernels.cu``).
"""

from __future__ import annotations

import logging

import torch

from rtp_llm.models_py.modules.hybrid.sm100_selector import has_flashinfer_cutedsl
from rtp_llm.models_py.modules.moe.batched_experts import batched_experts_forward

logger = logging.getLogger(__name__)


if has_flashinfer_cutedsl():
    from rtp_llm.models_py.kernels.cuda.fp4_kernel.flashinfer_cutedsl_moe import (
        flashinfer_cutedsl_moe_masked,
    )
else:
    flashinfer_cutedsl_moe_masked = None


def _route_to_dense(
    x: torch.Tensor,  # (N, H)
    topk_idx: torch.Tensor,  # (N, K)
    gate_values: torch.Tensor,  # (N, K)
    num_experts: int,
):
    """Reorder routed tokens into a dense ``(E, M_max, H)`` MoE input.

    Returns ``(dense_x, masked_m, scatter_idx, sorted_gate)`` where:
      * ``dense_x`` is ``(E, M_max, H)`` zero-padded.
      * ``masked_m`` is ``(E,)`` int32 — per-expert valid-row count.
      * ``scatter_idx`` is the inverse permutation needed to scatter the
        per-expert outputs back into the ``(N, K, H)`` per-token layout.
      * ``sorted_gate`` is the gate value associated with each (expert,
        slot) pair, in the same order as ``dense_x``.

    The same trick the SGLang/vLLM MegaMoE launchers use — pad to a
    common ``M_max`` once so the grouped GEMM kernel can do everything
    in one launch.
    """
    N, K = topk_idx.shape
    H = x.shape[-1]
    flat_idx = topk_idx.reshape(-1)
    flat_gates = gate_values.reshape(-1)
    tok_ids = torch.arange(N, device=x.device).unsqueeze(-1).expand(N, K).reshape(-1)

    sorted_expert, sort_order = flat_idx.sort()
    sorted_tok = tok_ids[sort_order]
    sorted_gate = flat_gates[sort_order]
    counts = torch.bincount(sorted_expert, minlength=num_experts)
    M_max = int(counts.max().item()) if counts.numel() else 0
    M_max = max(M_max, 1)  # avoid zero-sized dense

    dense_x = torch.zeros(num_experts, M_max, H, dtype=x.dtype, device=x.device)
    # Where each (sorted) row lands inside its expert slab.
    intra = torch.zeros_like(sorted_expert)
    cursor = torch.zeros(num_experts, dtype=torch.long, device=x.device)
    # Vectorised intra-slab index: positions inside each expert's slab.
    # Build via cumulative count of expert ids.
    intra = torch.arange(sorted_expert.numel(), device=x.device)
    offsets = torch.cat(
        [
            torch.zeros(1, dtype=counts.dtype, device=counts.device),
            counts.cumsum(0),
        ]
    )
    # Subtract per-expert start to get the in-slab offset.
    intra = intra - offsets[sorted_expert]

    dense_x[sorted_expert, intra] = x[sorted_tok]
    masked_m = counts.to(torch.int32)
    return dense_x, masked_m, sorted_expert, intra, sorted_tok, sorted_gate


def megamoe_fp4_forward(
    x: torch.Tensor,  # (N, H) bf16
    topk_idx: torch.Tensor,  # (N, K) int64
    gate_values: torch.Tensor,  # (N, K) bf16/fp32
    w1: torch.Tensor,  # (E, 2*inter, H//2) uint8 fp4-packed
    w1_blockscale: torch.Tensor,  # (...) e4m3 swizzled
    w1_alpha: torch.Tensor,  # (E,) fp32
    w2: torch.Tensor,  # (E, H, inter//2) uint8 fp4-packed
    w2_blockscale: torch.Tensor,
    w2_alpha: torch.Tensor,  # (E,) fp32
    a1_global_scale: torch.Tensor,  # (E,) fp32
    a2_global_scale: torch.Tensor,  # (E,) fp32
) -> torch.Tensor:
    """One-launch NVFP4 MegaMoE expert forward. Falls back to bf16 path
    when the cute-DSL kernel is unavailable.

    Returns ``(N, H)`` weighted-sum of per-token top-k expert outputs.

    Note: the ``w1`` packed shape encodes both gate + up halves of the
    SwiGLU stacked along the intermediate axis (``2 * inter``); the
    cute-DSL kernel applies SiLU + multiplication internally
    (``silu_and_mul_scaled_nvfp4_experts_quantize``). Callers must
    pre-pack the weights in this layout — see vLLM PR #40760
    ``layers/fused_moe/cutlass_moe.py::pack_fp4_moe_weights``.
    """
    if flashinfer_cutedsl_moe_masked is None:
        raise RuntimeError(
            "megamoe_fp4_forward called but FlashInfer cute-DSL is unavailable; "
            "model selector should have routed to batched_experts_forward."
        )

    num_experts = w1.shape[0]
    dense_x, masked_m, sorted_expert, intra, sorted_tok, sorted_gate = _route_to_dense(
        x, topk_idx, gate_values, num_experts
    )

    # The cute-DSL launcher takes (hidden_states, None) when input quant
    # should happen inside the kernel; we hand it bf16 dense input plus
    # the per-expert global scale tensor.
    dense_out = flashinfer_cutedsl_moe_masked(
        hidden_states=(dense_x, None),
        input_global_scale=a1_global_scale,
        w1=w1,
        w1_blockscale=w1_blockscale,
        w1_alpha=w1_alpha,
        w2=w2,
        a2_global_scale=a2_global_scale,
        w2_blockscale=w2_blockscale,
        w2_alpha=w2_alpha,
        masked_m=masked_m,
    )  # (E, M_max, H)

    # Scatter back: pull (sorted_expert, intra) rows, scale by gate,
    # accumulate into per-token output.
    per_call = dense_out[sorted_expert, intra]  # (N*K, H)
    per_call = per_call * sorted_gate.unsqueeze(-1)
    out = torch.zeros_like(x)
    out.index_add_(0, sorted_tok, per_call)
    return out


def megamoe_or_batched(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    gate_values: torch.Tensor,
    *,
    bf16_weights: tuple,  # (W_gate, W_up, W_down)
    fp4_weights: tuple = None,  # (w1, w1_bs, w1_alpha, w2, w2_bs, w2_alpha,
    #  a1_gscale, a2_gscale)
    swiglu_limit: float,
) -> torch.Tensor:
    """Top-level selector — picks the FP4 path on Blackwell, else bf16.

    Keeps the model code free of the platform branch; the model passes
    *both* sets of weights and we pick. ``fp4_weights`` may be ``None``
    when the loader didn't pack them (e.g. weight format on disk is bf16
    only) — in that case we transparently use the bf16 path even on
    Blackwell.
    """
    if has_flashinfer_cutedsl() and fp4_weights is not None:
        (w1, w1_bs, w1_alpha, w2, w2_bs, w2_alpha, a1_gscale, a2_gscale) = fp4_weights
        return megamoe_fp4_forward(
            x,
            topk_idx,
            gate_values,
            w1,
            w1_bs,
            w1_alpha,
            w2,
            w2_bs,
            w2_alpha,
            a1_gscale,
            a2_gscale,
        )

    W_gate, W_up, W_down = bf16_weights
    return batched_experts_forward(
        x, topk_idx, gate_values, W_gate, W_up, W_down, swiglu_limit
    )


__all__ = [
    "megamoe_fp4_forward",
    "megamoe_or_batched",
]
