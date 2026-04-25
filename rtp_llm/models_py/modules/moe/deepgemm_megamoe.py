"""DeepGEMM-backed MegaMoE expert GEMM for DeepSeek-V4 routed experts.

DeepGEMM 2.2.0 ships ``m_grouped_fp8_gemm_nt_masked`` — a single masked
grouped GEMM that runs *all* experts in one kernel launch, regardless
of how many are active. For V4 (E=256 routed experts × top-k 6) this is
the highest-leverage kernel on the model: replacing the
:func:`batched_experts_forward` per-expert slab loop turns
``num_active_experts`` Python iterations + per-expert cuBLAS launches
into a single kernel call.

The masked GEMM is invoked twice per layer (once for the SwiGLU
``[gate | up]`` projection, once for the ``down`` projection). Activations
are FP8-quantised per-token (e4m3, ue8m0 scale) to match the kernel
contract. Weights are pre-quantised at load time per 128×128 block (the
V4-Flash on-disk format already), so the wrapper just hands the cached
``(weight_fp8, scale)`` pair through.

Source:
* DeepGEMM 2.2.0 — kernel + Python launcher.
  https://github.com/deepseek-ai/DeepGEMM
* SGLang PR #23600 — ``python/sglang/srt/layers/moe/deepseek_v4_topk.py``
  uses the same ``m_grouped_fp8_gemm_nt_masked`` API for the V4 MoE
  expert GEMM (``MegaMoEDeepGemm`` class).
* vLLM PR #40760 — ``vllm/model_executor/layers/fused_moe/cutlass_moe.py``
  CUDA path ``cutlass_moe_fp8`` — DeepGEMM is the equivalent CUDA backend
  but reachable from Python without going through CUTLASS templates.

Shape contract (per layer):
  x        : (N, hidden) bf16
  topk_idx : (N, K) int64
  gate_val : (N, K) bf16
  W_gate_up: (E, 2*inter, hidden)   FP8 e4m3, packed [gate; up] along inter
  W_down   : (E, hidden, inter)     FP8 e4m3
  *_scale  : per-block fp32 scales matching the FP8 weights

Returns ``(N, hidden)`` weighted-sum of per-token top-k contributions —
same contract as :func:`batched_experts_forward`.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from rtp_llm.models_py.modules.moe.batched_experts import batched_experts_forward
from rtp_llm.models_py.modules.moe.clamped_swiglu import clamped_swiglu_split

logger = logging.getLogger(__name__)


def _has_deepgemm() -> bool:
    try:
        import deep_gemm  # noqa: F401

        return True
    except Exception as e:
        logger.debug("deep_gemm import failed: %s", e)
        return False


_HAS_DEEPGEMM = _has_deepgemm()
if _HAS_DEEPGEMM:
    import deep_gemm


def _route_to_dense(
    x: torch.Tensor,  # (N, H)
    topk_idx: torch.Tensor,  # (N, K)
    gate_values: torch.Tensor,  # (N, K)
    num_experts: int,
):
    """Reorder routed (token, slot) pairs into a dense ``(E, M_max, H)``
    masked-MoE input. Returns scatter info needed to undo the reorder.
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
    M_max = max(int(counts.max().item()), 1) if counts.numel() else 1

    # Round M_max up to the kernel's m alignment (matches
    # SGLang's deepgemm wrapper convention).
    align = max(deep_gemm.get_m_alignment_for_contiguous_layout(), 1)
    M_max = ((M_max + align - 1) // align) * align

    dense_x = torch.zeros(num_experts, M_max, H, dtype=x.dtype, device=x.device)
    intra = torch.arange(sorted_expert.numel(), device=x.device)
    offsets = torch.cat(
        [
            torch.zeros(1, dtype=counts.dtype, device=counts.device),
            counts.cumsum(0),
        ]
    )
    intra = intra - offsets[sorted_expert]
    dense_x[sorted_expert, intra] = x[sorted_tok]
    return (
        dense_x,
        counts.to(torch.int32),
        sorted_expert,
        intra,
        sorted_tok,
        sorted_gate,
        M_max,
    )


def _per_token_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token FP8 e4m3 cast with ue8m0 scale, in the layout DeepGEMM expects.

    Caller passes ``(E, M_max, K)``; we flatten the leading two dims for
    the cast (which is per-row) and reshape back.
    """
    E, M, K = x.shape
    fp8, sf = deep_gemm.per_token_cast_to_fp8(x.reshape(E * M, K), True)
    return fp8.view(E, M, K), sf.view(E, M, -1)


def deepgemm_megamoe_forward(
    x: torch.Tensor,  # (N, H) bf16
    topk_idx: torch.Tensor,  # (N, K) int64
    gate_values: torch.Tensor,  # (N, K) bf16/fp32
    W_gate_up_fp8: torch.Tensor,  # (E, 2*inter, H) fp8
    W_gate_up_sf: torch.Tensor,  # block-scale tensor for W_gate_up
    W_down_fp8: torch.Tensor,  # (E, H, inter) fp8
    W_down_sf: torch.Tensor,  # block-scale tensor for W_down
    swiglu_limit: float,
) -> torch.Tensor:
    """Single-launch DeepGEMM MegaMoE forward.

    Two ``m_grouped_fp8_gemm_nt_masked`` calls per layer:
      1. ``[gate; up] = X @ W_gate_up^T``  (E × M × 2*inter)
      2. ``Y = SwiGLU(gate, up) @ W_down^T``  (E × M × H)

    Then scatters back into ``(N, H)``.
    """
    if not _HAS_DEEPGEMM:
        raise RuntimeError(
            "deepgemm_megamoe_forward called without deep_gemm available"
        )

    E = W_gate_up_fp8.shape[0]
    H = x.shape[-1]
    inter2 = W_gate_up_fp8.shape[-2]
    inter = inter2 // 2

    (
        dense_x,
        masked_m,
        sorted_expert,
        intra,
        sorted_tok,
        sorted_gate,
        M_max,
    ) = _route_to_dense(x, topk_idx, gate_values, E)

    # FP8 quant the activations.
    a_fp8, a_sf = _per_token_quant(dense_x)

    # First grouped GEMM → (E, M_max, 2*inter)
    gate_up = torch.empty(E, M_max, inter2, device=x.device, dtype=x.dtype)
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (a_fp8, a_sf),
        (W_gate_up_fp8, W_gate_up_sf),
        gate_up,
        masked_m,
        expected_m=int(masked_m.max().item()),
    )

    # SwiGLU: split the inter axis [gate | up], apply silu(clamp(gate)) * clamp(up).
    gate, up = gate_up.split(inter, dim=-1)
    inner = clamped_swiglu_split(gate, up, swiglu_limit)

    # Second FP8 quant + grouped GEMM → (E, M_max, H)
    inner_fp8, inner_sf = _per_token_quant(inner)
    out_dense = torch.empty(E, M_max, H, device=x.device, dtype=x.dtype)
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (inner_fp8, inner_sf),
        (W_down_fp8, W_down_sf),
        out_dense,
        masked_m,
        expected_m=int(masked_m.max().item()),
    )

    # Scatter back: (sorted_expert, intra) → token row, scaled by gate_value.
    per_call = out_dense[sorted_expert, intra]  # (N*K, H)
    per_call = per_call * sorted_gate.unsqueeze(-1)
    out = torch.zeros_like(x)
    out.index_add_(0, sorted_tok, per_call)
    return out


def deepgemm_megamoe_or_batched(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    gate_values: torch.Tensor,
    *,
    bf16_weights: tuple,  # (W_gate, W_up, W_down) — used as fallback
    fp8_weights: Optional[tuple] = None,  # (W_gate_up_fp8, W_gate_up_sf,
    #  W_down_fp8, W_down_sf)
    swiglu_limit: float,
) -> torch.Tensor:
    """Top-level selector: DeepGEMM FP8 path when available + weights packed,
    else the bf16 :func:`batched_experts_forward` reference.

    The selector signature mirrors :func:`megamoe_or_batched` so callers
    can swap between the FlashInfer-NVFP4 and DeepGEMM-FP8 backends with
    one switch in the model.
    """
    if _HAS_DEEPGEMM and fp8_weights is not None:
        W_gu_fp8, W_gu_sf, W_d_fp8, W_d_sf = fp8_weights
        return deepgemm_megamoe_forward(
            x,
            topk_idx,
            gate_values,
            W_gu_fp8,
            W_gu_sf,
            W_d_fp8,
            W_d_sf,
            swiglu_limit,
        )
    W_gate, W_up, W_down = bf16_weights
    return batched_experts_forward(
        x, topk_idx, gate_values, W_gate, W_up, W_down, swiglu_limit
    )


__all__ = [
    "deepgemm_megamoe_forward",
    "deepgemm_megamoe_or_batched",
]
