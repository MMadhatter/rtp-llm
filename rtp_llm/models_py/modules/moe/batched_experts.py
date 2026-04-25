"""Batched-MoE Python forward.

Replaces the per-expert ``for e in range(num_experts): ...`` loop in
:class:`DeepSeekV4MoE.forward` with a single sort + grouped GEMM. This is
the single biggest perf delta on the reference path — a 256-expert / 6-top-k
forward goes from O(num_experts) Python iterations to O(top_k * effective_experts)
without changing the math.

This is a Python-only optimisation; production will swap it for the fused
DeepGEMM "MegaMoE" kernel from SGLang PR #23600
(``python/sglang/srt/layers/moe/deepseek_v4_topk.py`` + the TileLang
``mega_moe_pre_dispatch.cuh`` kernel). See also vLLM PR #40760's
``csrc/moe/topk_softplus_sqrt_kernels.cu`` for the equivalent CUDA path.

Math reference (paper §2.4):

    for each token t and slot k in top_k:
        e            = topk_idx[t, k]
        expert_in    = x[t]
        expert_out   = silu(clamp(x @ W_gate[e], ≤10)) * clamp(x @ W_up[e], ±10)
        token_out[t] += gate_vals[t, k] * (expert_out @ W_down[e])

The grouping below (sort by expert, contiguous slabs per expert, single bmm
per stage) is the same trick used by SGLang's ``fused_moe_kernel`` and vLLM's
``moe_align_block_size`` + ``invoke_fused_moe_kernel``.
"""

from typing import Tuple

import torch

from rtp_llm.models_py.modules.moe.clamped_swiglu import clamped_swiglu_split


def batched_experts_forward(
    x: torch.Tensor,  # (N, hidden)
    topk_idx: torch.Tensor,  # (N, top_k) int64
    gate_values: torch.Tensor,  # (N, top_k)
    W_gate: torch.Tensor,  # (E, hidden, inter)
    W_up: torch.Tensor,  # (E, hidden, inter)
    W_down: torch.Tensor,  # (E, inter, hidden)
    swiglu_limit: float,
) -> torch.Tensor:
    """Run the routed-expert forward over ``(N, hidden)`` tokens once.

    Returns ``(N, hidden)`` weighted-sum of the per-token top-k expert
    contributions. Numerically equivalent to the per-expert scatter loop in
    :class:`DeepSeekV4MoE._expert_forward` + the outer loop, just with the
    Python overhead amortised over ``N * top_k`` rows instead of
    ``num_experts`` iterations.
    """
    N, H = x.shape
    top_k = topk_idx.shape[-1]
    assert topk_idx.shape == (N, top_k)
    assert gate_values.shape == (N, top_k)

    # Flatten (token, slot) into one dim so each row is one expert call.
    flat_idx = topk_idx.reshape(-1)  # (N * top_k,)
    flat_gates = gate_values.reshape(-1)  # (N * top_k,)
    # Token id repeated top_k times.
    tok_ids = (
        torch.arange(N, device=x.device).unsqueeze(-1).expand(N, top_k).reshape(-1)
    )

    # Sort all rows by expert id so we can dispatch contiguous slabs.
    sorted_expert, sort_order = flat_idx.sort()
    sorted_tok = tok_ids[sort_order]
    sorted_gate = flat_gates[sort_order]

    # Gather the input row for each (sorted) call.
    x_sorted = x[sorted_tok]  # (N*top_k, H)

    # Find slab boundaries: how many calls land on each expert.
    # ``bincount`` over ``num_experts`` keeps the output expert-aligned.
    num_experts = W_gate.shape[0]
    counts = torch.bincount(sorted_expert, minlength=num_experts)
    offsets = torch.cat(
        [
            torch.zeros(1, dtype=counts.dtype, device=counts.device),
            counts.cumsum(0),
        ]
    )

    out_sorted = torch.zeros_like(x_sorted)
    # Per-expert slab GEMM. Empty slabs are skipped — matches the V3 impl.
    # When ``num_active_experts`` << ``num_experts`` (typical V4 with 256
    # experts × top_k=6 over short prompts), this is much cheaper than
    # iterating all 256 expert ids unconditionally.
    active_experts = counts.nonzero(as_tuple=False).reshape(-1).tolist()
    for e in active_experts:
        start = int(offsets[e].item())
        end = int(offsets[e + 1].item())
        if end == start:
            continue
        x_e = x_sorted[start:end]  # (m, H)
        gate = x_e @ W_gate[e]  # (m, inter)
        linear = x_e @ W_up[e]  # (m, inter)
        h = clamped_swiglu_split(gate, linear, swiglu_limit)
        out_sorted[start:end] = h @ W_down[e]  # (m, H)

    # Apply per-row gate scale, then scatter back to token rows.
    out_sorted = out_sorted * sorted_gate.unsqueeze(-1)
    out = torch.zeros_like(x)
    out.index_add_(0, sorted_tok, out_sorted)
    return out


def topk_to_onehot(
    topk_idx: torch.Tensor, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert ``(N, top_k)`` indices into a ``(N, num_experts)`` one-hot mask
    plus the per-expert token count. Equivalent to vLLM PR #40760's
    ``topk_softplus_sqrt_kernels::compute_expert_offsets`` warm-up pass.

    The mask is fp32 so downstream weighted reductions stay numerically
    precise; the count is int32 to feed straight into ``moe_align_block_size``.
    """
    N, top_k = topk_idx.shape
    mask = torch.zeros(N, num_experts, dtype=torch.float32, device=topk_idx.device)
    mask.scatter_(1, topk_idx, 1.0)
    counts = mask.sum(0).to(torch.int32)
    return mask, counts


__all__ = ["batched_experts_forward", "topk_to_onehot"]
