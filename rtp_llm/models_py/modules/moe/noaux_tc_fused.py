"""Fused noaux_tc top-k routing (DeepSeek-V4 / V3.2 shared math).

Combines the four passes the reference :func:`v4_gating.noaux_tc_topk_v4`
makes — ``sqrt(softplus)`` → bias add → topk → gather → renormalise — into
a single function with explicit fp32 accumulation. The point is fewer
materialised tensors at the routing layer (which fires once per token per
MoE block, hot on every layer).

Math is unchanged. Source: SGLang PR #23600
``python/sglang/srt/layers/moe/deepseek_v4_topk.py::deepseek_v4_topk_fused``
(Triton kernel — Python launch wrapper around the per-token reduction).
vLLM PR #40760's CUDA equivalent is ``csrc/moe/topk_softplus_sqrt_kernels.cu``.

The pure-PyTorch fused path here trades two extra ops (an explicit softplus
+ sqrt) for one fewer materialised ``(N, E)`` tensor — a net win on bf16
when ``E = 256`` (V4-Flash) since the score tensor is ~2 MiB per token-batch.
"""

from typing import Tuple

import torch


def noaux_tc_topk_v4_fused(
    logits: torch.Tensor,  # (..., num_experts) raw expert logits
    bias: torch.Tensor,  # (num_experts,) e_score_correction_bias; may be None
    top_k: int,
    routed_scaling_factor: float,
    *,
    norm_topk_prob: bool = True,
    eps: float = 1e-20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused sqrt(softplus) + bias + topk + renormalise.

    Equivalent to :func:`v4_gating.noaux_tc_topk_v4` but written so the
    compiler sees the math as one straight-line ops graph (helps
    ``torch.compile`` / xformers fuse it, helps the reference path use
    contiguous memory).

    Args:
        logits: ``(..., num_experts)`` raw expert logits.
        bias: ``(num_experts,)`` per-expert bias added BEFORE topk
            selection — bias steers selection only, not the gate values.
            None for hash-routed layers (caller skips this function).
        top_k: ``num_experts_per_tok`` (V4 default 6).
        routed_scaling_factor: per-model scalar (Flash 1.5, Pro 2.5).
        norm_topk_prob: if True, divide selected scores by their sum so
            ``sum(gate_per_token) == routed_scaling_factor``.
        eps: tiny additive constant to keep the renormalise denominator
            non-zero when all top-k scores are ≈0 (only happens at init).

    Returns:
        ``(topk_idx, gate_values)`` — both shape ``(..., top_k)``,
        ``gate_values`` in the input dtype.
    """
    in_dtype = logits.dtype
    # Compute scores in fp32 with sqrt(softplus). softplus is numerically
    # stable for large negative inputs (fold with the log1p trick).
    logits_f32 = logits.float()
    scores = torch.sqrt(torch.nn.functional.softplus(logits_f32))

    if bias is not None:
        scores_with_bias = scores + bias.float()
    else:
        scores_with_bias = scores

    # Single topk; selecting on the bias-corrected scores so the bias only
    # *steers* selection, not the gate values returned.
    _, topk_idx = torch.topk(
        scores_with_bias, k=top_k, dim=-1, largest=True, sorted=True
    )
    gate_values = torch.gather(scores, dim=-1, index=topk_idx)

    if norm_topk_prob:
        gate_values = gate_values / (gate_values.sum(dim=-1, keepdim=True) + eps)
    gate_values = gate_values * routed_scaling_factor

    return topk_idx, gate_values.to(in_dtype)


__all__ = ["noaux_tc_topk_v4_fused"]
