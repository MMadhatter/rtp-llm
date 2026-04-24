"""DeepSeek-V4 MoE gating primitives (M5 reference impls).

Three additions vs V3 / V3.2:

  1. ``sqrt_softplus_score(logits)`` replaces V3's ``sigmoid(logits)`` as the
     scoring function (config field ``scoring_func == 2``).
  2. ``noaux_tc_topk_v4`` drops the V3 ``n_group / topk_group`` node-routing
     constraint — V4 picks the global top-k experts directly.
  3. The router can split the layer index into a "hash routing" prefix and a
     learned-routing tail (see :mod:`hash_router`); this module just supplies
     the scoring + topk math, the layer-type dispatch is a model-level choice.

All math here is plain PyTorch — production will replace it with the
fused TileLang kernel from SGLang PR #23600
(``python/sglang/srt/layers/moe/deepseek_v4_topk.py``) once PR-C lands.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def sqrt_softplus_score(logits: torch.Tensor) -> torch.Tensor:
    """``sqrt(softplus(x))`` — the V4 expert scoring function (paper § 2.4).

    For ``x → -∞``: softplus(x) → 0, score → 0.
    For ``x → +∞``: softplus(x) ≈ x, score ≈ √x (sub-linear, unlike sigmoid
    which saturates at 1).

    Args:
        logits: ``(..., E)`` raw expert logits.

    Returns:
        ``(..., E)`` non-negative scores.
    """
    # Compute in fp32 for stability; F.softplus is numerically safe.
    return torch.sqrt(F.softplus(logits.float())).to(logits.dtype)


def noaux_tc_topk_v4(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V4's noaux_tc top-k router (no group / topk_group constraint).

    Pipeline (per the paper):

      scores      = sqrt(softplus(logits))           # E experts, per token
      score_with_bias = scores + bias                 # bias is the static
                                                      # ``e_score_correction_bias``
                                                      # learned during training
      topk_idx    = topk(score_with_bias, k=top_k)    # global top-k (no
                                                      # n_group filtering)
      gate_values = scores[topk_idx]                  # original scores —
                                                      # bias only steers
                                                      # selection, not gating
      if norm_topk_prob:
          gate_values /= gate_values.sum() + eps      # renormalise across k
      gate_values *= routed_scaling_factor

    Args:
        logits: ``(..., num_experts)`` raw expert logits.
        bias: ``(num_experts,)`` per-expert bias added before topk selection.
        top_k: ``num_experts_per_tok`` (V4 default 6).
        routed_scaling_factor: per-model scalar (Flash 1.5, Pro 2.5).
        norm_topk_prob: if True, divide selected scores by their sum so that
            ``sum(gate_values_per_token) == routed_scaling_factor``.

    Returns:
        ``(topk_idx, gate_values)`` — both shape ``(..., top_k)``.
    """
    # Compute scores in fp32 for numerical headroom.
    scores = sqrt_softplus_score(logits).float()
    if bias is not None:
        scores_with_bias = scores + bias.float()
    else:
        scores_with_bias = scores

    # Global top-k selection (no V3-style n_group node filtering).
    _, topk_idx = torch.topk(
        scores_with_bias, k=top_k, dim=-1, largest=True, sorted=True
    )
    gate_values = torch.gather(scores, dim=-1, index=topk_idx)

    if norm_topk_prob:
        eps = 1e-20
        gate_values = gate_values / (gate_values.sum(dim=-1, keepdim=True) + eps)
    gate_values = gate_values * routed_scaling_factor

    return topk_idx, gate_values.to(logits.dtype)
