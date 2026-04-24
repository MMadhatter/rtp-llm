"""Clamped SwiGLU activation — DeepSeek-V4 (paper § 2.4).

V4 clamps both branches of SwiGLU before multiplying:

  linear  = clamp(linear_branch, -SWIGLU_LIMIT, SWIGLU_LIMIT)
  gate    = clamp(gate_branch,   None,           SWIGLU_LIMIT)   # upper-only
  swiglu  = silu(gate) * linear

The default ``SWIGLU_LIMIT == 10.0`` matches the Flash-Base config
(``"swiglu_limit": 10.0``). Setting ``swiglu_limit <= 0`` disables clamping
and reduces to ordinary SwiGLU (handy for V3 / non-V4 models reusing the
same activation registry).
"""

from typing import Optional

import torch
import torch.nn.functional as F


def clamped_swiglu(
    gate_and_linear: torch.Tensor,
    swiglu_limit: Optional[float] = 10.0,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """SwiGLU on a fused ``[gate | linear]`` projection.

    Args:
        gate_and_linear: ``(..., 2 * inter)`` tensor; ``gate`` is the first
            half along ``dim``, ``linear`` the second half. This matches
            ``W3 / W1`` packed format used elsewhere in rtp_llm.
        swiglu_limit: clamp bound. ``None`` or ``<= 0`` disables clamping.
        dim: split dimension (default last).

    Returns:
        ``(..., inter)`` activated tensor.
    """
    gate, linear = torch.chunk(gate_and_linear, 2, dim=dim)
    if swiglu_limit is not None and swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        linear = linear.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * linear


def clamped_swiglu_split(
    gate: torch.Tensor,
    linear: torch.Tensor,
    swiglu_limit: Optional[float] = 10.0,
) -> torch.Tensor:
    """Variant that takes the two branches as separate tensors.

    Useful when the caller already has ``W_gate(x)`` and ``W_linear(x)`` as
    distinct tensors (e.g. unfused MoE expert path).
    """
    if swiglu_limit is not None and swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        linear = linear.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * linear
