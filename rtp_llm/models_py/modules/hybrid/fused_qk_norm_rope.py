"""Fused QK-RMSNorm + partial-RoPE — DeepSeek-V4 attention pre-pass.

Combines the two passes that bookend Q / K_compressed in the V4 attention
forward (RMSNorm over the head_dim, then partial RoPE on the trailing
``rope_head_dim``) into a single function. Saves one materialised tensor
per Q / K per layer.

Math is identical to the unfused
``HcaAttention._rmsnorm`` + ``apply_partial_rope`` chain — verified by the
parity test in :mod:`tests.fused_qk_norm_rope_test`. The fused CUDA kernel
that ships with the production path lives in vLLM PR #40760 at
``vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py`` (Python launch
wrapper) and ``csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`` (CUDA);
this module is the pure-PyTorch reference equivalent.
"""

from typing import Optional

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def fused_qk_norm_rope(
    x: torch.Tensor,  # (..., L, head_dim)
    norm_weight: torch.Tensor,  # (head_dim,) RMSNorm gamma
    cos: torch.Tensor,  # (L, rope_head_dim // 2)
    sin: torch.Tensor,  # (L, rope_head_dim // 2)
    rope_head_dim: int,
    *,
    inverse_rope: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm-then-partial-RoPE in one pass.

    Args:
        x: ``(..., L, head_dim)`` Q or K tensor; the leading axes are
            broadcast over.
        norm_weight: ``(head_dim,)`` RMSNorm gamma. Applied across the full
            head_dim *before* the RoPE rotation, matching the V4 paper's
            "Q/K RMSNorm" placement (paper §2.3, Eq. 16).
        cos, sin: ``(L, rope_head_dim // 2)`` from
            :func:`hca_attention.precompute_rope_cache`. Only the trailing
            ``rope_head_dim`` lanes get rotated (partial RoPE).
        rope_head_dim: number of trailing lanes to rotate.
        inverse_rope: if True, rotate by ``-pos`` (sign-flips ``sin``).
            Used on the attention output to undo the Q-side rotation, so
            the residual stream stays in the un-rotated frame.
        eps: RMSNorm epsilon. Match the model's ``rms_norm_eps``.

    Returns:
        Same shape as ``x``, normalised + partially rotated.
    """
    head_dim = x.shape[-1]
    if rope_head_dim > head_dim:
        raise ValueError(f"rope_head_dim {rope_head_dim} > head_dim {head_dim}")

    # ----- RMSNorm in fp32 over head_dim -------------------------------------
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x32 * torch.rsqrt(var + eps) * norm_weight.to(torch.float32)

    if rope_head_dim == 0:
        return x_norm.to(in_dtype)

    # ----- Partial RoPE on the trailing rope_head_dim lanes -----------------
    nope = head_dim - rope_head_dim
    if nope:
        x_nope, x_rope = x_norm.split([nope, rope_head_dim], dim=-1)
    else:
        x_nope, x_rope = None, x_norm

    cos_full = torch.cat([cos, cos], dim=-1).to(torch.float32)
    sin_full = torch.cat([sin, sin], dim=-1).to(torch.float32)
    if inverse_rope:
        sin_full = -sin_full
    while cos_full.dim() < x_rope.dim():
        cos_full = cos_full.unsqueeze(0)
        sin_full = sin_full.unsqueeze(0)

    rotated = (x_rope * cos_full) + (_rotate_half(x_rope) * sin_full)
    if x_nope is None:
        return rotated.to(in_dtype)
    return torch.cat([x_nope, rotated], dim=-1).to(in_dtype)


__all__ = ["fused_qk_norm_rope"]
