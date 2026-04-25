"""DeepSeek-V4 dual-base partial RoPE.

DeepSeek-V4 uses **two** RoPE bases simultaneously (paper §2.3.5):

  * ``rope_theta`` (Flash: 10_000) — applied to Q on raw-position indices.
  * ``compress_rope_theta`` (Flash: 160_000) — applied to the **compressed**
    K stream, which carries one RoPE entry per ``m`` raw tokens. The high
    base lets the rotated dim absorb a larger effective receptive field
    without aliasing.

Both rotations are *partial* — only the trailing ``rope_head_dim`` lanes of
the head are touched (`nope` lanes pass through). The cache module below
holds **two** ``(cos, sin)`` tables and lets callers pick a side per call.

Source: vLLM PR #40760
``vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py``
(Python launch wrapper for the dual-base RoPE) and SGLang PR #23600
``python/sglang/srt/layers/deepseek_v4_rope.py``. CUDA fused launch lives
in vLLM's ``csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`` —
this module is the pure-PyTorch reference equivalent.
"""

from typing import Tuple

import torch
import torch.nn as nn


def _build_inv_freq(rope_dim: int, base: float, dtype, device) -> torch.Tensor:
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    half = rope_dim // 2
    return 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )


def _build_cache(
    max_pos: int, rope_dim: int, base: float, *, device, dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` of shape ``(max_pos, rope_dim // 2)``."""
    inv = _build_inv_freq(rope_dim, base, dtype, device)
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool,
) -> torch.Tensor:
    head_dim = x.shape[-1]
    if rope_dim == 0:
        return x
    if rope_dim > head_dim:
        raise ValueError(f"rope_dim {rope_dim} > head_dim {head_dim}")

    nope = head_dim - rope_dim
    if nope:
        x_nope, x_rope = x.split([nope, rope_dim], dim=-1)
    else:
        x_nope, x_rope = None, x

    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    if inverse:
        sin_full = -sin_full
    while cos_full.dim() < x_rope.dim():
        cos_full = cos_full.unsqueeze(0)
        sin_full = sin_full.unsqueeze(0)

    rotated = (x_rope * cos_full) + (_rotate_half(x_rope) * sin_full)
    if x_nope is None:
        return rotated
    return torch.cat([x_nope, rotated], dim=-1)


# ---------------------------------------------------------------------------
class DeepSeekV4DualRope(nn.Module):
    """Dual-base partial-RoPE cache for DeepSeek-V4 attention.

    Holds two ``(cos, sin)`` tables, one per base. Callers index into them
    with absolute positions; for the K branch the position is the
    *compressed* index (e.g. ``i * m + m // 2``).

    Args:
        rope_head_dim: trailing lanes of each head that get rotated (V4: 64).
        max_pos_q: max query position (covers the full original sequence).
        max_pos_k: max compressed position (typically ``max_pos_q // m``).
        rope_theta_q: base for the Q branch (Flash: 10_000).
        rope_theta_k: base for the K (compressed) branch (Flash: 160_000).
    """

    def __init__(
        self,
        rope_head_dim: int,
        max_pos_q: int,
        max_pos_k: int,
        rope_theta_q: float,
        rope_theta_k: float,
        *,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        self.rope_head_dim = rope_head_dim
        cos_q, sin_q = _build_cache(
            max_pos_q, rope_head_dim, rope_theta_q, device=device, dtype=dtype
        )
        cos_k, sin_k = _build_cache(
            max_pos_k, rope_head_dim, rope_theta_k, device=device, dtype=dtype
        )
        self.register_buffer("cos_q", cos_q, persistent=False)
        self.register_buffer("sin_q", sin_q, persistent=False)
        self.register_buffer("cos_k", cos_k, persistent=False)
        self.register_buffer("sin_k", sin_k, persistent=False)

    # ------------------------------------------------------------------
    def apply_q(
        self, x: torch.Tensor, positions: torch.Tensor, *, inverse: bool = False
    ) -> torch.Tensor:
        """Rotate Q with the *raw-position* base.

        ``positions`` shape ``(L,)``; ``x`` shape ``(..., L, head_dim)``.
        ``inverse=True`` rotates by ``-pos`` — used to undo the Q rotation
        on the attention output (V4 §2.3 inverse-RoPE-on-output trick).
        """
        cos = self.cos_q[positions]
        sin = self.sin_q[positions]
        return _apply(x, cos, sin, self.rope_head_dim, inverse=inverse)

    def apply_k(
        self, x: torch.Tensor, positions: torch.Tensor, *, inverse: bool = False
    ) -> torch.Tensor:
        """Rotate K (compressed branch) with the *compressed-position* base."""
        cos = self.cos_k[positions]
        sin = self.sin_k[positions]
        return _apply(x, cos, sin, self.rope_head_dim, inverse=inverse)


__all__ = ["DeepSeekV4DualRope"]
