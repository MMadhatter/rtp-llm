"""Token-level compressors for DeepSeek-V4's CSA + HCA attention paths.

Two compression flavours, both reducing a length-``n`` KV sequence to
length ``ceil(n / m)``:

  * :func:`hca_compress` — Heavily Compressed Attention (paper Eq. 22-23).
    Non-overlapping: compress every ``m'`` raw tokens into one entry by a
    softmax-weighted sum within each block. ``m' = 128`` for V4.

  * :func:`csa_compress` — Compressed Sparse Attention (paper Eq. 9-12).
    Overlapping: each compressed entry pulls from its own ``m`` window
    (``C_a``) plus the previous block's ``m`` window (``C_b``), producing
    ``2m`` candidates that get jointly softmax-normalised. ``m = 4`` for V4.
    For ``i = 0`` the missing previous block is padded with ``-inf`` softmax
    weights and zero values, so the first compressed entry collapses to the
    HCA-style single-block softmax.

Both share the same shape contract:

    H : (..., n, d)              hidden states being compressed
    -> C_comp : (..., ceil(n/m), c)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
def _block_softmax_2m(
    logits: torch.Tensor,
    pad_mask_first_block: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Softmax over the 2m elements that feed one compressed entry.

    ``logits`` shape: ``(..., 2*m, c)`` — softmax over dim ``-2``.
    ``pad_mask_first_block``, if given, is a ``(..., 2*m)`` boolean mask
    where True means "this position is padding" — its logits get ``-inf``
    so the softmax routes around them.
    """
    if pad_mask_first_block is not None:
        logits = logits.masked_fill(
            pad_mask_first_block.unsqueeze(-1), float("-inf")
        )
    return torch.softmax(logits.float(), dim=-2).to(logits.dtype)


def hca_compress(
    H: torch.Tensor,
    W_kv: torch.Tensor,
    W_z: torch.Tensor,
    bias_pos: torch.Tensor,
    m_prime: int,
    eps_pad_value: float = 0.0,
) -> torch.Tensor:
    """Compress ``H`` into ``ceil(n/m')`` entries using non-overlapping blocks.

    Implements paper Eq. (20)-(23):

      C   = H @ W_kv          # (..., n, c)
      Z   = H @ W_z           # (..., n, c)
      block(i) = [m'·i : m'·(i+1) - 1]
      S_block  = softmax_row(Z[block(i)] + bias_pos)         # (..., m', c)
      C_comp_i = sum_j  S_block[j] ⊙ C[block(i), j]          # (..., c)

    Args:
        H: ``(..., n, d)`` input hidden states.
        W_kv: ``(d, c)`` KV projection.
        W_z: ``(d, c)`` compression-weight projection.
        bias_pos: ``(m_prime, c)`` learnable per-position bias inside a block.
        m_prime: HCA compression ratio (V4: 128).
        eps_pad_value: value used for tail padding when ``n`` is not a
            multiple of ``m_prime``. The padded ``Z`` rows go to ``-inf``
            so they contribute zero softmax weight.

    Returns:
        ``(..., ceil(n / m_prime), c)`` compressed entries.
    """
    if m_prime < 1:
        raise ValueError(f"m_prime must be >= 1, got {m_prime}")
    if bias_pos.shape[0] != m_prime:
        raise ValueError(
            f"bias_pos.shape[0] = {bias_pos.shape[0]} != m_prime ({m_prime})"
        )

    *prefix, n, d = H.shape
    c = W_kv.shape[-1]
    if W_z.shape != W_kv.shape:
        raise ValueError(
            f"W_kv {tuple(W_kv.shape)} and W_z {tuple(W_z.shape)} must match"
        )

    # Pad to a multiple of m_prime along the seq dim.
    pad = (-n) % m_prime
    if pad:
        H_pad = torch.nn.functional.pad(H, (0, 0, 0, pad), value=eps_pad_value)
    else:
        H_pad = H
    n_padded = n + pad

    C = H_pad @ W_kv                                       # (..., n_padded, c)
    Z = H_pad @ W_z                                        # (..., n_padded, c)

    n_blocks = n_padded // m_prime
    # Reshape (..., n_padded, c) -> (..., n_blocks, m', c)
    new_shape = (*prefix, n_blocks, m_prime, c)
    C_blocked = C.view(*new_shape)
    Z_blocked = Z.view(*new_shape)

    # Per-position bias broadcasts across blocks.
    Z_with_bias = Z_blocked + bias_pos                     # (..., n_blocks, m', c)

    # Mask out tail-padding rows of the very last block (softmax along -2).
    if pad:
        pad_mask = torch.zeros(
            (n_blocks, m_prime),
            dtype=torch.bool, device=H.device,
        )
        pad_mask[-1, m_prime - pad :] = True
        Z_with_bias = Z_with_bias.masked_fill(
            pad_mask.unsqueeze(-1), float("-inf")
        )
    S = torch.softmax(Z_with_bias.float(), dim=-2).to(Z_with_bias.dtype)

    # Weighted sum across the m'_dim.
    C_comp = (S * C_blocked).sum(dim=-2)                   # (..., n_blocks, c)
    return C_comp


def csa_compress(
    H: torch.Tensor,
    W_a_kv: torch.Tensor,
    W_b_kv: torch.Tensor,
    W_a_z: torch.Tensor,
    W_b_z: torch.Tensor,
    bias_a: torch.Tensor,
    bias_b: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """CSA two-branch overlapping compressor (paper Eq. 9-12).

    For block index ``i``:

      C_a, C_b  = H @ W_a_kv, H @ W_b_kv          # (..., n, c)
      Z_a, Z_b  = H @ W_a_z,  H @ W_b_z           # (..., n, c)
      window_a  = [m·i : m·(i+1) - 1]              # current m positions
      window_b  = [m·(i-1) : m·i - 1]              # previous m positions

      For i == 0, window_b is padded with -inf softmax weights & zero values.

      [S_a; S_b] = softmax_row( [Z_a[window_a] + B_a; Z_b[window_b] + B_b] )
      C_comp_i  = sum_j  S_a[j] ⊙ C_a[window_a, j]  +  sum_j  S_b[j] ⊙ C_b[window_b, j]

    Returns ``C_comp`` of shape ``(..., ceil(n/m), c)``.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    for name, w in (("W_a_kv", W_a_kv), ("W_b_kv", W_b_kv),
                    ("W_a_z", W_a_z), ("W_b_z", W_b_z)):
        if w.shape != W_a_kv.shape:
            raise ValueError(
                f"{name} shape {tuple(w.shape)} != W_a_kv "
                f"{tuple(W_a_kv.shape)}"
            )
    for name, b in (("bias_a", bias_a), ("bias_b", bias_b)):
        if b.shape[0] != m:
            raise ValueError(
                f"{name}.shape[0] = {b.shape[0]} != m ({m})"
            )

    *prefix, n, d = H.shape
    c = W_a_kv.shape[-1]

    # Pad n -> multiple of m.
    pad = (-n) % m
    if pad:
        H_pad = torch.nn.functional.pad(H, (0, 0, 0, pad), value=0.0)
    else:
        H_pad = H
    n_padded = n + pad
    n_blocks = n_padded // m

    Ca = (H_pad @ W_a_kv).view(*prefix, n_blocks, m, c)        # current windows
    Cb = (H_pad @ W_b_kv).view(*prefix, n_blocks, m, c)        # for window_b reads
    Za = (H_pad @ W_a_z).view(*prefix, n_blocks, m, c)
    Zb = (H_pad @ W_b_z).view(*prefix, n_blocks, m, c)

    # window_b for block i is the *previous* block's m positions.
    # Block 0 has no previous: zero values, -inf softmax weights.
    Cb_shifted = torch.roll(Cb, shifts=1, dims=-3)
    Zb_shifted = torch.roll(Zb, shifts=1, dims=-3)
    Cb_shifted.select(-3, 0).zero_()
    Zb_shifted.select(-3, 0).fill_(float("-inf"))

    # Tail padding: zero out the padded rows of the final block in Ca.
    if pad:
        # Construct a (n_blocks, m) bool mask for padding positions in Ca.
        pad_mask = torch.zeros(
            (n_blocks, m), dtype=torch.bool, device=H.device,
        )
        pad_mask[-1, m - pad :] = True
        Ca = Ca.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        # Set Za pad rows to -inf so they get zero softmax weight.
        Za = Za.masked_fill(pad_mask.unsqueeze(-1), float("-inf"))

    # Add per-position learnable biases.
    Za_with_bias = Za + bias_a
    Zb_with_bias = Zb_shifted + bias_b

    # Concatenate along the m-dim to get 2m logits per block.
    Z_concat = torch.cat([Za_with_bias, Zb_with_bias], dim=-2)  # (..., n_blocks, 2m, c)
    C_concat = torch.cat([Ca, Cb_shifted], dim=-2)              # (..., n_blocks, 2m, c)

    S = torch.softmax(Z_concat.float(), dim=-2).to(Z_concat.dtype)
    C_comp = (S * C_concat).sum(dim=-2)                         # (..., n_blocks, c)
    return C_comp


# ---------------------------------------------------------------------------
class HcaCompressor(nn.Module):
    """nn.Module wrapper around :func:`hca_compress`.

    Owns the projection weights and per-position bias so they participate in
    parameter loading / state_dict / TP sharding the same way as other layers.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        m_prime: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        f = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.m_prime = m_prime
        self.W_kv = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_z = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.bias_pos = nn.Parameter(torch.empty(m_prime, head_dim, **f))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.W_kv, std=0.02)
        nn.init.normal_(self.W_z, std=0.02)
        nn.init.zeros_(self.bias_pos)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return hca_compress(H, self.W_kv, self.W_z, self.bias_pos, self.m_prime)


class CsaCompressor(nn.Module):
    """nn.Module wrapper around :func:`csa_compress`."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        m: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        f = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.m = m
        self.W_a_kv = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_b_kv = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_a_z = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_b_z = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.bias_a = nn.Parameter(torch.empty(m, head_dim, **f))
        self.bias_b = nn.Parameter(torch.empty(m, head_dim, **f))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z):
            nn.init.normal_(p, std=0.02)
        nn.init.zeros_(self.bias_a)
        nn.init.zeros_(self.bias_b)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return csa_compress(
            H, self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z,
            self.bias_a, self.bias_b, self.m,
        )
