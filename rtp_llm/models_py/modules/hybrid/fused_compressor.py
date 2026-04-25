"""Fused token-level compressor — single matmul per (W_kv, W_z) pair.

The reference :func:`compressor.hca_compress` does two separate matmuls
(``H @ W_kv`` and ``H @ W_z``) and a separate add of ``bias_pos`` after
reshaping. Combining them into a single concatenated matmul
``H @ [W_kv | W_z]`` saves one full pass over the activations + the
intermediate ``Z`` tensor.

Source: vLLM PR #40760
``vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py``
(Python launch wrapper that fuses compress + FP8 quant of the cache write
in one CUDA call) + SGLang PR #23600
``python/sglang/srt/layers/attention/compressed/compressor.py``. CUDA fused
kernel lives at ``csrc/deepseek_v4/fused_compress_quant_cache_kernel.cu``;
this Python module is the bring-up reference.
"""

from typing import Optional

import torch


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def fused_hca_compress(
    H: torch.Tensor,  # (..., n, d)
    W_kv: torch.Tensor,  # (d, c)
    W_z: torch.Tensor,  # (d, c)
    bias_pos: torch.Tensor,  # (m_prime, c)
    m_prime: int,
) -> torch.Tensor:
    """One-matmul HCA compressor: project to ``[C | Z]`` jointly, split,
    softmax-weighted reduce per block.

    Identical math to :func:`compressor.hca_compress`. Returns
    ``(..., ceil(n / m'), c)``.
    """
    if m_prime < 1:
        raise ValueError(f"m_prime must be >= 1, got {m_prime}")
    if W_z.shape != W_kv.shape:
        raise ValueError(f"W_z {W_z.shape} != W_kv {W_kv.shape}")
    if bias_pos.shape[0] != m_prime:
        raise ValueError(
            f"bias_pos.shape[0] = {bias_pos.shape[0]} != m_prime ({m_prime})"
        )

    *prefix, n, d = H.shape
    c = W_kv.shape[-1]
    n_blocks = _ceil_div(n, m_prime)
    pad = n_blocks * m_prime - n
    if pad:
        # Zero-pad the tail. The matching softmax mask below kills those rows.
        H = torch.nn.functional.pad(H, (0, 0, 0, pad))

    # Joint matmul: (..., n+pad, 2c). Cheaper than two separate (n+pad, c)
    # gemms because the activation pass over H happens once.
    W_joint = torch.cat([W_kv, W_z], dim=-1)  # (d, 2c)
    proj = H @ W_joint  # (..., n+pad, 2c)
    C, Z = proj.split(c, dim=-1)  # each (..., n+pad, c)

    # Reshape to (..., n_blocks, m_prime, c) for per-block softmax.
    new_shape = (*prefix, n_blocks, m_prime, c)
    C_b = C.reshape(*new_shape)
    Z_b = Z.reshape(*new_shape)
    Z_b = Z_b + bias_pos  # broadcast (m_prime, c) over the batch dims

    if pad:
        # Mask the trailing ``pad`` positions inside the last block so their
        # softmax weight is 0. Build a (m_prime,) bool mask, then expand.
        last_block_valid = m_prime - pad
        mask = torch.zeros(m_prime, dtype=torch.bool, device=H.device)
        mask[last_block_valid:] = True
        # Expand to (..., n_blocks, m_prime, 1); only the LAST block masks.
        full_mask = torch.zeros(n_blocks, m_prime, 1, dtype=torch.bool, device=H.device)
        full_mask[-1] = mask.unsqueeze(-1)
        Z_b = Z_b.masked_fill(full_mask, float("-inf"))

    weights = torch.softmax(Z_b.float(), dim=-2).to(Z_b.dtype)
    return (weights * C_b).sum(dim=-2)  # (..., n_blocks, c)


def fused_csa_compress(
    H: torch.Tensor,  # (..., n, d)
    W_a_kv: torch.Tensor,  # (d, c)
    W_b_kv: torch.Tensor,  # (d, c)
    W_a_z: torch.Tensor,  # (d, c)
    W_b_z: torch.Tensor,  # (d, c)
    bias_a: torch.Tensor,  # (m, c)
    bias_b: torch.Tensor,  # (m, c)
    m: int,
) -> torch.Tensor:
    """One-matmul CSA compressor for the dual-branch overlapping case.

    The CSA reference does FOUR independent matmuls (``W_a_kv``, ``W_a_z``,
    ``W_b_kv``, ``W_b_z``); joining them into a single ``[W_a_kv|W_a_z|W_b_kv|W_b_z]``
    projection cuts the activation pass count from 4 to 1.

    See :func:`compressor.csa_compress` for the math (paper Eq. 9-12).
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    *prefix, n, d = H.shape
    c = W_a_kv.shape[-1]
    n_blocks = _ceil_div(n, m)
    pad = n_blocks * m - n
    if pad:
        H = torch.nn.functional.pad(H, (0, 0, 0, pad))

    # 4-way joint matmul over the channel axis.
    W_joint = torch.cat([W_a_kv, W_a_z, W_b_kv, W_b_z], dim=-1)
    proj = H @ W_joint  # (..., n+pad, 4c)
    C_a, Z_a, C_b, Z_b = proj.split(c, dim=-1)  # each (..., n+pad, c)

    # Per-block tensors: (..., n_blocks, m, c)
    new_shape = (*prefix, n_blocks, m, c)
    C_a_b = C_a.reshape(*new_shape)
    Z_a_b = Z_a.reshape(*new_shape) + bias_a
    C_b_b = C_b.reshape(*new_shape)
    Z_b_b = Z_b.reshape(*new_shape) + bias_b

    # CSA pulls from current block (a) + previous block (b). For block i=0
    # the previous block is missing → pad with -inf logit / zero value.
    prev_C = torch.zeros_like(C_b_b)
    prev_C[..., 1:, :, :] = C_b_b[..., :-1, :, :]
    prev_Z = torch.full_like(Z_b_b, -1e9)
    prev_Z[..., 1:, :, :] = Z_b_b[..., :-1, :, :]

    # Concat along the m axis: (..., n_blocks, 2m, c).
    cat_C = torch.cat([prev_C, C_a_b], dim=-2)
    cat_Z = torch.cat([prev_Z, Z_a_b], dim=-2)

    if pad:
        # Mask trailing ``pad`` rows of the LAST block in the second half.
        last_valid = m - pad
        mask = torch.zeros(2 * m, dtype=torch.bool, device=H.device)
        mask[m + last_valid :] = True
        full_mask = torch.zeros(n_blocks, 2 * m, 1, dtype=torch.bool, device=H.device)
        full_mask[-1] = mask.unsqueeze(-1)
        cat_Z = cat_Z.masked_fill(full_mask, float("-inf"))

    weights = torch.softmax(cat_Z.float(), dim=-2).to(cat_Z.dtype)
    return (weights * cat_C).sum(dim=-2)  # (..., n_blocks, c)


__all__ = ["fused_hca_compress", "fused_csa_compress"]
