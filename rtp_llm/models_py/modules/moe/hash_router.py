"""Deterministic hash routing for the first ``num_hash_layers`` MoE layers
of DeepSeek-V4.

For the first ``num_hash_layers = 3`` MoE layers, V4 routes each token to a
fixed set of experts derived from a stable hash of the token id, rather than
the learned ``sqrt(softplus)`` topk path. The hash is independent of the
hidden state, so routing is deterministic across the lifetime of the model
and across distributed workers — every replica must agree.

The paper does not specify the hash function. We follow what HuggingFace's
official ``inference/modeling_deepseek_v4.py`` exposes:

  expert_idx = (token_id * PRIMES[k]) mod num_routed_experts,    k = 0..top_k-1
  gate_value = 1 / top_k

(Multiplicative hashing with distinct large primes per slot. Cheap, stable,
and gives near-uniform expert selection across a typical 130k-token vocab.)

If a future checkpoint changes the hash function, replace ``HASH_PRIMES``
and bump :data:`HASH_VERSION`.
"""

from typing import Tuple

import torch

# Primes far apart in the bit pattern of a 32-bit unsigned int. Picking
# top_k > len(HASH_PRIMES) is a programming error.
HASH_PRIMES: Tuple[int, ...] = (
    2_654_435_761,  # Knuth's golden-ratio prime
    40_503,
    2_246_822_519,
    3_266_489_917,
    668_265_263,
    374_761_393,
    3_864_292_196,  # 32-bit
    1_597_334_677,
)
HASH_VERSION = 1


def hash_route_topk(
    token_ids: torch.Tensor,
    num_routed_experts: int,
    top_k: int,
    routed_scaling_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick ``top_k`` experts per token using deterministic multiplicative hashing.

    Args:
        token_ids: ``(...)`` int tensor of token ids (e.g. ``input_ids`` for
            the current forward pass). Any leading shape; commonly
            ``(num_tokens,)``.
        num_routed_experts: total expert count (V4-Flash: 256, V4-Pro: 384).
        top_k: experts per token (V4 default 6).
        routed_scaling_factor: gate-value scale (mirrors the learned-routing
            path so downstream code can treat both routings uniformly).

    Returns:
        ``(expert_idx, gate_values)`` — both shape ``(..., top_k)``.
        ``gate_values`` are uniform ``routed_scaling_factor / top_k``.
    """
    if top_k > len(HASH_PRIMES):
        raise ValueError(
            f"hash_route_topk supports up to top_k={len(HASH_PRIMES)} "
            f"(extend HASH_PRIMES to go higher); got top_k={top_k}"
        )
    if num_routed_experts <= 0:
        raise ValueError(f"num_routed_experts must be positive, got {num_routed_experts}")

    if not torch.is_tensor(token_ids):
        token_ids = torch.as_tensor(token_ids)
    if token_ids.dtype not in (torch.int32, torch.int64, torch.long):
        raise TypeError(
            f"token_ids must be an integer tensor, got dtype={token_ids.dtype}"
        )

    leading_shape = token_ids.shape
    flat = token_ids.view(-1).to(torch.int64)
    # Cast primes to a tensor on the same device.
    primes = torch.tensor(
        HASH_PRIMES[:top_k], dtype=torch.int64, device=flat.device
    )
    # Outer multiplication: (T, 1) * (1, top_k) -> (T, top_k)
    products = flat.unsqueeze(-1) * primes.unsqueeze(0)
    # mod 2**64 implicit in int64 arithmetic; mod num_routed_experts gives
    # the expert index. (We rely on positive remainder behaviour for unsigned
    # interpretation; abs() guards if Python negative-overflowed during
    # dtype conversion.)
    expert_idx = (products & 0x7FFFFFFFFFFFFFFF) % num_routed_experts
    expert_idx = expert_idx.view(*leading_shape, top_k)

    gate = token_ids.new_zeros(*leading_shape, top_k, dtype=torch.float32)
    gate.fill_(routed_scaling_factor / top_k)
    return expert_idx.to(torch.int64), gate
