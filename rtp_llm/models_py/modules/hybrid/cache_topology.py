"""Per-layer KV cache topology for DeepSeek-V4 (M4, Python side).

DeepSeek-V4 ships a heterogeneous KV cache: each transformer layer can be one
of four kinds, dictated by its ``compress_ratios`` entry in the HF config:

  * ``128`` → **HCA** layer. KV is compressed at ratio ``m' = 128``; one cache
    block holds ``k2 = lcm(m, m') / m' = 1`` compressed entry.
  * ``4``   → **CSA** layer. KV is compressed at ratio ``m = 4``; one cache
    block holds ``k1 = lcm(m, m') / m = 32`` compressed entries.
  * ``0``   → **Non-cached** layer. Used for the MTP module and for pure-SWA
    layers in V4-Pro: no compressed KV cache, only the per-request state cache.
  * (anything else) — currently unused; reserved for future compression ratios.

This module is pure metadata and is the **single source of truth** that the
C++ ``CacheConfig`` / ``BlockPoolConfigHelper`` rewrite (M4 cpp side) must
align with. It does not allocate any tensors.

Two pieces:

  * :class:`LayerCacheKind` — enum naming the four cache types
  * :func:`derive_layer_cache_plan` — turns ``compress_ratios`` + ``m`` + ``m'``
    into ``List[LayerCacheSpec]`` that the cache pool can iterate over

The state cache is global (one fixed-size block per request) and not encoded
per-layer here; see :func:`state_cache_block_size` for its sizing.
"""

from dataclasses import dataclass
from enum import IntEnum
from math import gcd
from typing import List


class LayerCacheKind(IntEnum):
    """Mirrors the cpp enum the cache pool will dispatch on."""

    NON_CACHE = 0   # MTP / SWA-only — no compressed KV cache
    CSA = 1         # m=4 compression, top-k indexer + sparse MQA
    HCA = 2         # m'=128 compression, dense MQA
    SWA_ONLY = 3    # pure sliding-window, no compressor (V4-Pro layer 0/1)


@dataclass(frozen=True)
class LayerCacheSpec:
    """One layer's cache topology row.

    Attributes:
        layer_idx: 0-based transformer block index (matches HF naming).
        kind: which cache pool the layer reads/writes.
        compress_ratio: raw tokens per compressed entry (1 for SWA/non-cache).
        entries_per_block: how many compressed entries fit in one cache block,
            i.e. ``lcm(m, m') / compress_ratio``. ``0`` for non-cached layers.
    """

    layer_idx: int
    kind: LayerCacheKind
    compress_ratio: int
    entries_per_block: int


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def derive_layer_cache_plan(
    compress_ratios: List[int],
    *,
    m: int = 4,
    m_prime: int = 128,
    pure_swa_ratios: tuple = (),
) -> List[LayerCacheSpec]:
    """Convert HF ``compress_ratios`` → per-layer cache spec.

    Args:
        compress_ratios: per-layer ratio table from HF config.
            Each entry must be one of ``{0, m, m'}`` plus values listed in
            ``pure_swa_ratios``. Anything else raises ``ValueError`` so the
            cache pool never silently misroutes a layer.
        m: CSA compression ratio (V4-Flash uses 4).
        m_prime: HCA compression ratio (V4-Flash uses 128).
        pure_swa_ratios: ratios that should be classified as ``SWA_ONLY``
            rather than ``NON_CACHE``. V4-Pro uses this for its first 2
            layers; V4-Flash leaves it empty.

    Returns:
        One :class:`LayerCacheSpec` per layer, in the input order.
    """
    if m <= 0 or m_prime <= 0:
        raise ValueError(f"m and m_prime must be positive, got {m}, {m_prime}")

    block_raw_tokens = lcm(m, m_prime)
    pure_swa_set = set(pure_swa_ratios)
    plan: List[LayerCacheSpec] = []
    for i, r in enumerate(compress_ratios):
        if r == 0:
            plan.append(LayerCacheSpec(i, LayerCacheKind.NON_CACHE, 1, 0))
        elif r in pure_swa_set:
            plan.append(LayerCacheSpec(i, LayerCacheKind.SWA_ONLY, 1, 0))
        elif r == m:
            plan.append(LayerCacheSpec(
                i, LayerCacheKind.CSA, m, block_raw_tokens // m,
            ))
        elif r == m_prime:
            plan.append(LayerCacheSpec(
                i, LayerCacheKind.HCA, m_prime, block_raw_tokens // m_prime,
            ))
        else:
            raise ValueError(
                f"layer {i}: compress_ratio={r} is not one of "
                f"{{0, {m}, {m_prime}}} or pure_swa_ratios={pure_swa_set}"
            )
    return plan


def block_raw_tokens(m: int, m_prime: int) -> int:
    """Raw-token coverage of one cache block (= ``lcm(m, m')``)."""
    return lcm(m, m_prime)


def state_cache_block_size(
    n_win: int,
    head_dim: int,
    *,
    m: int = 4,
    m_prime: int = 128,
    bytes_per_elem: int = 2,
) -> int:
    """Bytes per request-pinned state block.

    The state cache holds the most recent ``n_win`` raw KV (for the SWA
    bypass) plus the un-compressed *tail* of the last partial CSA / HCA
    window (at most ``max(m, m') - 1`` raw tokens for each). Conservative
    upper bound — actual implementation may pack tighter:

      bytes = (n_win + max(m, m') - 1) * (K + V) * head_dim * bytes_per_elem

    K and V are stored separately, so the ``2 *`` accounts for both.
    """
    tail = max(m, m_prime) - 1
    return (n_win + tail) * 2 * head_dim * bytes_per_elem


def total_compressed_blocks(
    plan: List[LayerCacheSpec], num_blocks_per_layer: int,
) -> int:
    """Sum of compressed cache blocks across all *cached* layers.

    Useful when sizing the global pool: non-cached / SWA-only layers
    contribute nothing.
    """
    return sum(
        num_blocks_per_layer
        for spec in plan
        if spec.kind in (LayerCacheKind.CSA, LayerCacheKind.HCA)
    )
