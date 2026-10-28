# SPDX-License-Identifier: Apache-2.0
"""Shard layout for distributed Engram storage over EGM.

The Engram table is split into ``shard_count`` contiguous row ranges, one per
holder process.  Every consumer maps all shards into **two** contiguous virtual
address ranges (weights and scales), so the existing lookup kernel keeps using
``weight + row * dim`` and ``scales + row * (dim // block)`` unchanged.

For that trick to work both the per-shard weight region and the per-shard scale
region must start at a CUDA allocation-granularity boundary.  ``shard_rows`` is
therefore rounded up so that

    shard_rows * dim            % granularity == 0
    shard_rows * (dim // block) % granularity == 0

which for dim=256, block=32, granularity=2 MiB means ``shard_rows`` must be a
multiple of 262144 rows.  The padding is at most one alignment unit per shard
(64 MiB of weights) and is never addressed by a lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

DEFAULT_GRANULARITY = 2 * 1024 * 1024
DEFAULT_BLOCK = 32
DEFAULT_SHARD_COUNT = 4


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def _round_up(value: int, unit: int) -> int:
    if unit <= 0:
        raise ValueError("alignment unit must be positive")
    return -(-value // unit) * unit


@dataclass(frozen=True)
class ShardPlan:
    """Immutable geometry of one sharded Engram tensor."""

    total_rows: int
    shard_count: int
    shard_rows: int
    dim: int
    block: int
    granularity: int

    @property
    def weight_bytes(self) -> int:
        return self.shard_rows * self.dim

    @property
    def scale_bytes(self) -> int:
        return self.shard_rows * (self.dim // self.block)

    @property
    def shard_bytes(self) -> int:
        return self.weight_bytes + self.scale_bytes

    @property
    def padded_rows(self) -> int:
        return self.shard_rows * self.shard_count

    @property
    def weight_span(self) -> int:
        """Size of the concatenated weight VA range."""
        return self.weight_bytes * self.shard_count

    @property
    def scale_span(self) -> int:
        """Size of the concatenated scale VA range."""
        return self.scale_bytes * self.shard_count

    def shard_of_row(self, row: int) -> int:
        if not 0 <= row < self.padded_rows:
            raise IndexError(f"row {row} outside padded table of {self.padded_rows}")
        return row // self.shard_rows

    def row_range(self, shard: int) -> tuple[int, int]:
        """Half-open *real* row range owned by ``shard`` (padding excluded)."""
        if not 0 <= shard < self.shard_count:
            raise IndexError(f"shard {shard} outside 0..{self.shard_count - 1}")
        start = shard * self.shard_rows
        end = min(start + self.shard_rows, self.total_rows)
        return start, end

    def weight_offset(self, shard: int) -> int:
        """Byte offset of ``shard`` inside the concatenated weight VA."""
        if not 0 <= shard < self.shard_count:
            raise IndexError(f"shard {shard} outside 0..{self.shard_count - 1}")
        return shard * self.weight_bytes

    def scale_offset(self, shard: int) -> int:
        """Byte offset of ``shard`` inside the concatenated scale VA."""
        if not 0 <= shard < self.shard_count:
            raise IndexError(f"shard {shard} outside 0..{self.shard_count - 1}")
        return shard * self.scale_bytes

    def as_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "shard_count": self.shard_count,
            "shard_rows": self.shard_rows,
            "dim": self.dim,
            "block": self.block,
            "granularity": self.granularity,
            "weight_bytes": self.weight_bytes,
            "scale_bytes": self.scale_bytes,
            "shard_bytes": self.shard_bytes,
            "weight_span": self.weight_span,
            "scale_span": self.scale_span,
        }


def plan_shards(
    total_rows: int,
    dim: int,
    shard_count: int = DEFAULT_SHARD_COUNT,
    *,
    block: int = DEFAULT_BLOCK,
    granularity: int = DEFAULT_GRANULARITY,
) -> ShardPlan:
    """Compute the shard geometry, padding rows so every shard is mappable.

    Raises ``ValueError`` for geometries that cannot be mapped at a granularity
    boundary (for example a ``dim`` that is not a multiple of ``block``).
    """
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if dim <= 0 or dim % block:
        raise ValueError("dim must be a positive multiple of block")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if granularity <= 0 or granularity & (granularity - 1):
        raise ValueError("granularity must be a positive power of two")

    weight_unit = granularity // gcd(granularity, dim)
    scale_unit = granularity // gcd(granularity, dim // block)
    align_rows = _lcm(weight_unit, scale_unit)

    rows_per_shard = -(-total_rows // shard_count)
    shard_rows = _round_up(rows_per_shard, align_rows)
    plan = ShardPlan(
        total_rows=total_rows,
        shard_count=shard_count,
        shard_rows=shard_rows,
        dim=dim,
        block=block,
        granularity=granularity,
    )
    # Post-conditions the native mapping relies on.
    assert plan.weight_bytes % granularity == 0
    assert plan.scale_bytes % granularity == 0
    assert plan.shard_bytes % granularity == 0
    assert plan.padded_rows >= total_rows
    return plan


def split_rows(total_rows: int, plan: ShardPlan) -> list[tuple[int, int]]:
    """Real row ranges for every shard, in shard order."""
    return [plan.row_range(shard) for shard in range(plan.shard_count)]


def local_rows(plan: ShardPlan, shard_index: int) -> tuple[int, int]:
    """Rows the process at ``shard_index`` must load (``-1`` means none)."""
    if shard_index < 0:
        return (0, 0)
    return plan.row_range(shard_index)
