# SPDX-License-Identifier: Apache-2.0
"""Process-level configuration for distributed Engram shards.

Two deployment modes share the same knobs:

``prefill``  every prefill instance holds one shard; instances fetch each
             other's handles.  ``shard_count`` equals the number of instances.
``decode``   one decode instance splits the table across its own parts; the
             parts fetch each other's handles inside the instance.

``shard_index == -1`` makes a process a pure consumer: it maps every shard but
holds none.  That is what a dynamically added instance uses when the shard
count is already fixed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

ENV_COUNT = "ENGRAM_SHARD_COUNT"
ENV_INDEX = "ENGRAM_SHARD_INDEX"
ENV_ENDPOINTS = "ENGRAM_SHARD_ENDPOINTS"
ENV_MODE = "ENGRAM_SHARD_MODE"

MODES = ("prefill", "decode")
# Absent env means "no sharing" so existing single-copy deployments keep
# working; the recommended deployment value for a shared group is 4.
DEFAULT_COUNT = 1


class ShardConfigError(ValueError):
    """Bounded configuration error; never echoes endpoint credentials."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _parse_int(raw: str, name: str) -> int:
    try:
        return int(raw, 10)
    except (TypeError, ValueError):
        raise ShardConfigError(f"INVALID_{name}") from None


def _parse_endpoints(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    endpoints = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            raise ShardConfigError("INVALID_ENDPOINTS")
        host, _, port = item.rpartition(":")
        if not host or not port.isdigit():
            raise ShardConfigError("INVALID_ENDPOINTS")
        if not 0 < int(port) < 65536:
            raise ShardConfigError("INVALID_ENDPOINTS")
        endpoints.append(f"{host}:{int(port)}")
    if len(set(endpoints)) != len(endpoints):
        raise ShardConfigError("DUPLICATE_ENDPOINTS")
    return tuple(endpoints)


@dataclass(frozen=True)
class ShardConfig:
    shard_count: int
    shard_index: int
    endpoints: tuple[str, ...]
    mode: str = "prefill"

    @property
    def holds_shard(self) -> bool:
        return self.shard_index >= 0

    @property
    def sharded(self) -> bool:
        return self.shard_count > 1

    def endpoint(self, shard: int) -> str:
        if not 0 <= shard < self.shard_count:
            raise ShardConfigError("SHARD_OUT_OF_RANGE")
        if len(self.endpoints) != self.shard_count:
            raise ShardConfigError("ENDPOINTS_MISSING")
        return self.endpoints[shard]

    def validate(self) -> "ShardConfig":
        if self.mode not in MODES:
            raise ShardConfigError("INVALID_MODE")
        if self.shard_count < 1:
            raise ShardConfigError("INVALID_SHARD_COUNT")
        if not -1 <= self.shard_index < self.shard_count:
            raise ShardConfigError("INVALID_SHARD_INDEX")
        if self.shard_count > 1 and len(self.endpoints) != self.shard_count:
            raise ShardConfigError("ENDPOINTS_MISSING")
        if self.shard_count == 1 and self.endpoints and len(self.endpoints) != 1:
            raise ShardConfigError("ENDPOINTS_MISMATCH")
        if self.shard_index >= 0 and self.endpoints and len(self.endpoints) != self.shard_count:
            raise ShardConfigError("ENDPOINTS_MISMATCH")
        return self

    @classmethod
    def from_environ(cls, environ=None, *, default_count: int = DEFAULT_COUNT) -> "ShardConfig":
        env = os.environ if environ is None else environ
        count = _parse_int(env.get(ENV_COUNT, str(default_count)), "SHARD_COUNT")
        index = _parse_int(env.get(ENV_INDEX, "0"), "SHARD_INDEX")
        endpoints = _parse_endpoints(env.get(ENV_ENDPOINTS, ""))
        mode = env.get(ENV_MODE, "prefill").strip() or "prefill"
        return cls(shard_count=count, shard_index=index, endpoints=endpoints, mode=mode).validate()

    def describe(self) -> dict:
        """Safe representation: no handle material, endpoints are host:port only."""
        return {
            "shard_count": self.shard_count,
            "shard_index": self.shard_index,
            "holds_shard": self.holds_shard,
            "sharded": self.sharded,
            "mode": self.mode,
            "endpoints": list(self.endpoints),
        }
