# SPDX-License-Identifier: Apache-2.0
"""Distributed storage P is independent of model instances and TP/DP/EP.

No role election, endpoints, sidecar or implicit single-owner deployment.
Startup exchange delivers one immutable in-memory package.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from .placement import MAX_LOCAL_SEGMENTS, owner_groups

ENV_COUNT = "ENGRAM_SHARD_COUNT"
MAX_SHARDS = 64  # manifest ABI limit, not an NVL rack size


def shared_storage_requested(environ):
    return StoragePolicy.from_environ(environ).shared


class ShardConfigError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class StorageMode(str, Enum):
    HOST = "host"
    HOST_SHARED = "host_shared"
    GPU_SHARED = "gpu_shared"


@dataclass(frozen=True)
class StoragePolicy:
    """One storage choice; topology and physical placement do not enable it.

    The canonical selector takes precedence over legacy mode/enable switches,
    allowing a deployment to switch back to local host storage without deleting
    its cohort topology. Old launch configurations retain their semantics.
    """

    mode: StorageMode
    explicit: bool = False

    @property
    def shared(self):
        return self.mode != StorageMode.HOST

    @property
    def memory_kind(self):
        return "gpu" if self.mode == StorageMode.GPU_SHARED else "host"

    @property
    def backend(self):
        return "fabric_gpu" if self.memory_kind == "gpu" else "fabric_egm"

    @property
    def default_graph_mode(self):
        # Explicit storage modes own their mappings for the worker lifetime.
        # Preserve the old opt-in contract for legacy launch configurations.
        return "process_lifetime" if self.explicit and self.shared else "eager"

    @classmethod
    def from_environ(cls, environ=None):
        env = os.environ if environ is None else environ
        if "RTP_ENGRAM_STORAGE" in env:
            try:
                return cls(StorageMode(env["RTP_ENGRAM_STORAGE"]), explicit=True)
            except (ValueError, TypeError):
                raise ShardConfigError("INVALID_ENGRAM_STORAGE_MODE") from None
        kind = env.get("RTP_ENGRAM_MEMORY", "host")
        if kind not in ("host", "gpu"):
            raise ShardConfigError("ENGRAM_MEMORY_MUST_BE_HOST_OR_GPU")
        shared = ENV_COUNT in env or "RTP_ENGRAM_COHORT" in env
        if shared:
            return cls(
                StorageMode.GPU_SHARED if kind == "gpu" else StorageMode.HOST_SHARED
            )
        if kind == "gpu":
            raise ShardConfigError("LEGACY_GPU_MEMORY_REQUIRES_SHARED_STORAGE")
        return cls(StorageMode.HOST)


def validate_count(value):
    if type(value) is not int or not 2 <= value <= MAX_SHARDS:
        raise ShardConfigError("STORAGE_HOST_COUNT_MUST_BE_2_TO_64")
    return value


@dataclass(frozen=True)
class ShardConfig:
    shard_count: int

    def __post_init__(self):
        validate_count(self.shard_count)

    @classmethod
    def from_environ(cls, environ=None, *, inferred_count=None):
        env = os.environ if environ is None else environ
        if any(
            key in env
            for key in (
                "ENGRAM_SHARD_INDEX",
                "ENGRAM_SHARD_MODE",
                "ENGRAM_SHARD_ENDPOINTS",
            )
        ):
            raise ShardConfigError("REMOVED_SHARD_DEPLOYMENT_OPTIONS")
        raw = env.get(ENV_COUNT)
        if raw is None:
            return cls(inferred_count)
        if type(raw) is not str or not raw.isascii() or not raw.isdecimal():
            raise ShardConfigError("INVALID_STORAGE_HOST_COUNT")
        return cls(int(raw))

    def validate_manifest(self, manifest):
        expected_owners = None
        for tensor in manifest.tensors:
            groups = owner_groups(tensor)
            if len(groups) != self.shard_count:
                raise ShardConfigError("STORAGE_HOST_COUNT_MISMATCH")
            owners = tuple((g[0].owner_id, g[0].owner_incarnation) for g in groups)
            if len({owner[0] for owner in owners}) != self.shard_count:
                raise ShardConfigError("STORAGE_HOSTS_MUST_DIFFER")
            for group in groups:
                if manifest.schema_version == 1 and len(group) != 1:
                    raise ShardConfigError("STORAGE_HOSTS_MUST_DIFFER")
                if manifest.schema_version == 2 and (
                    len(group) > MAX_LOCAL_SEGMENTS
                    or len(
                        {
                            (
                                s.device_uuid
                                if manifest.backend == "fabric_gpu"
                                else s.numa_node
                            )
                            for s in group
                        }
                    )
                    != len(group)
                ):
                    raise ShardConfigError("INVALID_NUMA_SEGMENTS")
            if expected_owners is None:
                expected_owners = owners
            elif owners != expected_owners:
                raise ShardConfigError("INCONSISTENT_STORAGE_HOSTS")
        if expected_owners is None:
            raise ShardConfigError("EMPTY_STORAGE_GROUP")
        return self


def storage_host_count(manifest):
    return len(owner_groups(manifest.tensors[0]))


def memory_kind_from_environ(environ=None):
    env = os.environ if environ is None else environ
    if "RTP_ENGRAM_STORAGE" in env:
        return StoragePolicy.from_environ(env).memory_kind
    kind = env.get("RTP_ENGRAM_MEMORY", "host")
    if kind not in ("host", "gpu"):
        raise ShardConfigError("ENGRAM_MEMORY_MUST_BE_HOST_OR_GPU")
    return kind
