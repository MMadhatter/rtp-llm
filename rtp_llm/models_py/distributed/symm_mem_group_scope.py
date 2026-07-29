"""Resolve the CUDA handle policy from the actual SymmetricMemory group.

The multicast keeper owns CUDA object lifetime; it must not infer collective
topology from process placement.  This module is the small bridge between the
two layers.  A rank describes the real ProcessGroup immediately before a
SymmetricMemory allocation and this module publishes the resulting process-wide
handle policy to the preloaded keeper shim.

PyTorch's CUDA symmetric-memory allocator caches its handle type for the
process lifetime.  Consequently a Level3 rank cannot first allocate a
single-node POSIX group and later allocate a cross-node FABRIC group.  We reject
that unsupported mix at startup instead of silently selecting the wrong CUDA
handle type.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

import torch.distributed as dist

KEEPER_ENABLE_ENV = "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER"
LOCAL_GPUS_ENV = "RTP_LLM_MC_LOCAL_GPUS"
FABRIC_TEAM_SIZE_ENV = "RTP_LLM_MC_FABRIC_TEAM_SIZE"
HANDLE_POLICY_ENV = "RTP_LLM_MC_SYMM_MEM_HANDLE_POLICY"


class SymmMemGroupScope(str, Enum):
    LOCAL = "local"
    CROSS_NODE = "cross_node"


class SymmMemHandlePolicy(str, Enum):
    LOCAL_POSIX = "local_posix"
    FABRIC = "fabric"


@dataclass(frozen=True)
class SymmMemGroupDecision:
    owner: str
    ranks: Tuple[int, ...]
    local_ranks: Tuple[int, ...]
    scope: SymmMemGroupScope
    policy: SymmMemHandlePolicy


@dataclass(frozen=True)
class _RankTopology:
    world_rank: int
    local_rank: int
    local_world_size: int
    world_size: int

    @property
    def local_ranks(self) -> Tuple[int, ...]:
        first = self.world_rank - self.local_rank
        return tuple(
            rank
            for rank in range(first, min(first + self.local_world_size, self.world_size))
        )


_lock = threading.RLock()
_rank_topology: Optional[_RankTopology] = None
_selected_policy: Optional[SymmMemHandlePolicy] = None


def _keeper_enabled() -> bool:
    return os.environ.get(KEEPER_ENABLE_ENV, "0") == "1"


def configure_rank_topology(parallelism_config: Any) -> None:
    """Record this rank's node boundary before any SymmMem allocation."""

    global _rank_topology
    if not _keeper_enabled():
        return
    topology = _RankTopology(
        world_rank=int(parallelism_config.world_rank),
        local_rank=int(parallelism_config.local_rank),
        local_world_size=int(parallelism_config.local_world_size),
        world_size=int(parallelism_config.world_size),
    )
    if (
        topology.world_size <= 0
        or topology.local_world_size <= 0
        or topology.local_world_size > topology.world_size
        or topology.world_rank < 0
        or topology.world_rank >= topology.world_size
        or topology.local_rank < 0
        or topology.local_rank >= topology.local_world_size
    ):
        raise RuntimeError(f"invalid rank topology for SymmMem: {topology}")
    with _lock:
        if _rank_topology is not None and _rank_topology != topology:
            raise RuntimeError(
                "SymmMem rank topology changed within one process: "
                f"old={_rank_topology}, new={topology}"
            )
        _rank_topology = topology


def _parse_local_gpu_count() -> int:
    value = os.environ.get(LOCAL_GPUS_ENV, "")
    fields = [field.strip() for field in value.split(",") if field.strip()]
    if not fields:
        raise RuntimeError(f"{LOCAL_GPUS_ENV} is missing for multicast keeper")
    return len(fields)


def _infer_rank_topology() -> _RankTopology:
    """Fallback for standalone tests which initialize torch.distributed directly."""

    world_rank = int(os.environ.get("RANK", dist.get_rank()))
    world_size = int(os.environ.get("WORLD_SIZE", dist.get_world_size()))
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", _parse_local_gpu_count())
    )
    local_rank = int(os.environ.get("LOCAL_RANK", world_rank % local_world_size))
    return _RankTopology(
        world_rank=world_rank,
        local_rank=local_rank,
        local_world_size=local_world_size,
        world_size=world_size,
    )


def _group_ranks(group: Any) -> Tuple[int, ...]:
    try:
        ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    except (AttributeError, RuntimeError):
        if group is not dist.group.WORLD:
            raise RuntimeError(
                "cannot resolve ranks for the SymmMem ProcessGroup"
            ) from None
        ranks = tuple(range(dist.get_world_size(group)))
    if not ranks:
        raise RuntimeError("SymmMem ProcessGroup has no ranks")
    return ranks


def _validate_keeper_contract(
    scope: SymmMemGroupScope,
    ranks: Tuple[int, ...],
    local_ranks: Tuple[int, ...],
) -> None:
    """Reject layouts the current node-local holder cannot represent safely."""

    if scope is SymmMemGroupScope.LOCAL:
        local_gpu_count = _parse_local_gpu_count()
        if len(ranks) != local_gpu_count or set(ranks) != set(local_ranks):
            raise RuntimeError(
                "Level3 multicast keeper currently requires a local SymmMem "
                "group to cover every rank/GPU assigned to this node: "
                f"group_ranks={list(ranks)}, local_ranks={list(local_ranks)}, "
                f"keeper_local_gpu_count={local_gpu_count}"
            )
        return

    try:
        fabric_team_size = int(os.environ[FABRIC_TEAM_SIZE_ENV])
    except (KeyError, ValueError):
        raise RuntimeError(
            f"{FABRIC_TEAM_SIZE_ENV} must contain the explicit FABRIC group size"
        ) from None
    if len(ranks) != fabric_team_size:
        raise RuntimeError(
            "Level3 multicast keeper FABRIC contract does not match the actual "
            f"SymmMem group: group_ranks={list(ranks)}, "
            f"configured_team_size={fabric_team_size}"
        )


def configure_group_scope(
    group: Any, *, owner: str
) -> Optional[SymmMemGroupDecision]:
    """Select the keeper handle policy from an actual ProcessGroup.

    This is an initialization-only operation and has no inference/request hot
    path cost.  It is a no-op when the Level3 keeper is not enabled.
    """

    global _selected_policy
    if not _keeper_enabled():
        return None
    if not dist.is_initialized():
        raise RuntimeError(
            f"cannot configure SymmMem group scope for {owner}: "
            "torch.distributed is not initialized"
        )

    with _lock:
        topology = _rank_topology or _infer_rank_topology()
        ranks = _group_ranks(group)
        local_ranks = topology.local_ranks
        scope = (
            SymmMemGroupScope.LOCAL
            if set(ranks).issubset(local_ranks)
            else SymmMemGroupScope.CROSS_NODE
        )
        requested_policy = (
            SymmMemHandlePolicy.LOCAL_POSIX
            if scope is SymmMemGroupScope.LOCAL
            else SymmMemHandlePolicy.FABRIC
        )
        _validate_keeper_contract(scope, ranks, local_ranks)

        if _selected_policy is None:
            _selected_policy = requested_policy
            os.environ[HANDLE_POLICY_ENV] = requested_policy.value
            logging.info(
                "[MulticastKeeper][SymmMemScope] selected process policy=%s "
                "owner=%s scope=%s group_ranks=%s local_ranks=%s",
                requested_policy.value,
                owner,
                scope.value,
                list(ranks),
                list(local_ranks),
            )
        elif _selected_policy is not requested_policy:
            raise RuntimeError(
                "one process cannot mix local POSIX and cross-node FABRIC "
                "SymmMem groups under PyTorch's process-wide CUDA symmetric-"
                "memory allocator: "
                f"selected_policy={_selected_policy.value}, owner={owner}, "
                f"requested_scope={scope.value}, group_ranks={list(ranks)}"
            )
        else:
            logging.info(
                "[MulticastKeeper][SymmMemScope] confirmed process policy=%s "
                "owner=%s scope=%s group_ranks=%s",
                _selected_policy.value,
                owner,
                scope.value,
                list(ranks),
            )

        return SymmMemGroupDecision(
            owner=owner,
            ranks=ranks,
            local_ranks=local_ranks,
            scope=scope,
            policy=_selected_policy,
        )


def _release_fabric_backing_fences(owner: str) -> int:
    import ctypes

    try:
        release = ctypes.CDLL(None).rtp_llm_mc_release_fabric_backings
    except AttributeError:
        raise RuntimeError(
            "multicast keeper shim does not export "
            "rtp_llm_mc_release_fabric_backings"
        ) from None
    release.argtypes = []
    release.restype = ctypes.c_int
    released = int(release())
    if released < 0:
        raise RuntimeError(
            f"failed to release FABRIC backing creator fences for {owner}"
        )
    logging.info(
        "[MulticastKeeper][SymmMemScope] released %d FABRIC backing "
        "creator fence(s) owner=%s",
        released,
        owner,
    )
    return released


@contextmanager
def symm_mem_allocation_scope(group: Any, *, owner: str):
    """Keep broker creators alive exactly through cross-node rendezvous.

    The post-rendezvous group barrier guarantees every peer has imported all
    raw FABRIC identities before their exporting helper processes exit. This is
    initialization/wake-only coordination; it is not on the inference hot path.
    """

    decision = configure_group_scope(group, owner=owner)
    fabric = decision is not None and decision.policy is SymmMemHandlePolicy.FABRIC
    try:
        yield decision
    except BaseException:
        if fabric:
            try:
                _release_fabric_backing_fences(owner)
            except Exception:
                logging.exception(
                    "[MulticastKeeper][SymmMemScope] failed to abort FABRIC "
                    "backing creators owner=%s",
                    owner,
                )
        raise
    if fabric:
        try:
            dist.barrier(group=group)
        except BaseException:
            try:
                _release_fabric_backing_fences(owner)
            except Exception:
                logging.exception(
                    "[MulticastKeeper][SymmMemScope] failed to release FABRIC "
                    "backing creators after barrier failure owner=%s",
                    owner,
                )
            raise
        else:
            _release_fabric_backing_fences(owner)


def _reset_for_test() -> None:
    global _rank_topology, _selected_policy
    with _lock:
        _rank_topology = None
        _selected_policy = None
        os.environ.pop(HANDLE_POLICY_ENV, None)


__all__ = [
    "HANDLE_POLICY_ENV",
    "SymmMemGroupDecision",
    "SymmMemGroupScope",
    "SymmMemHandlePolicy",
    "configure_group_scope",
    "configure_rank_topology",
    "symm_mem_allocation_scope",
]
