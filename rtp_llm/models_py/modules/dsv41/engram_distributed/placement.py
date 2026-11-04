# SPDX-License-Identifier: Apache-2.0
"""Startup-only, single-copy physical placement within a storage host's row shard.

Storage host count and row ownership never change. Physical allocations divide
the existing padded extent at boundaries valid for BOTH weight and scale VMM.
No replicas, lookup routing, CPU affinity mutation or online migration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import gcd
from pathlib import Path

from rtp_llm.utils.numa import available_memory_nodes, parse_nodes

MAX_LOCAL_SEGMENTS = 8


def parse_numa(value):
    """Keep scalar placement compatible; lists/auto opt into row segmentation."""
    if value is None:
        return 0
    if value == "auto":
        return "auto"
    if type(value) is not str or not value:
        raise ValueError("INVALID_ENGRAM_NUMA_PLACEMENT")
    nodes = tuple(sorted(parse_nodes(value)))
    if len(nodes) > MAX_LOCAL_SEGMENTS:
        raise ValueError("INVALID_ENGRAM_NUMA_PLACEMENT")
    return nodes[0] if len(nodes) == 1 else nodes


def discover_gpu_numas():
    """CUDA ordinals already honor CUDA_VISIBLE_DEVICES; never assume GPU ID=NUMA."""
    from cuda.bindings import driver

    def checked(result):
        if int(result[0]) != 0:
            raise RuntimeError("ENGRAM_CUDA_NUMA_QUERY_FAILED")
        return result[1] if len(result) > 1 else None

    checked(driver.cuInit(0))
    affinity = {}
    for ordinal in range(checked(driver.cuDeviceGetCount())):
        device = checked(driver.cuDeviceGet(ordinal))
        node = int(
            checked(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device
                )
            )
        )
        if node < 0:
            raise ValueError("ENGRAM_HOST_NUMA_UNAVAILABLE")
        affinity[ordinal] = node
    if not affinity:
        raise ValueError("ENGRAM_NO_VISIBLE_GPU")
    return affinity


def resolve_numas(
    value,
    *,
    sysfs=Path("/sys/devices/system/node"),
    status=Path("/proc/self/status"),
    discover=None,
):
    if type(value) is int:
        if not 0 <= value <= 65535:
            raise ValueError("INVALID_ENGRAM_NUMA_PLACEMENT")
        return (value,)
    affinity = (discover or discover_gpu_numas)() if value == "auto" else None
    nodes = tuple(sorted(set(affinity.values()))) if affinity is not None else value
    if (
        type(nodes) is not tuple
        or not 1 <= len(nodes) <= MAX_LOCAL_SEGMENTS
        or any(type(n) is not int or not 0 <= n <= 65535 for n in nodes)
        or len(set(nodes)) != len(nodes)
    ):
        raise ValueError("INVALID_ENGRAM_NUMA_PLACEMENT")
    if not set(nodes) <= available_memory_nodes(sysfs, status):
        raise ValueError("ENGRAM_NUMA_OUTSIDE_CPUSET")
    for node in nodes:
        root = sysfs / f"node{node}"
        total = next(
            (
                int(line.split()[-2])
                for line in (root / "meminfo").read_text().splitlines()
                if "MemTotal:" in line
            ),
            0,
        )
        if total <= 0 or not (root / "cpulist").read_text().strip():
            raise ValueError("ENGRAM_NUMA_REQUIRES_CPU_MEMORY")
    logging.getLogger(__name__).info(
        "engram event=numa_placement mode=single_copy_rows nodes=%s gpu_numa=%s",
        nodes,
        affinity if affinity is not None else "explicit",
    )
    return nodes


def segment_rows(plan, shard, nodes):
    """(location, global begin, global end, padded rows), without extra backing."""
    begin, end = plan.row_range(shard)
    # Weight row width is a multiple of scale row width, so scale alignment
    # also aligns weight. Split whole alignment units, keeping total unchanged.
    unit = plan.granularity // gcd(plan.granularity, plan.dim // plan.block)
    units = plan.shard_rows // unit
    if not nodes or len(nodes) > MAX_LOCAL_SEGMENTS or units < len(nodes):
        raise ValueError("ENGRAM_SHARD_TOO_SMALL_FOR_LOCAL_SPLIT")
    result = []
    for index, node in enumerate(nodes):
        offset = (units * index // len(nodes)) * unit
        stop = (units * (index + 1) // len(nodes)) * unit
        start, finish = begin + offset, min(begin + stop, end)
        if start >= finish:
            raise ValueError("ENGRAM_SHARD_TOO_SMALL_FOR_LOCAL_SPLIT")
        result.append((node, start, finish, stop - offset))
    assert sum(part[3] for part in result) == plan.shard_rows
    return tuple(result)


@dataclass(frozen=True)
class PhysicalAllocation:
    """One actual native allocation, including alignment padding."""

    layer_id: int
    tensor_name: str
    location: int
    row_begin: int
    row_end: int
    padded_rows: int
    columns: int

    @property
    def size_bytes(self):
        return self.padded_rows * self.columns


@dataclass(frozen=True)
class PlacementPlan:
    """Immutable plan shared by admission, allocation, loading and publication."""

    allocations: tuple[PhysicalAllocation, ...]

    @classmethod
    def build(cls, checkpoint, shard, shards, granularity, locations):
        from .shard import plan_shards

        allocations = []
        for layer in checkpoint.layers:
            weight = next(t for t in layer.tensors if t.name.endswith(".weight"))
            rows = segment_rows(
                plan_shards(*weight.shape, shards, granularity=granularity),
                shard,
                locations[layer.layer_id],
            )
            for tensor in layer.tensors:
                for location, begin, end, padded in rows:
                    allocations.append(
                        PhysicalAllocation(
                            layer.layer_id,
                            tensor.name,
                            location,
                            begin,
                            end,
                            padded,
                            tensor.shape[1],
                        )
                    )
        return cls(tuple(allocations))

    @property
    def total_bytes(self):
        return sum(part.size_bytes for part in self.allocations)

    def for_tensor(self, name):
        return tuple(part for part in self.allocations if part.tensor_name == name)

    def bytes_by_location(self):
        required = {}
        for part in self.allocations:
            required[part.location] = required.get(part.location, 0) + part.size_bytes
        return required

    def validate_capacity(self, allocation_limit, budget):
        if any(part.size_bytes > allocation_limit for part in self.allocations):
            raise ValueError("local shard exceeds allocation limit")
        if self.total_bytes > budget:
            raise ValueError("local shards exceed host budget")


def tensor_segments(document, name):
    entry = document["tensors"][name]
    return entry["segments"] if document["schema_version"] == 3 else (entry,)


def owner_groups(tensor):
    """Consecutive physical pieces belonging to each logical storage host."""
    groups = []
    for part in tensor.shards:
        key = (part.owner_id, part.owner_incarnation)
        if not groups or key != (
            groups[-1][0].owner_id,
            groups[-1][0].owner_incarnation,
        ):
            groups.append([])
        groups[-1].append(part)
    return tuple(tuple(group) for group in groups)


def gpu_locations(devices):
    """Validate participant ordinals and return stable GPU UUIDs, without allocation."""
    from uuid import UUID

    from cuda.bindings import driver

    if (
        type(devices) is not tuple
        or not 1 <= len(devices) <= MAX_LOCAL_SEGMENTS
        or any(type(d) is not int or d < 0 for d in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("INVALID_ENGRAM_GPU_DEVICES")

    def checked(result):
        if int(result[0]) != 0:
            raise RuntimeError("ENGRAM_GPU_PLACEMENT_QUERY_FAILED")
        return result[1] if len(result) > 1 else None

    checked(driver.cuInit(0))
    count = checked(driver.cuDeviceGetCount())
    result = {}
    for ordinal in devices:
        if ordinal >= count:
            raise ValueError("ENGRAM_GPU_OUTSIDE_VISIBLE_DEVICES")
        device = checked(driver.cuDeviceGet(ordinal))
        if not checked(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                device,
            )
        ):
            raise ValueError("ENGRAM_GPU_FABRIC_UNSUPPORTED")
        raw = checked(driver.cuDeviceGetUuid(device))
        result[ordinal] = str(UUID(bytes=bytes(raw.bytes)))
    return result


def check_gpu_capacity(required, reserve_bytes):
    """Preflight ALL selected GPUs before acquiring any persistent backing.

    This is current free memory, not a prediction of later model/KV allocations.
    The caller must retain the model's normal KV/runtime budget independently.
    """
    import torch

    if type(reserve_bytes) is not int or not 0 < reserve_bytes < 2**63:
        raise ValueError("INVALID_ENGRAM_GPU_RESERVE")
    for device, needed in sorted(required.items()):
        free, total = torch.cuda.mem_get_info(device)
        logging.getLogger(__name__).info(
            "engram event=gpu_capacity device=%d required_bytes=%d free_bytes=%d total_bytes=%d reserve_bytes=%d",
            device,
            needed,
            free,
            total,
            reserve_bytes,
        )
        if needed > free - reserve_bytes:
            raise ValueError(
                f"ENGRAM_GPU_CAPACITY_INSUFFICIENT device={device} required_bytes={needed} free_bytes={free} reserve_bytes={reserve_bytes}"
            )
