# SPDX-License-Identifier: Apache-2.0
"""Native allocation policy for Fabric host and HBM storage.

The publisher only executes an immutable placement plan. CUDA ordinal/NUMA
resolution, allocation granularity, HBM reserve and native Owner construction
belong here and are never consulted by an online lookup.
"""
from __future__ import annotations

from dataclasses import dataclass

from .placement import check_gpu_capacity, gpu_locations, resolve_numas


def _placement_map(raw, layers, default, label):
    """Complete an explicit per-layer map; partial placement fails loudly."""
    if raw is None:
        return {layer: default for layer in layers}
    if type(raw) is not dict or set(raw) != set(layers):
        raise ValueError(
            f"layer {label} map must cover exactly the model's Engram layers"
        )
    if any(type(node) is not int or node < 0 for node in raw.values()):
        raise ValueError(f"layer {label} IDs must be nonnegative local integers")
    return {layer: raw[layer] for layer in layers}


@dataclass(frozen=True)
class FabricBacking:
    native: object
    memory_kind: str
    locations: dict
    devices: dict
    numas: dict
    nodes: tuple
    segmented: bool
    granularity: int

    @classmethod
    def resolve(
        cls,
        native,
        layers,
        *,
        memory_kind,
        device,
        numa,
        gpu_devices,
        layer_devices=None,
        layer_numas=None,
    ):
        if memory_kind not in ("host", "gpu"):
            raise ValueError("ENGRAM_MEMORY_MUST_BE_HOST_OR_GPU")
        gpu = memory_kind == "gpu"
        if gpu and (layer_devices is not None or layer_numas is not None):
            raise ValueError("GPU backing cannot combine with per-layer host placement")
        locations = gpu_locations(gpu_devices) if gpu else {}
        devices = _placement_map(layer_devices, layers, device, "device")
        segmented = gpu or type(numa) is not int
        if segmented and layer_numas is not None:
            raise ValueError("NUMA row split cannot combine with per-layer placement")
        nodes = tuple(locations) if gpu else resolve_numas(numa)
        numas = _placement_map(layer_numas, layers, nodes[0], "NUMA")
        if gpu and not hasattr(native, "device_granularity"):
            raise RuntimeError("PREBUILT_ENGRAM_GPU_NATIVE_REQUIRED")
        granularities = {
            (
                native.device_granularity(node)
                if gpu
                else native.host_numa_granularity(node, devices[layer])
            )
            for layer in layers
            for node in (nodes if segmented else (numas[layer],))
        }
        if len(granularities) != 1:
            raise ValueError(
                "selected backing locations must share the same allocation granularity"
            )
        return cls(
            native,
            memory_kind,
            locations,
            devices,
            numas,
            nodes,
            segmented,
            granularities.pop(),
        )

    def layer_locations(self, layer):
        return self.nodes if self.segmented else (self.numas[layer],)

    def target_device(self, allocation):
        return (
            allocation.location
            if self.memory_kind == "gpu"
            else self.devices[allocation.layer_id]
        )

    def validate_capacity(self, plan, reserve):
        if self.memory_kind == "gpu":
            check_gpu_capacity(plan.bytes_by_location(), reserve)

    def allocate(self, allocation):
        import torch

        gpu = self.memory_kind == "gpu"
        device = self.target_device(allocation)
        with torch.cuda.device(device):
            return self.native.Owner(
                allocation.size_bytes,
                -1 if gpu else allocation.location,
                device,
                **({"memory_kind": "gpu"} if gpu else {}),
            )

    def descriptor_location(self, location):
        if self.memory_kind == "gpu":
            return {"device_uuid": self.locations[location]}
        return {"numa_node": location} if self.segmented else {}

    @property
    def descriptor_fields(self):
        return {"memory_kind": "gpu"} if self.memory_kind == "gpu" else {}
