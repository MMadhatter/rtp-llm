# SPDX-License-Identifier: Apache-2.0
"""Map N Engram shards into two contiguous VA spans.

The lookup kernel is unchanged: it still indexes ``weight + row * dim`` and
``scales + row * (dim // block)``.  This module only arranges the address space
so that global row ``r`` lands in the shard that owns it.

Driver calls are behind :class:`DriverOps` so the layout and rollback logic can
be tested on a CPU-only machine with a fake recorder.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from .engram_shard import MappingRequest, ShardPlan, mapping_requests


class DriverOps(Protocol):
    def reserve(self, size: int) -> int: ...

    def import_handle(self, raw: bytes) -> int: ...

    def map(self, va: int, size: int, offset: int, handle: int) -> None: ...

    def unmap(self, va: int, size: int) -> None: ...

    def free(self, va: int, size: int) -> None: ...

    def view(self, va: int, shape: tuple[int, ...]): ...


class CudaDriverOps:
    """Thin wrapper over ``cuda.bindings.driver``; every call is checked."""

    def __init__(self, device: int):
        from cuda.bindings import driver

        self._driver = driver
        self.device = device

    def _check(self, result, what: str):
        error = result[0] if isinstance(result, tuple) else result
        if int(error) != 0:
            raise RuntimeError(f"engram shard {what} failed: {int(error)}")
        return result[1] if isinstance(result, tuple) and len(result) > 1 else None

    def reserve(self, size: int) -> int:
        pointer = self._check(
            self._driver.cuMemAddressReserve(size, 0, 0, 0), "address_reserve"
        )
        return int(pointer)

    def import_handle(self, raw: bytes) -> int:
        handle_type = self._driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        handle = self._check(
            self._driver.cuMemImportFromShareableHandle(raw, handle_type),
            "import_handle",
        )
        return int(handle)

    def map(self, va: int, size: int, offset: int, handle: int) -> None:
        self._check(self._driver.cuMemMap(va, size, offset, handle, 0), "map")

    def unmap(self, va: int, size: int) -> None:
        self._check(self._driver.cuMemUnmap(va, size), "unmap")

    def free(self, va: int, size: int) -> None:
        self._check(self._driver.cuMemAddressFree(va, size), "address_free")

    def view(self, va: int, shape: tuple[int, ...]):
        import torch

        class _ArrayView:
            def __init__(self, pointer, tensor_shape):
                self.__cuda_array_interface__ = {
                    "shape": tensor_shape,
                    "strides": None,
                    "typestr": "|u1",
                    "data": (pointer, False),
                    "version": 3,
                }

        return torch.as_tensor(_ArrayView(va, shape), device=f"cuda:{self.device}")


class ShardedLayerMapping:
    """One tensor (layer) mapped from ``shard_count`` fabric handles."""

    def __init__(self, plan: ShardPlan, handles: Sequence[bytes], device: int, ops=None):
        if len(handles) != plan.shard_count:
            raise ValueError("handle count must equal shard_count")
        if any(not isinstance(handle, bytes) or not handle for handle in handles):
            raise ValueError("fabric handles must be non-empty bytes")
        self.plan = plan
        self.handles = tuple(handles)
        self.device = device
        self.ops = ops if ops is not None else CudaDriverOps(device)
        self._weight_ptr = 0
        self._scale_ptr = 0
        self._mapped: list[MappingRequest] = []
        self._imported: list[int] = []
        self._state = "NEW"

    @property
    def state(self) -> str:
        return self._state

    def attach(self) -> "ShardedLayerMapping":
        if self._state != "NEW":
            raise RuntimeError("shard mapping can only be attached once")
        try:
            self._weight_ptr = self.ops.reserve(self.plan.weight_span)
            self._scale_ptr = self.ops.reserve(self.plan.scale_span)
            for request in mapping_requests(self.plan, self._weight_ptr, self._scale_ptr):
                handle = self.ops.import_handle(self.handles[request.shard])
                self._imported.append(handle)
                self.ops.map(request.va, request.size, request.handle_offset, handle)
                self._mapped.append(request)
            self._state = "MAPPED"
            return self
        except BaseException:
            self.close()
            raise

    def _require_mapped(self):
        if self._state != "MAPPED":
            raise RuntimeError("shard mapping is not MAPPED")

    def weight_view(self):
        self._require_mapped()
        return self.ops.view(self._weight_ptr, (self.plan.padded_rows, self.plan.dim))

    def scale_view(self):
        self._require_mapped()
        return self.ops.view(
            self._scale_ptr, (self.plan.padded_rows, self.plan.dim // self.plan.block)
        )

    def close(self) -> None:
        """Unmap in reverse order, then free the two reservations."""
        for request in reversed(self._mapped):
            self.ops.unmap(request.va, request.size)
        self._mapped.clear()
        if self._weight_ptr:
            self.ops.free(self._weight_ptr, self.plan.weight_span)
            self._weight_ptr = 0
        if self._scale_ptr:
            self.ops.free(self._scale_ptr, self.plan.scale_span)
            self._scale_ptr = 0
        self._imported.clear()
        self._state = "CLOSED"
