# SPDX-License-Identifier: Apache-2.0
"""Explicit EGM attach and graph lifetimes; no RPC or checkpoint loading.

Only operations submitted under ``use()`` participate in the drain contract.
Raw tensor views must not be captured in an unmanaged CUDA graph. This first
slice uses an injected native backend; it never compiles an extension in forward.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, Protocol

from .config import ShardConfig, storage_host_count
from .manifest import (
    AllocationSlice,
    ExpectedLayout,
    Limits,
    TensorSpec,
    ValidatedManifest,
    parse_for_attach,
)
from .placement import owner_groups
from .shard import ShardPlan, plan_shards

_LOGGER = logging.getLogger(__name__)


class _ManagedMapping(Protocol):
    """Both native allocation imports and stitched layer spans own leases."""

    def info(self) -> dict: ...

    def close(self): ...


@dataclass(frozen=True)
class _LayerPlan:
    weight: TensorSpec
    scale: TensorSpec
    shards: ShardPlan
    segment_rows: tuple[int, ...]


@dataclass(frozen=True)
class _AttachPlan:
    validated: ValidatedManifest
    allocations: tuple[AllocationSlice, ...]
    layers: tuple[_LayerPlan, ...]

    @classmethod
    def build(cls, validated):
        """Finish all geometry checks before any native resource is acquired."""
        if not isinstance(validated, ValidatedManifest):
            raise TypeError("attach execution requires a validated manifest")
        manifest = validated.manifest
        if manifest.backend not in ("fabric_egm", "fabric_gpu"):
            raise ValueError("FabricEngramStorage requires fabric_egm")
        allocations = {
            part.allocation_id: part
            for tensor in manifest.tensors
            for part in tensor.shards
        }
        layers = []
        if any(len(t.shards) > 1 for t in manifest.tensors):
            count = storage_host_count(manifest)
            ShardConfig(count).validate_manifest(manifest)
            by_name = {t.name: t for t in manifest.tensors}
            for weight in manifest.tensors:
                if not weight.name.endswith(".weight"):
                    continue
                scale = by_name[weight.name.removesuffix(".weight") + ".scale"]
                shards = plan_shards(
                    total_rows=weight.shape[0],
                    shard_count=count,
                    dim=weight.shape[1],
                    block=weight.quant_block,
                    granularity=weight.shards[0].allocation_granularity,
                )
                for tensor, size in (
                    (weight, shards.weight_bytes),
                    (scale, shards.scale_bytes),
                ):
                    for index, parts in enumerate(owner_groups(tensor)):
                        begin, end = shards.row_range(index)
                        cursor, allocated = begin, 0
                        for part in parts:
                            if (
                                part.allocation_granularity != shards.granularity
                                or part.byte_offset != 0
                                or part.row_begin != cursor
                                or part.row_end > end
                                or (
                                    part.row_end < end
                                    and part.logical_bytes != part.allocation_bytes
                                )
                            ):
                                raise ValueError(
                                    "shard geometry does not match the mapping plan"
                                )
                            cursor = part.row_end
                            allocated += part.allocation_bytes
                        if cursor != end or allocated != size:
                            raise ValueError(
                                "shard geometry: NUMA segments must exactly partition a host shard"
                            )
                rows = tuple(
                    s.allocation_bytes // weight.row_bytes for s in weight.shards
                )
                layers.append(_LayerPlan(weight, scale, shards, rows))
        return cls(
            validated,
            tuple(allocations[aid] for aid in sorted(allocations)),
            tuple(layers),
        )


class StorageBusyError(RuntimeError):
    pass


class CleanupPendingError(RuntimeError):
    """Caller must retain/retry this resource or terminate the worker."""

    def __init__(self, message, resource):
        super().__init__(message)
        self.resource = resource


class FabricEngramStorage:
    """One worker's mappings of one immutable, complete Engram generation."""

    def __init__(self):
        self._lock = threading.RLock()
        self._mappings: dict[str, _ManagedMapping] = {}
        self._allocation_count = 0
        self._shard_count = 0
        self._views = {}
        self._graph_pins = 0
        self._lookup_events = []
        self._state = "ATTACHING"

    @property
    def device_resident_tables(self):
        """Both weight and scales already reside in HBM; never cache a copy."""
        return self.memory_kind == "gpu"

    @classmethod
    def attach(
        cls,
        manifest_bytes: bytes,
        *,
        expected_layout: ExpectedLayout,
        dataset_id: str,
        generation: str,
        handles: Mapping[str, bytes],
        device: int,
        native,
        limits: Limits = Limits(),
        shard_ops=None,
    ):
        validated = parse_for_attach(
            manifest_bytes, expected_layout, dataset_id, generation, limits
        )
        return cls._attach_validated(
            validated,
            handles=handles,
            device=device,
            native=native,
            shard_ops=shard_ops,
        )

    @classmethod
    def _attach_validated(
        cls,
        validated: ValidatedManifest,
        *,
        handles: Mapping[str, bytes],
        device: int,
        native,
        shard_ops=None,
    ):
        """Internal execution boundary; raw callers enter through attach()."""
        if type(device) is not int or device < 0:
            raise ValueError("device must be a local CUDA ordinal")
        plan = _AttachPlan.build(validated)
        if not isinstance(handles, Mapping) or set(handles) != {
            s.allocation_id for s in plan.allocations
        }:
            raise ValueError("handle allocation set does not match the manifest")
        if any(
            not isinstance(h, bytes) or not 0 < len(h) <= 1024 for h in handles.values()
        ):
            raise ValueError("Fabric capabilities must be bounded opaque bytes")
        result = cls()
        result.validated = validated
        result.device = device
        result.memory_kind = (
            "gpu" if validated.manifest.backend == "fabric_gpu" else "host"
        )
        result._allocation_count = validated.allocation_count
        result._shard_count = storage_host_count(validated.manifest)
        started = time.monotonic()
        result._log_state("storage_attach_started", "attach", started)
        try:
            result._execute_plan(plan, handles, native, shard_ops)
            result._state = "MAPPED"
            result._log_state("storage_attached", "attach", started)
            return result
        except BaseException as original:
            # No GPU work can have been submitted through this unpublished object.
            result._views.clear()
            failures = []
            for mapping in reversed(list(result._mappings.values())):
                try:
                    mapping.close()
                except Exception as error:
                    failures.append(type(error).__name__)
            result._state = "FAILED"
            result._log_state(
                "storage_attach_failed",
                "attach",
                started,
                original,
                cleanup_pending=bool(failures),
            )
            if failures:
                raise CleanupPendingError(
                    "attach failed and native cleanup was incomplete", result
                ) from original
            raise

    def _execute_plan(self, plan, handles, native, shard_ops):
        if plan.layers:
            from .mapping import LayerMappingContext, ShardedLayerMapping

            for layer in plan.layers:
                mapping = ShardedLayerMapping(
                    layer.shards,
                    [handles[s.allocation_id] for s in layer.weight.shards],
                    [handles[s.allocation_id] for s in layer.scale.shards],
                    self.device,
                    ops=shard_ops,
                    memory_kind=self.memory_kind,
                    context=LayerMappingContext(
                        plan.validated.manifest.generation,
                        int(layer.weight.name.split(".")[1]),
                        tuple(s.owner_id for s in layer.weight.shards),
                        tuple(s.owner_id for s in layer.scale.shards),
                    ),
                    segment_rows=layer.segment_rows,
                )
                # Register ownership before attach: partial failures must be retryable.
                key = layer.weight.name.removesuffix(".weight")
                self._mappings[key] = mapping
                mapping.attach()
                self._views[layer.weight.name] = mapping.weight_view()
                self._views[layer.scale.name] = mapping.scale_view()
        else:
            # Preserve the standalone native Consumer primitive and its slice offsets.
            by_allocation = {
                part.allocation_id: tensor
                for tensor in plan.validated.manifest.tensors
                for part in tensor.shards
            }
            for part in plan.allocations:
                with self._native_operation(
                    by_allocation[part.allocation_id],
                    part,
                    "native_consumer",
                    "allocation",
                ):
                    self._mappings[part.allocation_id] = native.Consumer(
                        handles[part.allocation_id],
                        part.allocation_bytes,
                        part.allocation_granularity,
                        self.device,
                        **({"memory_kind": "gpu"} if self.memory_kind == "gpu" else {}),
                    )
            for tensor in plan.validated.manifest.tensors:
                part = tensor.shards[0]
                with self._native_operation(
                    tensor, part, "tensor_view", tensor.name.rsplit(".", 1)[1]
                ):
                    self._views[tensor.name] = self._mappings[
                        part.allocation_id
                    ].tensor_view(list(tensor.shape), part.byte_offset)

    @contextmanager
    def _native_operation(self, tensor, part, operation, kind):
        """Add safe identity at the native boundary without changing exceptions."""
        try:
            yield
        except BaseException as error:
            try:
                _LOGGER.error(
                    "engram event=mapping_failed phase=attach generation=%s layer=%s "
                    "tensor_kind=%s shard=0 owner_sha256=%s device=%s operation=%s error_class=%s",
                    self.validated.manifest.generation,
                    int(tensor.name.split(".")[1]),
                    kind,
                    hashlib.sha256(part.owner_id.encode()).hexdigest()[:16],
                    self.device,
                    operation,
                    type(error).__name__[:64],
                )
            except Exception:
                pass
            raise

    def _log_state(self, event, phase, started, error=None, *, cleanup_pending=False):
        # Called only at initialization transitions. Do not walk native mappings
        # or perform CUDA work to collect diagnostics; failures are best effort.
        try:
            _LOGGER.log(
                logging.INFO if error is None else logging.ERROR,
                "engram event=%s phase=%s state=%s generation=%s device=%s shard_count=%s memory=%s "
                "physical_bytes=%s allocation_count=%s mapping_count=%s graph_pins=%s "
                "duration_ms=%.3f error_class=%s cleanup_pending=%s",
                event,
                phase,
                self._state,
                self.validated.manifest.generation,
                self.device,
                self._shard_count,
                self.memory_kind,
                self.validated.physical_bytes,
                self._allocation_count,
                len(self._mappings),
                self._graph_pins,
                (time.monotonic() - started) * 1000,
                type(error).__name__[:64] if error is not None else "none",
                cleanup_pending,
            )
        except Exception:
            pass

    def require_model_layout(self, layer_ids, num_embeddings, head_dim):
        """Check the complete model against the contract validated at attach."""
        expected = self.validated.layout
        layers, rows = tuple(layer_ids), tuple(num_embeddings)
        if (
            len(layers) != len(rows)
            or tuple(sorted(zip(layers, rows))) != expected.layers
            or head_dim != expected.head_dim
            or expected.backend not in ("fabric_egm", "fabric_gpu")
        ):
            raise ValueError(
                "external generation does not match the complete model Engram layout"
            )

    def verify(self, verify_views):
        """Verify GPU reads before READY; callback must drain and return True.

        The authenticated control channel binds capabilities to allocations.
        Metadata hashes alone cannot prove the physical bytes. The caller's
        verifier compares device-side samples to trusted expected data; full
        content hashing belongs to the owner loading path.
        """
        from types import MappingProxyType

        with self._lock:
            if self._state != "MAPPED":
                raise RuntimeError("verification requires freshly MAPPED storage")
            self._state = "VERIFYING"
            started = time.monotonic()
            try:
                if (
                    verify_views(MappingProxyType(self._views), self.validated)
                    is not True
                ):
                    raise ValueError("GPU content verifier must explicitly return True")
                if self._state != "VERIFYING":
                    raise RuntimeError("verification state was changed unexpectedly")
            except BaseException as error:
                self._state = "FAILED"
                self._log_state("storage_verification_failed", "verify", started, error)
                raise
            self._state = "READY"
            self._log_state("storage_verified", "verify", started)

    def _require_ready(self):
        if self._state != "READY":
            raise RuntimeError("Engram storage is not READY")

    @contextmanager
    def use(self):
        """Serialize enqueue vs drain. Hold until GPU work has been submitted."""
        with self._lock:
            self._require_ready()
            yield

    def table_views(self, layer_id: int):
        with self.use():
            prefix = f"layers.{layer_id}.engram.embed."
            try:
                return self._views[prefix + "weight"], self._views[prefix + "scale"]
            except KeyError:
                raise KeyError(
                    "requested layer is absent from this generation"
                ) from None

    def register_lookup_event(self, event):
        """Retain a side-stream completion fence until imports are drained."""
        with self.use():
            self._lookup_events.append(event)

    def _pin_graph(self):
        with self.use():
            self._graph_pins += 1

    def _unpin_graph(self):
        with self._lock:
            if self._graph_pins <= 0:
                raise RuntimeError("unbalanced graph lifetime")
            self._graph_pins -= 1

    def info(self):
        with self._lock:
            return dict(
                state=self._state,
                generation=self.validated.manifest.generation,
                dataset_id=self.validated.manifest.dataset_id,
                device=self.device,
                shard_count=self._shard_count,
                physical_bytes=self.validated.physical_bytes,
                allocation_count=self._allocation_count,
                mapping_count=len(self._mappings),
                graph_pins=self._graph_pins,
                allocations={
                    aid: mapping.info() for aid, mapping in self._mappings.items()
                },
            )

    def close(self):
        """Stop enqueue, drop owned views, then drain and release imports.

        Live graph pins reject close without changing READY. Other externally
        held views leave the object CLOSING: release them and retry close.
        """
        with self._lock:
            if self._state == "CLOSED":
                return
            if self._state == "VERIFYING":
                raise StorageBusyError("content verification is still in progress")
            if self._graph_pins:
                raise StorageBusyError("captured graphs still hold this storage")
            self._state = "CLOSING"
            for event in self._lookup_events:
                event.synchronize()
            self._lookup_events.clear()
            self._views.clear()
            if any(m.info()["active_views"] for m in self._mappings.values()):
                raise StorageBusyError(
                    "external tensor views must be released before close"
                )
            try:
                for mapping in reversed(list(self._mappings.values())):
                    mapping.close()
            except Exception:
                self._state = "FAILED"
                raise
            self._mappings.clear()
            self._state = "CLOSED"
