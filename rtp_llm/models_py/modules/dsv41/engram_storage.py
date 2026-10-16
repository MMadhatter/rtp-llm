# SPDX-License-Identifier: Apache-2.0
"""Explicit EGM attach and graph lifetimes; no RPC or checkpoint loading.

Only operations submitted under ``use()`` participate in the drain contract.
Raw tensor views must not be captured in an unmanaged CUDA graph. This first
slice uses an injected native backend; it never compiles an extension in forward.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Mapping

from .engram_manifest import ExpectedLayout, Limits, parse_manifest, validate_for_attach


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
        self._mappings = {}
        self._views = {}
        self._graph_pins = 0
        self._state = "ATTACHING"

    @classmethod
    def attach(cls, manifest_bytes: bytes, *, expected_layout: ExpectedLayout,
               dataset_id: str, generation: str, handles: Mapping[str, bytes],
               device: int, native, limits: Limits = Limits()):
        if type(device) is not int or device < 0:
            raise ValueError("device must be a local CUDA ordinal")
        verified = validate_for_attach(parse_manifest(manifest_bytes, limits),
                                       expected_layout, dataset_id, generation, limits)
        if verified.manifest.backend != "fabric_egm":
            raise ValueError("FabricEngramStorage requires fabric_egm")
        slices = {t.shards[0].allocation_id: t.shards[0]
                  for t in verified.manifest.tensors}
        if not isinstance(handles, Mapping) or set(handles) != set(slices):
            raise ValueError("handle allocation set does not match the manifest")
        if any(not isinstance(h, bytes) for h in handles.values()):
            raise ValueError("Fabric capabilities must be opaque bytes")
        result = cls()
        result.validated = verified
        result.device = device
        try:
            for aid, part in sorted(slices.items()):
                result._mappings[aid] = native.Consumer(
                    handles[aid], part.allocation_bytes,
                    part.allocation_granularity, device)
            for tensor in verified.manifest.tensors:
                part = tensor.shards[0]
                result._views[tensor.name] = result._mappings[part.allocation_id].tensor_view(
                    list(tensor.shape), part.byte_offset)
            result._state = "MAPPED"
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
            if failures:
                raise CleanupPendingError("attach failed and native cleanup was incomplete", result) from original
            raise

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
            try:
                if verify_views(MappingProxyType(self._views), self.validated) is not True:
                    raise ValueError("GPU content verifier must explicitly return True")
            except BaseException:
                self._state = "FAILED"
                raise
            if self._state != "VERIFYING":
                raise RuntimeError("verification state was changed unexpectedly")
            self._state = "READY"

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
                raise KeyError("requested layer is absent from this generation") from None

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
            return dict(state=self._state, generation=self.validated.manifest.generation,
                        dataset_id=self.validated.manifest.dataset_id,
                        physical_bytes=self.validated.physical_bytes,
                        allocation_count=len(self._mappings), graph_pins=self._graph_pins,
                        allocations={aid: mapping.info() for aid, mapping in self._mappings.items()})

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
            self._views.clear()
            if any(m.info()["active_views"] for m in self._mappings.values()):
                raise StorageBusyError("external tensor views must be released before close")
            try:
                for mapping in reversed(list(self._mappings.values())):
                    mapping.close()
            except Exception:
                self._state = "FAILED"
                raise
            self._mappings.clear()
            self._state = "CLOSED"


class CapturedEngramGraph:
    """A CUDA graph that pins every storage it can read until reset + drain.

    ``forward`` must use only the supplied storages. Keep graph ownership here:
    do not expose/reuse the private graph outside this wrapper. The wrapper has
    no unsafe GC-driven unmap; explicit close is required to release its pins.
    """

    def __init__(self):
        self._closed = False
        self._replay_allowed = False
        self._graph = None
        self._storages = ()
        self.output = None
        self._lock = threading.RLock()

    @classmethod
    def capture(cls, forward, storages, *, warmup=3):
        import torch
        if type(warmup) is not int or warmup < 1:
            raise ValueError("warmup must be positive")
        unique = tuple(dict.fromkeys(storages))
        if not unique or len({s.device for s in unique}) != 1:
            raise ValueError("capture requires storages on one CUDA device")
        result = cls()
        result.device = unique[0].device
        pinned = []
        try:
            for storage in unique:
                storage._pin_graph()
                pinned.append(storage)
            result._storages = tuple(pinned)
            with torch.cuda.device(result.device):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(warmup):
                        forward()
                stream.synchronize()
                result._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(result._graph, stream=stream):
                    result.output = forward()
            result._replay_allowed = True
            return result
        except BaseException as original:
            # Never release pins unless all work and partial capture are drained.
            result._storages = tuple(pinned)
            try:
                result.close()
            except BaseException:
                raise CleanupPendingError("capture failed and cleanup remains pending", result) from original
            raise

    def replay(self):
        from contextlib import ExitStack
        with self._lock:
            if not self._replay_allowed:
                raise RuntimeError("graph is not ready for replay")
            # Fixed order avoids deadlocks when a graph reads multiple storages.
            with ExitStack() as scope:
                for storage in sorted(self._storages, key=id):
                    scope.enter_context(storage.use())
                import torch
                with torch.cuda.device(self.device):
                    self._graph.replay()
            return self.output

    def close(self):
        import torch
        with self._lock:
            if self._closed:
                return
            self._replay_allowed = False
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                if self._graph is not None:
                    self._graph.reset()
                    self._graph = None
            self.output = None
            for storage in self._storages:
                storage._unpin_graph()
            self._storages = ()
            self._closed = True
