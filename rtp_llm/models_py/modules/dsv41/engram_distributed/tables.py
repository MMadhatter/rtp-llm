# SPDX-License-Identifier: Apache-2.0
"""Model-facing table sources; checkpoint projections never depend on backing.

A source returns the existing lookup adapter, preserving its lease, NUMA and
prefetch optimizations. Fabric sources borrow process-owned storage; local
sources retain the existing shared-file/reader-lease lifetime. Neither source
introduces an extra copy or changes the lookup kernel contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class TableSource(Protocol):
    def validate_layout(self, layout, target): ...

    def load_embedding(
        self, layer_id, shape, target, load, *, weight_map, scale_group=None
    ): ...


@dataclass(frozen=True)
class LocalHostTables:
    checkpoint: Path

    def validate_layout(self, layout, target):
        pass

    def load_embedding(
        self, layer_id, shape, target, load, *, weight_map, scale_group=None
    ):
        from ..engram import (
            HostEngramEmbedding,
            _PinnedSharedTable,
            _prefault_host_tensor,
        )

        prefix = f"layers.{layer_id}.engram."
        table, scales = load("embed.weight"), load("embed.scale")
        if tuple(table.shape) != shape:
            raise ValueError(f"Engram table shape {table.shape} does not match {shape}")
        pinned = None
        identity = None
        if target.type == "cuda":
            parts = [str(self.checkpoint.resolve()), prefix]
            for suffix in ("embed.weight", "embed.scale"):
                path = self.checkpoint / weight_map[prefix + suffix]
                stat = path.stat()
                parts.extend(
                    [str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns)]
                )
            identity = "|".join(parts)
            pinned = _PinnedSharedTable(table, scales, identity, target)
            table, scales = pinned.weight, pinned.scales
        else:
            _prefault_host_tensor(table, prefix + "embed.weight")
            _prefault_host_tensor(scales, prefix + "embed.scale")
        return HostEngramEmbedding(
            table,
            scales,
            pinned=pinned,
            scale_group=scale_group,
            scale_identity=identity,
        )


@dataclass(frozen=True)
class FabricTables:
    storage: object

    def validate_layout(self, layout, target):
        import torch

        if target.type != "cuda":
            raise ValueError("external Engram storage requires a CUDA model")
        ordinal = (
            target.index if target.index is not None else torch.cuda.current_device()
        )
        if ordinal != self.storage.device:
            raise ValueError(
                "model and external Engram storage must use the same CUDA device"
            )
        with self.storage.use():
            self.storage.require_model_layout(
                layout.layer_ids, layout.num_embeddings, layout.head_dim
            )

    def load_embedding(
        self, layer_id, shape, target, load, *, weight_map, scale_group=None
    ):
        from ..engram import HostEngramEmbedding

        manifest = self.storage.validated.manifest
        return HostEngramEmbedding.from_storage(
            self.storage,
            layer_id,
            scale_group=scale_group,
            scale_identity=f"{manifest.dataset_id}|{manifest.generation}",
        )


def table_source(checkpoint, storage=None) -> TableSource:
    return FabricTables(storage) if storage is not None else LocalHostTables(checkpoint)
