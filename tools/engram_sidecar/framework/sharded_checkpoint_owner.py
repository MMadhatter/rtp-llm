#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish one N-way Engram shard from the real checkpoint and hold it.

The owner loads only its row range of each layer's weight/scale tensor into
separate fabric allocations (a fabric handle can only be mapped at offset 0),
writes a private shard descriptor, and keeps the allocations alive until the
coordinator creates ``STOP``. The coordinator merges the descriptors from all
owners into one manifest/capability bundle.

    python3 sharded_checkpoint_owner.py --checkpoint /path/ckpt \
        --model-revision rev --shard 0 --shards 2 \
        --publication /run/shard0 --device 0 --numa 0

Never log or copy the private descriptor outside the authenticated transfer.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

LOAD_CHUNK = 64 << 20
READ_ROWS = 1 << 20


def emit(status, **fields):
    try:
        print(json.dumps(dict(status=status, **fields), sort_keys=True), flush=True)
    except (OSError, ValueError):
        pass


def private_write(path, payload):
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_rows(owner, reader, name, start, end, columns):
    """Stream a row range into the owner; padding stays zero."""
    import torch

    view = reader.get_slice(name)
    digest = hashlib.sha256()
    offset = 0
    for row in range(start, end, READ_ROWS):
        stop = min(row + READ_ROWS, end)
        block = view[row:stop].contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(block)
        for index in range(0, len(block), LOAD_CHUNK):
            piece = block[index:index + LOAD_CHUNK]
            owner.load(piece, offset)
            offset += len(piece)
    expected = (end - start) * columns
    if offset != expected:
        raise RuntimeError("shard row range has an unexpected byte count")
    return "sha256:" + digest.hexdigest()


def tensor_shard_layout(tensor, weight_shape, shard, shards, granularity):
    """Use the weight row partition for both tensors; only byte widths differ."""
    from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards

    rows, dim = weight_shape
    plan = plan_shards(rows, dim, shards, granularity=granularity)
    role = tensor.name.rsplit(".", 1)[-1]
    if role not in ("weight", "scale"):
        raise ValueError("unsupported Engram tensor role")
    width = dim if role == "weight" else dim // plan.block
    if tuple(tensor.shape) != (rows, width):
        raise ValueError("weight and scale shapes must share the same row layout")
    start, end = plan.row_range(shard)
    allocation_bytes = plan.weight_bytes if role == "weight" else plan.scale_bytes
    return start, end, allocation_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numa", type=int, default=0)
    parser.add_argument("--max-allocation-bytes", type=int, default=128 << 30)
    parser.add_argument("--budget-bytes", type=int, default=256 << 30)
    parser.add_argument("--report-timeout", type=float, default=3600)
    args = parser.parse_args(argv)

    if (not args.checkpoint.is_absolute() or not args.publication.is_absolute()
            or args.device < 0 or args.numa < 0 or args.shards < 1
            or not 0 <= args.shard < args.shards):
        parser.error("absolute paths, valid shard index and nonnegative ordinals required")
    args.publication.mkdir(mode=0o700, parents=True, exist_ok=False)

    import torch
    from safetensors import safe_open
    from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _load_native, hash_state_digest
    from rtp_llm.models_py.modules.dsv41.engram_checkpoint import inspect_checkpoint
    from rtp_llm.models_py.modules.dsv41.engram_manifest import ExpectedLayout
    checkpoint = args.checkpoint
    config = json.loads((checkpoint / "config.json").read_text())
    config = {**config, **config.get("text_config", {})}
    layout = EngramLayout(config)
    actual_hash_abi = hash_state_digest(NgramHashState(layout, str(checkpoint)))
    expected = ExpectedLayout(
        layers=tuple(zip(layout.layer_ids, layout.num_embeddings)),
        head_dim=layout.head_dim, model_revision=args.model_revision,
        hash_abi=actual_hash_abi)

    torch.cuda.set_device(args.device)
    native = _load_native()
    granularity = native.host_numa_granularity(args.numa, args.device)
    plan = inspect_checkpoint(
        checkpoint, expected, allocation_granularity=granularity,
        budget_bytes=args.budget_bytes, max_allocation_bytes=args.max_allocation_bytes)

    owners = []
    tensors = {}
    try:
        for layer_plan in plan.layers:
            weight_shape = next(t.shape for t in layer_plan.tensors if t.name.endswith(".weight"))
            for tensor in layer_plan.tensors:
                rows, columns = tensor.shape
                start, end, local_bytes = tensor_shard_layout(
                    tensor, weight_shape, args.shard, args.shards, granularity)
                owner = native.Owner(local_bytes, args.numa, args.device)
                owners.append(owner)
                with safe_open(checkpoint / tensor.relative_path, framework="pt", device="cpu") as reader:
                    digest = load_rows(owner, reader, tensor.name, start, end, columns)
                tensors[tensor.name] = dict(
                    allocation_id=str(uuid4()),
                    handle=base64.b64encode(owner.export_handle()).decode("ascii"),
                    row_begin=start, row_end=end,
                    logical_bytes=(end - start) * columns,
                    allocation_bytes=local_bytes,
                    allocation_granularity=granularity,
                    sha256=digest)
        document = dict(schema_version=1, shard=args.shard, shards=args.shards,
                        generation=args.generation, model_revision=args.model_revision,
                        hash_abi=actual_hash_abi, owner_id=f"shard-owner-{args.shard}",
                        owner_incarnation=str(uuid4()), tensors=tensors)
        private_write(args.publication / "shard.json",
                      json.dumps(document, sort_keys=True).encode("utf-8"))
        emit("READY", shard=args.shard, shards=args.shards, generation=args.generation,
             tensors=len(tensors), device=args.device, numa=args.numa,
             counters=native.resource_counters())
        deadline = time.monotonic() + args.report_timeout
        while not (args.publication / "STOP").is_file():
            if time.monotonic() > deadline:
                emit("WAITING_FOR_STOP", action="backing_retained")
                deadline = float("inf")
            time.sleep(1)
    finally:
        for owner in reversed(owners):
            try:
                owner.close()
            except BaseException:
                pass
        emit("CLOSED", counters=native.resource_counters())
        (args.publication / "shard.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
