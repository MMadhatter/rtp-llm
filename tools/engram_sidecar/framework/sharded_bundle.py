#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge N shard descriptors into one Engram manifest/capability bundle.

Each owner publishes a private ``shard.json`` from ``sharded_checkpoint_owner``.
This coordinator validates that the descriptors agree on generation, shard
count, tensor set and row coverage, then writes the bundle the model consumes.
It runs on a host with the checkpoint (samples are hashed from checkpoint
bytes, never from GPU data).

    python3 sharded_bundle.py --checkpoint /path/ckpt --model-revision rev \
        --generation <uuid> --shards 2 --publication /run/bundle \
        --shard-json /run/shard0/shard.json --shard-json /run/shard1/shard.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def digest(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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


def validate_documents(documents, shards, generation, model_revision=None, hash_abi=None):
    from rtp_llm.models_py.modules.dsv41.engram_shard_config import validate_count
    from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards
    validate_count(shards)
    if len(documents) != shards:
        raise ValueError("one descriptor per storage host required")
    if any(type(d.get("shard")) is not int or type(d.get("shards")) is not int
           or type(d.get("schema_version")) is not int or d.get("schema_version") != 2 for d in documents):
        raise ValueError("version 2 descriptors with integer shard indices required")
    if {d["shard"] for d in documents} != set(range(shards)):
        raise ValueError("descriptor shard indices are incomplete or duplicated")
    documents = sorted(documents, key=lambda d: d["shard"])
    model_revision = documents[0]["model_revision"] if model_revision is None else model_revision
    hash_abi = documents[0]["hash_abi"] if hash_abi is None else hash_abi
    if any(d["shards"] != shards or d["generation"] != generation
           or d["model_revision"] != model_revision or d["hash_abi"] != hash_abi for d in documents):
        raise ValueError("descriptor generation/count/revision/hash mismatch")
    hosts = [d.get("host_id") for d in documents]
    if any(type(host) is not str or not host.strip() for host in hosts) or len(set(hosts)) != shards:
        raise ValueError("storage host IDs must be nonempty and distinct")
    if any(d["owner_id"] != d["host_id"] for d in documents):
        raise ValueError("owner identity must be the storage host ID")
    names = set(documents[0]["tensors"])
    if not names or any(set(d["tensors"]) != names for d in documents):
        raise ValueError("descriptor tensor sets differ")
    allocations, handles = set(), set()
    for name in names:
        role = name.rsplit(".", 1)[-1]
        if role not in ("weight", "scale"):
            raise ValueError("unsupported tensor role")
        weight_name = name.rsplit(".", 1)[0] + ".weight"
        scale_name = name.rsplit(".", 1)[0] + ".scale"
        if not {weight_name, scale_name} <= names:
            raise ValueError("missing weight/scale pair")
        first = documents[0]["tensors"][weight_name]
        shape = first["shape"]
        if type(shape) is not list or len(shape) != 2:
            raise ValueError("invalid weight shape")
        plan = plan_shards(*shape, shards, granularity=first["allocation_granularity"])
        columns = plan.dim if role == "weight" else plan.dim // plan.block
        size = plan.weight_bytes if role == "weight" else plan.scale_bytes
        for index, document in enumerate(documents):
            entry = document["tensors"][name]
            for key in ("row_begin", "row_end", "logical_bytes", "allocation_bytes", "allocation_granularity"):
                if type(entry[key]) is not int:
                    raise ValueError("descriptor geometry must use integers")
            if (type(entry["shape"]) is not list or len(entry["shape"]) != 2
                    or any(type(n) is not int for n in entry["shape"])):
                raise ValueError("descriptor shape must use integers")
            start, end = plan.row_range(index)
            if (entry["shape"] != [plan.total_rows, columns]
                    or (entry["row_begin"], entry["row_end"]) != (start, end)
                    or entry["allocation_bytes"] != size
                    or entry["allocation_granularity"] != plan.granularity
                    or entry["logical_bytes"] != (end - start) * columns):
                raise ValueError("descriptor geometry does not match storage host index")
            aid = entry["allocation_id"]
            if aid in allocations:
                raise ValueError("duplicate allocation ID")
            allocations.add(aid)
            try:
                raw = base64.b64decode(entry["handle"], validate=True)
            except (ValueError, TypeError):
                raise ValueError("invalid capability encoding") from None
            if not 0 < len(raw) <= 768 or raw in handles:
                raise ValueError("empty, oversized or duplicate capability")
            handles.add(raw)
    return documents


def read_shards(paths, shards, generation):
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _private_read, _json
    documents = [_json(_private_read(Path(path), 1 << 20)) for path in paths]
    return validate_documents(documents, shards, generation)


def build_manifest(documents, shards, generation, model_revision, hash_abi):
    documents = validate_documents(documents, shards, generation, model_revision, hash_abi)
    names = sorted(documents[0]["tensors"])
    tensors = []
    for name in names:
        role = name.rsplit(".", 1)[1]
        columns = documents[0]["tensors"][name]["shape"][1]
        entries = []
        for document in documents:
            entry = document["tensors"][name]
            entries.append(dict(
                shard_id=str(uuid4()),
                row_begin=int(entry["row_begin"]), row_end=int(entry["row_end"]),
                logical_bytes=int(entry["logical_bytes"]), byte_offset=0,
                allocation_bytes=int(entry["allocation_bytes"]),
                allocation_granularity=int(entry["allocation_granularity"]),
                owner_id=document["owner_id"],
                owner_incarnation=document["owner_incarnation"],
                allocation_id=entry["allocation_id"],
                handle_ref="capability-" + entry["allocation_id"],
                sha256=entry["sha256"]))
        entries.sort(key=lambda item: item["row_begin"])
        rows = entries[-1]["row_end"]
        tensors.append(dict(
            name=name, shape=[rows, columns],
            dtype="float8_e4m3fn" if role == "weight" else "float8_e8m0fnu",
            storage_dtype="uint8", row_bytes=columns, quant_block=32,
            shards=entries))
    document = dict(schema_version=1, model_revision=model_revision,
                    generation=generation, layout_abi="engram-fp8-e8m0-v1",
                    hash_abi=hash_abi, backend="fabric_egm", state="READY",
                    tensors=tensors)
    from rtp_llm.models_py.modules.dsv41.engram_manifest import compute_dataset_id
    document["dataset_id"] = compute_dataset_id(document)
    from rtp_llm.models_py.modules.dsv41.engram_manifest import parse_manifest
    parse_manifest(json.dumps(document).encode())
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--shard-json", action="append", required=True)
    args = parser.parse_args(argv)

    if (not args.checkpoint.is_absolute() or not args.publication.is_absolute()
            or not 2 <= args.shards <= 64):
        parser.error("absolute paths and a positive shard count required")
    documents = read_shards(args.shard_json, args.shards, args.generation)
    hash_abi = documents[0]["hash_abi"]
    if any(d["hash_abi"] != hash_abi for d in documents):
        raise SystemExit("descriptor hash_abi mismatch")
    document = build_manifest(documents, args.shards, args.generation,
                              args.model_revision, hash_abi)
    manifest_bytes = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

    from rtp_llm.models_py.modules.dsv41.engram_manifest import parse_manifest
    from tools.engram_sidecar.framework.checkpoint_support import selected_samples
    manifest = parse_manifest(manifest_bytes)
    samples = selected_samples(args.checkpoint, manifest)
    bundle = dict(schema_version=1, manifest="manifest.json",
                  capabilities="capabilities.json",
                  manifest_sha256=digest(manifest_bytes),
                  dataset_id=document["dataset_id"], generation=document["generation"],
                  model_revision=document["model_revision"], hash_abi=hash_abi,
                  samples=samples)
    capabilities = {entry["allocation_id"]: entry["handle"]
                    for d in documents for entry in d["tensors"].values()}
    args.publication.mkdir(mode=0o700, parents=True, exist_ok=False)
    private_write(args.publication / "manifest.json", manifest_bytes)
    private_write(args.publication / "capabilities.json",
                  json.dumps(capabilities).encode())
    private_write(args.publication / "bundle.json",
                  json.dumps(bundle, sort_keys=True).encode())
    print(json.dumps(dict(status="READY", dataset_id=document["dataset_id"],
                          generation=document["generation"], shards=args.shards,
                          allocations=len(capabilities)), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
