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
import hashlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rtp_llm.models_py.modules.dsv41.engram_descriptors import validate_documents, build_manifest


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


def read_shards(paths, shards, generation):
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _private_read, _json
    documents = [_json(_private_read(Path(path), 1 << 20)) for path in paths]
    return validate_documents(documents, shards, generation)


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
