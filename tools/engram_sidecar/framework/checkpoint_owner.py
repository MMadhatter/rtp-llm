#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hold an immutable EGM checkpoint for cross-host model-hook development.

This is a test coordinator endpoint, not a production service. The launcher must
keep this process alive and create publication/STOP only after every consumer
process has exited. No timeout or lost output pipe releases published backing.
Private capabilities and their values must never be copied into public evidence.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
from uuid import uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


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


def digest(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def failure_fields(error):
    """Expose bounded native CUDA diagnostics without arbitrary exception text.

    Paths, payloads and capabilities must not reach publication logs through an
    exception's message. The native bridge's fixed stage/name/numeric code are
    enough to distinguish allocation pressure from import/access failures.
    """
    fields = dict(error_type=type(error).__name__, code=getattr(error, "code", None))
    if isinstance(error, RuntimeError):
        match = re.match(r"\A([a-z_]{1,64}): (CUDA_[A-Z_]{1,80}) \(code=([0-9]{1,6})\):", str(error))
        if match:
            fields.update(cuda_stage=match[1], cuda_error=match[2], cuda_code=int(match[3]))
    return fields


@dataclass(frozen=True)
class CheckpointSource:
    checkpoint: Path
    config: dict
    model_revision: str
    expected_hash_abi: str | None
    config_sha256: str
    index_sha256: str
    component_fixture: bool


def read_source(*, fixture_root=None, checkpoint=None, model_revision=None):
    """Resolve a trusted local input without importing Torch/native code.

    Revision is an operator-supplied checkpoint identity, not a claim that the
    weights match a public release. The manifest later binds table byte hashes;
    config/index hashes are emitted separately for experiment provenance.
    """
    if (fixture_root is None) == (checkpoint is None):
        raise ValueError("select exactly one of fixture_root and checkpoint")
    expected_hash = None
    if fixture_root is not None:
        fixture_root = Path(fixture_root)
        if not fixture_root.is_absolute() or model_revision is not None:
            raise ValueError("fixture root must be absolute; fixture owns its revision")
        fixture = json.loads((fixture_root / "fixture.json").read_text())
        checkpoint = fixture_root / "owner_checkpoint"
        model_revision, expected_hash = fixture["model_revision"], fixture["hash_abi"]
    checkpoint = Path(checkpoint)
    if (not checkpoint.is_absolute() or not isinstance(model_revision, str)
            or not model_revision.strip() or len(model_revision) > 512
            or any(ord(char) < 32 for char in model_revision)):
        raise ValueError("absolute checkpoint and explicit nonempty model revision required")
    config_bytes = (checkpoint / "config.json").read_bytes()
    raw = json.loads(config_bytes)
    if not isinstance(raw, dict) or not isinstance(raw.get("text_config", {}), dict):
        raise ValueError("invalid checkpoint config")
    config = {**raw, **raw.get("text_config", {})}
    config.pop("text_config", None)
    config["checkpoint_path"] = str(checkpoint)
    if fixture_root is None and raw.get("fixture_component_only"):
        raise ValueError("component fixture requires --fixture-root")
    return CheckpointSource(checkpoint, config, model_revision, expected_hash,
                            digest(config_bytes),
                            digest((checkpoint / "model.safetensors.index.json").read_bytes()),
                            fixture_root is not None)


def parser_for_owner():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture-root", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-revision", help="Required with --checkpoint; immutable operator identity")
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numa", type=int, default=0)
    parser.add_argument("--max-allocation-bytes", type=int, default=1 << 30)
    parser.add_argument("--budget-bytes", type=int, default=64 << 20)
    parser.add_argument("--discard-private-checkpoint-cache", action="store_true",
                        help="Advise release of source file cache during loading; only for an operator-owned private immutable checkpoint copy")
    parser.add_argument("--report-timeout", type=float, default=1800)
    parser.add_argument("--rpc-listen")
    parser.add_argument("--service-id")
    parser.add_argument("--tls-ca", type=Path)
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument("--allowed-client", action="append", default=[])
    return parser


def selected_samples(checkpoint, manifest):
    """Bind small expected reads to checkpoint bytes, not to imported GPU data."""
    import torch
    from safetensors import safe_open

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    samples = {}
    for tensor in manifest.tensors:
        rows = sorted({0, tensor.shape[0] // 2, tensor.shape[0] - 1})
        relative = Path(index["weight_map"][tensor.name])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe checkpoint relative path")
        with safe_open(checkpoint / relative, framework="pt", device="cpu") as source:
            sliced = source.get_slice(tensor.name)
            # Slice the mapped file a row at a time; never duplicate a full table.
            payload = b"".join(sliced[row:row + 1].contiguous().view(torch.uint8).numpy().tobytes()
                               for row in rows)
        samples[tensor.name] = {"row_ids": rows, "sha256": digest(payload)}
    return samples


def main(argv=None):
    parser = parser_for_owner()
    args = parser.parse_args(argv)
    if (not args.publication.is_absolute()
            or args.device < 0 or args.numa < 0
            or not 0 < args.report_timeout < float("inf")):
        parser.error("absolute paths, nonnegative ordinals and finite positive timeout required")
    from rtp_llm.models_py.modules.dsv41.engram_capacity import validate_capacity
    try:
        validate_capacity(args.max_allocation_bytes, args.budget_bytes)
        source = read_source(fixture_root=args.fixture_root, checkpoint=args.checkpoint,
                             model_revision=args.model_revision)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    args.publication.mkdir(mode=0o700, parents=True, exist_ok=False)
    control = None
    if args.rpc_listen:
        from rtp_llm.models_py.modules.dsv41.engram_rpc import OwnerControl
        if not all((args.service_id, args.tls_ca, args.tls_cert, args.tls_key, args.allowed_client)):
            parser.error("RPC requires service-id, TLS CA/cert/key and allowed-client")
        control = OwnerControl(args.service_id, args.allowed_client)
        control.start(args.rpc_listen, ca=args.tls_ca, cert=args.tls_cert, key=args.tls_key)
    checkpoint = source.checkpoint
    published, failed, generation, native = False, False, None, None

    def hold_until_stop():
        deadline = time.monotonic() + args.report_timeout
        reported = False
        while not (args.publication / "STOP").is_file():
            if not reported and time.monotonic() >= deadline:
                emit("WAITING_FOR_STOP", action="backing_retained")
                reported = True
            threading.Event().wait(1)

    def interrupt(signum, frame):
        nonlocal failed
        if not published:
            raise SystemExit(128 + signum)
        failed = True
        emit("SIGNAL_WHILE_PUBLISHED", signal=signum, action="backing_retained_until_STOP")

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        import torch
        from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState
        from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _load_native, hash_state_digest
        from rtp_llm.models_py.modules.dsv41.engram_checkpoint import inspect_checkpoint, load_checkpoint
        from rtp_llm.models_py.modules.dsv41.engram_manifest import ExpectedLayout

        torch.cuda.set_device(args.device)
        native = _load_native()
        layout = EngramLayout(source.config)
        actual_hash_abi = hash_state_digest(NgramHashState(layout, str(checkpoint)))
        if source.expected_hash_abi is not None and actual_hash_abi != source.expected_hash_abi:
            raise ValueError("fixture tokenizer changed")
        expected = ExpectedLayout(layers=tuple(zip(layout.layer_ids, layout.num_embeddings)),
                                  head_dim=layout.head_dim, model_revision=source.model_revision,
                                  hash_abi=actual_hash_abi)
        plan = inspect_checkpoint(checkpoint, expected,
                                  allocation_granularity=native.host_numa_granularity(args.numa, args.device),
                                  budget_bytes=args.budget_bytes,
                                  max_allocation_bytes=args.max_allocation_bytes)
        if (source.index_sha256 != "sha256:" + plan.index.header_sha256
                or source.config_sha256 != digest((checkpoint / "config.json").read_bytes())):
            raise ValueError("checkpoint metadata changed during preflight")
        emit("INSPECTED", model_revision=source.model_revision, hash_abi=actual_hash_abi,
             config_sha256=source.config_sha256, index_sha256=source.index_sha256,
             component_fixture=source.component_fixture, physical_bytes=plan.physical_bytes,
             budget_bytes=args.budget_bytes, max_allocation_bytes=args.max_allocation_bytes)
        generation = load_checkpoint(plan, native=native, device=args.device, numa=args.numa,
                                     discard_file_cache=args.discard_private_checkpoint_cache)
        manifest = generation.manifest
        bundle = dict(schema_version=1, manifest="manifest.json", capabilities="capabilities.json",
                      manifest_sha256=digest(generation.manifest_bytes),
                      dataset_id=manifest.dataset_id, generation=manifest.generation,
                      model_revision=manifest.model_revision, hash_abi=actual_hash_abi,
                      samples=selected_samples(checkpoint, manifest))
        capabilities = {key: base64.b64encode(value).decode("ascii")
                        for key, value in generation.private_handles.items()}
        # Once any export may be visible, cleanup requires confirmed consumer exit.
        published = True
        if control is not None:
            control.publish(bundle, generation.manifest_bytes, capabilities)
        else:
            private_write(args.publication / "capabilities.json", json.dumps(capabilities).encode())
        private_write(args.publication / "manifest.json", generation.manifest_bytes)
        if control is None:
            private_write(args.publication / "bundle.json", json.dumps(bundle, sort_keys=True).encode())
        del capabilities
        emit("READY", dataset_id=manifest.dataset_id, generation=manifest.generation,
             physical_bytes=generation.physical_bytes, allocation_count=len(generation.private_handles),
             device=args.device, numa=args.numa, counters=native.resource_counters())
        hold_until_stop()
    except BaseException as error:
        failed = True
        emit("FAILED", **failure_fields(error))
        if published:
            hold_until_stop()
    finally:
        if published and not (args.publication / "STOP").is_file():
            hold_until_stop()
        if control is not None:
            control.stop()
        if generation is not None:
            try:
                generation.close()
            except BaseException as error:
                failed = True
                emit("CLEANUP_FAILED", error_type=type(error).__name__)
        if native is not None:
            counters = native.resource_counters()
            failed = failed or any(counters.values())
            emit("CLOSED" if not any(counters.values()) else "RESOURCES_REMAIN", counters=counters)
        for name in ("bundle.json", "capabilities.json"):
            (args.publication / name).unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
