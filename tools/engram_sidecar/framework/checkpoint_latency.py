#!/usr/bin/env python3
"""Paired, single-layer real-checkpoint lookup latency; never a decode-step SLO.

Both paths call HostEngramEmbedding on the same GPU/IDs. The local control is
main's complete _PinnedSharedTable, populated from independent source shards;
the remote table is attached through the verified private bundle. Imports are
lazy so argument, evidence, and reporting contracts can be tested without CUDA.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from uuid import uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.engram_sidecar.framework import checkpoint_lookup as lookup
from tools.engram_sidecar.latency import bench_consumer as bench

COMPARISONS = {"raw_remote_minus_raw_local": ("remote_raw", "local_raw")}
_FAILED_RESOURCES = []


def emit(status, **values):
    print(json.dumps(dict(status=status, scope="real_single_layer_lookup_only",
                          decode_step_slo="NOT_MEASURED", **values),
                     sort_keys=True, allow_nan=False), flush=True)


def integer_list(value, *, allowed=None):
    if not re.fullmatch(r"\d+(,\d+)*", value):
        raise ValueError("expected comma-separated nonnegative integers")
    result = [int(item) for item in value.split(",")]
    if len(result) != len(set(result)) or (allowed is not None and not set(result) <= set(allowed)):
        raise ValueError("duplicate or unsupported list item")
    return result


def choice_list(value, allowed):
    result = value.split(",")
    if not result or len(result) != len(set(result)) or not set(result) <= set(allowed):
        raise ValueError("duplicate or unsupported choice")
    return result


def check_options(args):
    args.batches = integer_list(args.batches, allowed=(1, 2, 4, 8, 16, 32, 64))
    args.layers = None if args.layers == "first" else integer_list(args.layers)
    args.modes = choice_list(args.modes, ("eager", "graph"))
    args.distributions = choice_list(args.distributions, ("hot", "random"))
    if not 1 <= args.samples <= 1_000_000 or args.warmup < 1 or args.device < 0 or args.cpu_slot < 0:
        raise ValueError("invalid samples, warmup, device or CPU slot")
    if args.seed < 0 or args.calibration_samples < 0 or not math.isfinite(args.threshold_us) or args.threshold_us <= 0:
        raise ValueError("invalid seed, calibration count or threshold")
    for name in ("checkpoint", "source_checkpoint", "bundle", "private_inputs", "output"):
        if not getattr(args, name).is_absolute():
            raise ValueError("all input/output paths must be absolute")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("use a new output directory; prior samples must not be overwritten")
    return args


def parse_numa_pages(text, address, node):
    matches = [line for line in text.splitlines() if line.split()
               and int(line.split()[0], 16) == address and "rtp_llm_engram_" in line]
    if len(matches) != 1:
        raise RuntimeError("cannot uniquely locate pinned table in numa_maps")
    pages = {int(n): int(count) for n, count in re.findall(r"\bN(\d+)=(\d+)\b", matches[0]) if int(count)}
    if not pages or set(pages) != {node}:
        raise RuntimeError("baseline pages are not exclusively on GPU-local HOST_NUMA")
    return {"node_pages": {str(n): count for n, count in pages.items()}, "expected_gpu_host_numa": node}


def bind_gpu_local_cpu(device, slot):
    from cuda.bindings import driver
    status, = driver.cuInit(0)
    if int(status):
        raise RuntimeError("cuInit failed")
    status, gpu = driver.cuDeviceGet(device)
    if int(status):
        raise RuntimeError("cuDeviceGet failed")
    status, node = driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, gpu)
    if int(status) or int(node) < 0:
        raise RuntimeError("GPU-local host NUMA identity unavailable")
    node = int(node)
    selected = set()
    for part in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(","):
        low, _, high = part.partition("-")
        selected.update(range(int(low), int(high or low) + 1))
    choices = sorted(selected & os.sched_getaffinity(0))
    if slot >= len(choices):
        raise RuntimeError("no authorized CPU at requested GPU-local NUMA slot")
    os.sched_setaffinity(0, {choices[slot]})
    return node, choices[slot]


def chunk_selected_ids(ids, maximum=lookup.MAX_SELECTED_ROWS):
    if type(maximum) is not int or maximum < 1:
        raise ValueError("positive chunk size required")
    if any(type(row) is not int or row < -1 for row in ids):
        raise ValueError("invalid lookup row")
    rows = sorted(set(ids) - {-1})
    return [rows[start:start + maximum] for start in range(0, len(rows), maximum)]


def source_rows(source, index, layout, layer, matrix):
    flat = [row for item in matrix for row in item]
    decoded = {-1: [0.0] * layout.head_dim}
    for chunk in chunk_selected_ids(flat):
        selected = {str(lid): set(chunk) if lid == layer else set() for lid in layout.layer_ids}
        records, _ = lookup.read_selected_rows(source, index, layout, selected)
        for record in records[str(layer)]:
            decoded[record["row_id"]] = lookup.decode_row(bytes.fromhex(record["fp8_hex"]),
                                                         bytes.fromhex(record["e8m0_hex"]))
    return [[decoded[row] for row in item] for item in matrix]


def source_sample_check(source, index, layout, parsed):
    selected = {layer: set(rows) for layer, rows in parsed.items()}
    records, headers = lookup.read_selected_rows(source, index, layout, selected)
    for layer, values in records.items():
        for row in values:
            actual = bytes.fromhex(row["fp8_hex"]), bytes.fromhex(row["e8m0_hex"])
            if actual != parsed[layer][row["row_id"]]:
                raise ValueError("independent baseline source differs from verified oracle")
    return headers


def make_local_table(source, index, layout, layer, device):
    """Use main's exact tmpfs/pinned baseline, with mmap source and bounded copy.

    safe_open maps source tensors; _PinnedSharedTable copies in 64MiB chunks.
    Source file pages can still be resident during loading. Budget tmpfs + file
    cache headroom; this is not a claim of constant-RSS checkpoint staging.
    """
    from safetensors import safe_open
    from rtp_llm.models_py.modules.dsv41.engram import _PinnedSharedTable
    from tools.engram_sidecar.framework.stage_checkpoint import _basename, _snapshot
    names = [f"layers.{layer}.engram.embed.{role}" for role in ("weight", "scale")]
    paths = {name: source / _basename(index[name]) for name in names}
    before = {str(path): _snapshot(path) for path in set(paths.values())}
    with ExitStack() as stack:
        files = {path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                 for path in set(paths.values())}
        q, s = (files[paths[name]].get_tensor(name) for name in names)
        rows = dict(zip(layout.layer_ids, layout.num_embeddings))[layer]
        if tuple(q.shape) != (rows, layout.head_dim) or tuple(s.shape) != (rows, layout.head_dim // 32):
            raise ValueError("baseline source geometry differs from model")
        identity = json.dumps([before, layer, uuid4().hex], sort_keys=True)
        control = _PinnedSharedTable(q, s, identity, device)
        del q, s
    after = {str(path): _snapshot(path) for path in set(paths.values())}
    if before != after:
        retain_failed(control)
        raise RuntimeError("source changed while constructing baseline")
    return control


def retain_failed(*resources):
    for resource in resources:
        for name in ("_registration_finalizer", "_lease_finalizer"):
            finalizer = getattr(resource, name, None)
            if finalizer is not None:
                finalizer.atexit = False
    _FAILED_RESOURCES.append(resources)


def close_local(control, torch):
    torch.cuda.synchronize(control.device)
    finalizer = control._registration_finalizer
    if not finalizer.alive:
        raise RuntimeError("baseline registration finalizer ran unexpectedly")
    with torch.cuda.device(control.device):
        status, = control.runtime.cudaHostUnregister(control.storage.data_ptr())
    if int(status):
        raise RuntimeError("baseline cudaHostUnregister failed")
    if finalizer.detach() is None:
        raise RuntimeError("cannot detach completed baseline unregister")
    control._lease_finalizer()


def write_new_json(path, document):
    bench._validate_metadata(document)
    # Exclusive create prevents silent replacement of a prior run or result.
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def make_records(*, mode, distribution, pair_id, path, order, begin_ns,
                 submit_ns, complete_ns, gpu_us, before_switches, after_switches):
    if mode not in ("eager", "graph") or not math.isfinite(gpu_us) or gpu_us <= 0:
        raise ValueError("valid mode and positive finite GPU interval required")
    parts = bench._wall_parts(begin_ns, submit_ns, complete_ns, before_switches, after_switches)
    # Keep neutral naming for eager: this is a Python dispatch call, not Graph replay.
    parts["cpu_submit_call_us"] = parts.pop("cpu_graph_replay_call_us")
    base = dict(distribution=distribution, epoch=0, pair_id=pair_id, path=path,
                order_index=order.index(path), path_order=order,
                wall_begin_perf_counter_ns=begin_ns, wall_end_perf_counter_ns=complete_ns,
                mode=mode, **parts)
    return [dict(base, clock=f"gpu_{mode}_event_interval", duration_us=gpu_us),
            dict(base, clock="cpu_submit_to_event_complete", duration_us=(complete_ns - begin_ns) / 1000.0)]


def benchmark_case(*, paths, storage, update, verify, args, metadata, output):
    """One mode/case. Inputs, verification and calibration are outside timing.

    Both Graph paths use the SAME guarded replay wrapper/pinning overhead;
    local's guard is conservative (it does not actually read remote storage).
    The benchmark does not grant engine-owned CUDA Graph support.
    """
    import torch
    from rtp_llm.models_py.modules.dsv41.engram_storage import CapturedEngramGraph
    mode = metadata["mode"]
    graphs, events, outputs, records = {}, {}, {}, []
    stream = torch.cuda.Stream(device=args.device)
    stream.wait_stream(torch.cuda.current_stream(args.device))
    reader = bench._thread_switches_reader()
    l2 = bench._l2_bytes(torch, args.device)
    if args.l2_disturb and (l2 is None or 2 * l2 > 1 << 30):
        raise RuntimeError("runtime L2 size unavailable or scratch exceeds 1GiB")
    checks = []
    raw = output / "samples.jsonl"
    try:
        with raw.open("x") as logfile, torch.inference_mode(), torch.cuda.stream(stream):
            os.chmod(raw, 0o600)
            scratch = torch.zeros(2 * l2, dtype=torch.uint8, device=args.device) if args.l2_disturb else None
            for name in paths:
                events[name] = (torch.cuda.Event(enable_timing=True, external=mode == "graph"),
                                torch.cuda.Event(enable_timing=True, external=mode == "graph"))
            for sample in range(args.warmup):
                update(sample, "random")
                for name in bench.balanced_order(paths, sample, args.seed):
                    outputs[name] = paths[name]()
            stream.synchronize()
            if mode == "graph":
                for name, path in paths.items():
                    def captured(path=path, name=name):
                        # CapturedEngramGraph uses its own capture stream.
                        events[name][0].record()
                        value = path()
                        events[name][1].record()
                        return value
                    graphs[name] = CapturedEngramGraph.capture(captured, [storage], warmup=1)
                del captured

            def invoke(name):
                if mode == "graph":
                    return graphs[name].replay()
                begin, end = events[name]
                begin.record(stream)
                value = paths[name]()
                end.record(stream)
                return value

            # Verify actual replay results on two different input buffers; no
            # eager overwrite may precede these checks.
            for index in (0, 1):
                update(index, "random")
                stream.synchronize()
                for name in paths:
                    outputs[name] = invoke(name)
                    events[name][1].synchronize()
                verify(index, "random", outputs)
                checks.append(dict(phase="before_measurement", trace_index=index, mode=mode))
            for distribution in args.distributions:
                for pair_id in range(args.samples):
                    update(pair_id, distribution)
                    order = bench.balanced_order(paths, pair_id, args.seed)
                    for name in order:
                        if scratch is not None:
                            scratch.add_(1)
                        stream.synchronize()
                        before = reader() if reader else None
                        begin_ns = time.perf_counter_ns()
                        outputs[name] = invoke(name)
                        submit_ns = time.perf_counter_ns()
                        events[name][1].synchronize()
                        complete_ns = time.perf_counter_ns()
                        after = reader() if reader else None
                        rows = make_records(mode=mode, distribution=distribution, pair_id=pair_id,
                                            path=name, order=order, begin_ns=begin_ns, submit_ns=submit_ns,
                                            complete_ns=complete_ns,
                                            gpu_us=events[name][0].elapsed_time(events[name][1]) * 1000,
                                            before_switches=before, after_switches=after)
                        records.extend(rows)
                        for row in rows:
                            logfile.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    if (pair_id + 1) % 128 == 0:
                        logfile.flush()
                        os.fsync(logfile.fileno())
                verify(args.samples - 1, distribution, outputs)
                checks.append(dict(phase="after_distribution", distribution=distribution,
                                   trace_index=0 if distribution == "hot" else args.samples - 1, mode=mode))
                logfile.flush()
                os.fsync(logfile.fileno())
            del invoke
        stream.synchronize()
        for graph in graphs.values():
            graph.close()
        graphs.clear()
        outputs.clear()
        events.clear()
    except BaseException as error:
        # A failed close/capture may still have Graph references in traceback.
        # Preserve the entire chain; outer run also retains its pinned baseline.
        retain_failed(graphs, outputs, paths, storage, error)
        raise
    summary = bench.summarize_records(records, COMPARISONS, threshold_us=args.threshold_us,
                                      bootstrap_iterations=0, seed=args.seed)
    summary.update(case_metadata=metadata, correctness=checks,
                   protocol=dict(samples_per_distribution=args.samples, independent_runs=1,
                                 graph=mode == "graph", seed=args.seed, warmup=args.warmup,
                                 input_preparation_timed=False, per_request_sync=True,
                                 cache_claim="scratch_disturbed_not_guaranteed_cold" if args.l2_disturb else "uncontrolled",
                                 runtime_l2_bytes=l2, scratch_bytes=2 * l2 if args.l2_disturb else 0,
                                 common_graph_guard=mode == "graph", instrumentation_subtracted=False,
                                 eager_gpu_interval="includes CPU dispatch gaps between event records",
                                 graph_gpu_interval="captured external nodes; not exclusive kernel execution time",
                                 wall_partition="submit call + completion synchronization; sync is not exclusive GPU wait",
                                 context_switch_scope="current thread; correlation is not causal attribution",
                                 screening=args.samples < 10000))
    summary["instrumentation_control"] = (bench.calibrate_timing(args.calibration_samples)
        if args.calibration_samples and mode == "graph" else {"status": "NOT_RUN", "reason": "Graph-only control or disabled"})
    write_new_json(output / "summary.json", summary)
    return summary


def run(args):
    # CPU identity and source checks precede allocating either complete baseline.
    config, metadata, _ = lookup.checkpoint_config(args.checkpoint, require_no_tables=True)
    source_config, source_metadata, index = lookup.checkpoint_config(args.source_checkpoint)
    if source_metadata != metadata:
        raise ValueError("baseline source and consumer config/tokenizer differ")
    bundle, oracle = lookup.private_read(args.bundle), lookup.private_read(args.private_inputs)
    identity = lookup.bundle_identity(bundle)
    import torch
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError("selected authorized CUDA device unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    node, cpu = bind_gpu_local_cpu(args.device, args.cpu_slot)
    torch.set_num_threads(1)
    from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState, HostEngramEmbedding
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import attach_from_bundle, hash_state_digest
    layout = EngramLayout(config)
    if lookup.layout_metadata(EngramLayout(source_config)) != lookup.layout_metadata(layout):
        raise ValueError("source layout mismatch")
    state = NgramHashState(layout, str(args.checkpoint))
    if hash_state_digest(state) != identity["hash_abi"]:
        raise ValueError("checkpoint hash ABI differs from publication")
    parsed = lookup.validate_oracle(oracle, layout=layout, metadata=metadata, identity=identity,
                                    token_count=len(state.token_map))
    lookup.verify_bundle_samples(parsed, bundle, layout)
    headers = source_sample_check(args.source_checkpoint, index, layout, parsed)
    layers = args.layers if args.layers is not None else [layout.layer_ids[0]]
    if not set(layers) <= set(layout.layer_ids):
        raise ValueError("selected layer absent from real checkpoint")
    if (args.samples + args.warmup + 2) * max(args.batches) * layout.n_hash_cols * 8 > 512 << 20:
        raise ValueError("preallocated trace exceeds 512MiB")
    spec = importlib.util.find_spec("engram_native_dev_492ecf45")
    if spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        raise RuntimeError("prebuilt genuine native bridge required")
    native = importlib.import_module("engram_native_dev_492ecf45")
    if any(native.resource_counters().values()):
        raise RuntimeError("benchmark must start with zero native resources")
    args.output.mkdir(mode=0o700)
    source_files = [Path(__file__), Path(lookup.__file__), Path(bench.__file__)]
    source_files += [REPO / "rtp_llm/models_py/modules/dsv41" / name
                     for name in ("engram.py", "_engram_triton.py", "engram_storage.py", "engram_bootstrap.py")]
    write_new_json(args.output / "run.json", dict(schema_version=1, identity=identity,
                   source_headers=headers, checkpoint_metadata=metadata, layers=layers,
                   batches=args.batches, modes=args.modes, samples=args.samples, seed=args.seed,
                   source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in source_files},
                   local_gpu_host_numa=node, local_cpu=cpu,
                   baseline="independent source mmap -> complete _PinnedSharedTable; extra copy is experiment control",
                   scope="single-layer lookup only; excludes hash, WKV, gate, model and HTTP"))
    storage, local, remote, control = None, None, None, None
    paths = {}
    completed = []
    try:
        storage = attach_from_bundle(args.bundle, layout=layout, hash_state=state, device=device)
        for layer in layers:
            rows = dict(zip(layout.layer_ids, layout.num_embeddings))[layer]
            emit("BASELINE_LOADING", layer=layer, logical_bytes=rows * (layout.head_dim + layout.head_dim // 32))
            control = make_local_table(args.source_checkpoint, index, layout, layer, device)
            numa = parse_numa_pages(Path("/proc/self/numa_maps").read_text(), control.storage.data_ptr(), node)
            local = HostEngramEmbedding(control.weight, control.scales, pinned=control)
            remote = HostEngramEmbedding.from_storage(storage, layer)
            emit("BASELINE_READY", layer=layer, numa=numa)
            for batch in args.batches:
                ids = torch.empty((batch, layout.n_hash_cols), dtype=torch.int64, device=device)
                generator = torch.Generator().manual_seed(args.seed + layer * 100 + batch)
                trace = torch.randint(rows, (args.samples + args.warmup + 2, batch, layout.n_hash_cols),
                                      dtype=torch.int64, generator=generator).pin_memory()
                paths = {"local_raw": lambda: local(ids, device), "remote_raw": lambda: remote(ids, device)}
                def update(sample, distribution):
                    ids.copy_(trace[0 if distribution == "hot" else sample % len(trace)], non_blocking=True)
                def verify(sample, distribution, outputs):
                    matrix = trace[0 if distribution == "hot" else sample % len(trace)].tolist()
                    expected = torch.tensor(source_rows(args.source_checkpoint, index, layout, layer, matrix),
                                            dtype=torch.float32).to(torch.bfloat16)
                    for name, value in outputs.items():
                        actual = value.cpu()
                        if actual.dtype != torch.bfloat16 or not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                            raise AssertionError(f"{name} actual measured output differs from independent source BF16 oracle")
                for mode in args.modes:
                    output = args.output / f"layer{layer}-M{batch}-{mode}"
                    output.mkdir(mode=0o700)
                    case = dict(layer_id=layer, actual_model_engram_layers=list(layout.layer_ids), layer_count_measured=1,
                                batch=batch, heads=layout.n_hash_cols, dim=layout.head_dim, table_rows=rows,
                                table_bytes=rows * (layout.head_dim + layout.head_dim // 32), mode=mode,
                                distribution="uniform rows over complete actual table; not a production token distribution",
                                trace_seed=args.seed + layer * 100 + batch,
                                trace_sha256=hashlib.sha256(memoryview(trace.numpy())).hexdigest(),
                                scope="IDs ready -> lookup BF16 ready; no hash/WKV/gate/model/HTTP",
                                baseline_numa=numa, torch=torch.__version__, cuda=torch.version.cuda,
                                gpu_name=torch.cuda.get_device_properties(device).name,
                                capability=list(torch.cuda.get_device_capability(device)), identity=identity)
                    emit("CASE_START", layer=layer, batch=batch, mode=mode, samples=args.samples)
                    result = benchmark_case(paths=paths, storage=storage, update=update, verify=verify,
                                            args=args, metadata=case, output=output)
                    completed.append(output.name)
                    emit("CASE_COMPLETE", case=output.name, paired_deltas=result["paired_deltas"])
                paths.clear()
                del ids, trace, update, verify
            remote.release_external_views()
            remote = local = None
            gc.collect()
            close_local(control, torch)
            control = None
            gc.collect()
        storage.close()
        counters = native.resource_counters()
        if any(counters.values()):
            raise RuntimeError("native resources remain after ordered cleanup")
        write_new_json(args.output / "complete.json", dict(cases=completed, counters=counters,
                       decode_step_slo="NOT_MEASURED", scope="single-layer lookup component"))
        emit("CLOSED", cases=completed, counters=counters)
    except BaseException as error:
        retain_failed(control, local, remote, storage, paths, error)
        emit("FAILED_RETAINED_UNTIL_PROCESS_EXIT", error_type=type(error).__name__, completed=completed)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "source-checkpoint", "bundle", "private-inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batches", default="1")
    parser.add_argument("--layers", default="first")
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--distributions", default="hot,random")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4102101)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu-slot", type=int, default=0)
    parser.add_argument("--l2-disturb", action="store_true")
    parser.add_argument("--threshold-us", type=float, default=100.0)
    parser.add_argument("--calibration-samples", type=int, default=100)
    try:
        args = check_options(parser.parse_args(argv))
        run(args)
        return 0
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        emit("FAILED", error_type=type(error).__name__, message=str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
