"""CPU contracts for the real-table latency harness; no fabricated GPU pass."""
import copy
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("checkpoint_latency_under_test", Path(__file__).with_name("checkpoint_latency.py"))
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)


class LatencyContractTest(unittest.TestCase):
    def options(self, root):
        return SimpleNamespace(checkpoint=root / "consumer", source_checkpoint=root / "source",
                               bundle=root / "bundle.json", private_inputs=root / "oracle.json",
                               output=root / "new-run", batches="1,2,4,8,16,32,64", layers="first",
                               modes="eager,graph", distributions="hot,random", samples=100,
                               warmup=20, device=0, cpu_slot=0, seed=7,
                               layer_mode="single", memory_headroom_gib=16,
                               discard_private_source_cache=False,
                               calibration_samples=100, threshold_us=100.0)

    def test_default_screen_and_all_supported_decode_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            args = h.check_options(self.options(Path(directory)))
        self.assertEqual(args.batches, [1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(args.modes, ["eager", "graph"])
        self.assertIsNone(args.layers)

    def test_single_default_and_together_cover_actual_model_layers_in_order(self):
        self.assertEqual(h.layer_groups((1, 14), None, "single"), [(1,)])
        self.assertEqual(h.layer_groups((1, 14), [14, 1], "single"), [(14,), (1,)])
        self.assertEqual(h.layer_groups((1, 14), None, "together"), [(1, 14)])
        self.assertEqual(h.layer_groups((1, 14), [14, 1], "together"), [(1, 14)])
        for selected in ([1], [2, 14], [1, 1], []):
            with self.assertRaises(ValueError):
                h.layer_groups((1, 14), selected, "together")

    def test_together_invokes_each_layer_once_with_its_own_ids_and_returns_both(self):
        calls, ids, results = [], {1: object(), 14: object()}, {1: object(), 14: object()}
        def embedding(layer):
            def forward(value, device):
                calls.append((layer, value, device))
                return results[layer]
            return forward
        layers = {1: embedding(1), 14: embedding(14)}
        actual = h.invoke_lookup_chain(layers, (1, 14), ids, "cuda:0")
        self.assertEqual(calls, [(1, ids[1], "cuda:0"), (14, ids[14], "cuda:0")])
        self.assertEqual(actual, (results[1], results[14]))
        self.assertIs(h.invoke_lookup_chain(layers, (1,), ids, "cuda:0"), results[1])

    def test_joint_output_verification_reaches_both_paths_and_layers_and_rejects_missing(self):
        outputs = {"local_raw": (object(), object()), "remote_raw": (object(), object())}
        grouped = h.split_layer_outputs(outputs, (1, 14))
        for position, layer in enumerate((1, 14)):
            for name in outputs:
                self.assertIs(grouped[layer][name], outputs[name][position])
        for value in (object(), (object(),), (object(), object(), object())):
            with self.assertRaises(AssertionError):
                h.split_layer_outputs(dict(outputs, remote_raw=value), (1, 14))
        self.assertIs(h.split_layer_outputs({"local_raw": outputs["local_raw"][0]}, (1,))[1]["local_raw"], outputs["local_raw"][0])

    def test_explicit_layers_and_invalid_lists(self):
        self.assertEqual(h.integer_list("1,14"), [1, 14])
        for value in ("", "1,1", "1,-1", " 1", "1,", "1.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                h.integer_list(value)
        with self.assertRaises(ValueError):
            h.integer_list("3", allowed=(1, 2, 4))
        for value in ("eager,eager", "graph,other", ""):
            with self.assertRaises(ValueError):
                h.choice_list(value, ("eager", "graph"))

    def test_rejects_invalid_protocol_before_any_cuda_import(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.options(Path(directory))
            for key, value in (("samples", 0), ("samples", 1_000_001), ("warmup", 0),
                               ("device", -1), ("cpu_slot", -1), ("seed", -1),
                               ("calibration_samples", -1), ("threshold_us", float("nan")),
                               ("memory_headroom_gib", 0), ("layer_mode", "invalid"),
                               ("checkpoint", Path("relative"))):
                args = copy.copy(base)
                setattr(args, key, value)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    h.check_options(args)

    def test_refuses_prior_output_instead_of_overwriting_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.options(Path(directory))
            args.output.mkdir()
            with self.assertRaises(FileExistsError):
                h.check_options(args)

    def test_numa_requires_all_resident_pages_on_selected_node(self):
        text = "1000 default file=/dev/shm/rtp_llm_engram_1_abc N0=12 kernelpagesize_kB=64\n"
        self.assertEqual(h.parse_numa_pages(text, 0x1000, 0)["node_pages"], {"0": 12})
        for other in (text.replace("N0=12", "N1=12"), text.replace("N0=12", "N0=11 N1=1"),
                      text.replace("N0=12", "N0=0"), text + text, text.replace("1000", "2000")):
            with self.assertRaises(RuntimeError):
                h.parse_numa_pages(other, 0x1000, 0)

    def test_resolves_current_cgroup_from_mount_root_without_guessing(self):
        mount = "29 23 0:26 /jobs/task /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
        self.assertEqual(h.resolve_cgroup_directory("0::/jobs/task/worker\n", mount),
                         (Path("/sys/fs/cgroup/worker"), Path("/sys/fs/cgroup"), False))
        namespaced = "29 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
        self.assertEqual(h.resolve_cgroup_directory("0::/\n", namespaced)[0], Path("/sys/fs/cgroup"))
        for membership, mounts in (("1:memory:/x", mount), ("0::/unrelated", mount), ("0::/jobs/task", mount + mount)):
            with self.assertRaises(RuntimeError):
                h.resolve_cgroup_directory(membership, mounts)

    def test_only_hierarchy_root_may_omit_limit_and_missing_usage_always_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "worker"; child.mkdir()
            (root / "memory.current").write_text("10")
            (child / "memory.current").write_text("5")
            (child / "memory.max").write_text("100")
            values = h.read_cgroup_memory(child, root, True)
            self.assertEqual([item["max_bytes"] for item in values], [100, None])
            self.assertEqual(values[-1]["limit_source"], "root_no_limit_interface")
            with self.assertRaises(FileNotFoundError):
                h.read_cgroup_memory(child, root, False)
            (child / "memory.max").unlink()
            with self.assertRaises(FileNotFoundError):
                h.read_cgroup_memory(child, root, True)
            (root / "memory.current").unlink()
            with self.assertRaises(FileNotFoundError):
                h.read_cgroup_memory(root, root, True)

    def test_memory_kib_fields_accept_numa_prefix_and_reject_missing(self):
        self.assertEqual(h.parse_kib_field("Node 0 MemFree: 100 kB\n", "MemFree"), 102400)
        self.assertEqual(h.parse_kib_field("MemAvailable: 200 kB\n", "MemAvailable"), 204800)
        for text in ("", "MemFree: 12 bytes\n", "MemFree: 1 kB\nMemFree: 2 kB\n"):
            with self.assertRaises(RuntimeError):
                h.parse_kib_field(text, "MemFree")

    def memory(self):
        gib = 1 << 30
        return dict(cgroups=[dict(path="current", current_bytes=94 * gib, max_bytes=384 * gib)],
                    shm_free_bytes=162 * gib, host_available_bytes=400 * gib, numa_free_bytes=350 * gib)

    def test_loading_budget_includes_complete_source_cache_and_reserved_headroom(self):
        gib = 1 << 30
        args = dict(table_bytes=95 * gib, source_bytes=95 * gib, headroom_bytes=16 * gib, before=True)
        self.assertEqual(h.check_memory_budget(self.memory(), **args)["required_marginal_bytes"], 190 * gib)
        snapshot = self.memory()
        snapshot["cgroups"][0]["current_bytes"] = 188 * gib
        with self.assertRaisesRegex(RuntimeError, "cgroup"):
            h.check_memory_budget(snapshot, **args)
        snapshot = self.memory()
        snapshot["cgroups"].append(dict(path="ancestor", current_bytes=350 * gib, max_bytes=384 * gib))
        with self.assertRaisesRegex(RuntimeError, "cgroup"):
            h.check_memory_budget(snapshot, **args)
        for key in ("shm_free_bytes", "host_available_bytes", "numa_free_bytes"):
            snapshot = self.memory(); snapshot[key] = 0
            with self.assertRaises(RuntimeError):
                h.check_memory_budget(snapshot, **args)

    def test_postload_budget_checks_actual_headroom_without_readding_loaded_table(self):
        gib = 1 << 30
        snapshot = self.memory()
        snapshot["cgroups"][0]["current_bytes"] = 360 * gib
        result = h.check_memory_budget(snapshot, table_bytes=95*gib, source_bytes=95*gib,
                                       headroom_bytes=16*gib, before=False)
        self.assertEqual(result["required_marginal_bytes"], 0)
        snapshot["cgroups"][0]["current_bytes"] = 370 * gib
        with self.assertRaises(RuntimeError):
            h.check_memory_budget(snapshot, table_bytes=95*gib, source_bytes=95*gib,
                                  headroom_bytes=16*gib, before=False)

    def test_private_file_cache_advice_requires_readonly_single_link_owned_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "table.safetensors"
            path.write_bytes(b"immutable source bytes")
            path.chmod(0o400)
            with patch.object(os, "posix_fadvise", create=True) as advice, patch.object(os, "POSIX_FADV_DONTNEED", 4, create=True):
                record = h.private_source_cache_advice(root, [path])
                fd, offset, size, mode = advice.call_args[0]
                self.assertEqual((offset, size, mode), (0, 0, 4))
                self.assertEqual(record[0]["reclaimed_bytes"], "NOT_ASSUMED")
                with self.assertRaises(OSError):
                    os.fstat(fd)
                advice.reset_mock()
                for mode in (0o600, 0o444, 0o644):
                    path.chmod(mode)
                    with self.assertRaises(ValueError):
                        h.private_source_cache_advice(root, [path])
                advice.assert_not_called()
                path.chmod(0o400)
                os.link(path, root / "shared-hardlink")
                with self.assertRaises(ValueError):
                    h.private_source_cache_advice(root, [path])
                advice.assert_not_called()

    def test_real_numa_metadata_survives_summary_validation_and_json_write(self):
        text = "1000 default file=/dev/shm/rtp_llm_engram_1_abc N1=1546692 kernelpagesize_kB=64\n"
        numa = h.parse_numa_pages(text, 0x1000, 1)
        summary = {"case_metadata": {"baseline_numa": numa}, "model_slo": "NOT_RUN"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            h.write_new_json(path, summary)
            actual = json.loads(path.read_text())
        self.assertEqual(actual, summary)
        self.assertEqual(actual["case_metadata"]["baseline_numa"]["node_pages"], {"1": 1546692})

    def test_selected_row_oracle_chunks_large_batch_without_losing_rows(self):
        chunks = h.chunk_selected_ids(list(range(1536)) + [-1, 7, 7])
        self.assertEqual(list(map(len, chunks)), [512, 512, 512])
        self.assertEqual([row for chunk in chunks for row in chunk], list(range(1536)))
        self.assertEqual(h.chunk_selected_ids([-1]), [])
        for values in ([-2], [True], [1.5]):
            with self.assertRaises(ValueError):
                h.chunk_selected_ids(values)

    def test_independent_source_oracle_preserves_order_duplicates_and_mask(self):
        layout = SimpleNamespace(layer_ids=(1, 14), head_dim=32)
        def selected(source, index, layout, rows):
            self.assertEqual(rows["14"], set())
            values = [{"row_id": row, "fp8_hex": (b"\x38" * 32).hex(), "e8m0_hex": "7f"}
                      for row in rows["1"]]
            return {"1": values, "14": []}, {}
        with patch.object(h.lookup, "read_selected_rows", side_effect=selected):
            result = h.source_rows(Path("/source"), {}, layout, 1, [[7, -1, 7]])
        self.assertEqual(result, [[[1.0] * 32, [0.0] * 32, [1.0] * 32]])

    def test_source_samples_must_match_admitted_remote_generation(self):
        parsed = {"1": {7: (b"\x38" * 32, b"\x7f")}}
        values = {"1": [{"row_id": 7, "fp8_hex": (b"\x38" * 32).hex(), "e8m0_hex": "7f"}]}
        with patch.object(h.lookup, "read_selected_rows", return_value=(values, {"a": "digest"})):
            self.assertEqual(h.source_sample_check(None, None, None, parsed), {"a": "digest"})
        values["1"][0]["e8m0_hex"] = "80"
        with patch.object(h.lookup, "read_selected_rows", return_value=(values, {})), self.assertRaisesRegex(ValueError, "source differs"):
            h.source_sample_check(None, None, None, parsed)

    def records(self, mode="graph", path="remote_raw", pair=0, gpu=5.0):
        return h.make_records(mode=mode, distribution="random", pair_id=pair, path=path,
                              order=["remote_raw", "local_raw"], begin_ns=1000, submit_ns=8000,
                              complete_ns=21000, gpu_us=gpu,
                              before_switches=(2, 3), after_switches=(2, 4))

    def test_wall_is_split_without_calling_sync_gpu_wait(self):
        gpu, wall = self.records()
        self.assertEqual(gpu["duration_us"], 5)
        self.assertEqual(wall["duration_us"], 20)
        self.assertEqual(wall["cpu_submit_call_us"], 7)
        self.assertEqual(wall["cpu_completion_sync_call_us"], 13)
        self.assertNotIn("cpu_graph_replay_call_us", wall)
        self.assertEqual(wall["thread_nivcsw_delta"], 1)

    def test_eager_and_graph_gpu_intervals_are_distinct_metrics(self):
        self.assertEqual(self.records("graph")[0]["clock"], "gpu_graph_event_interval")
        self.assertEqual(self.records("eager")[0]["clock"], "gpu_eager_event_interval")
        for gpu in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.records(gpu=gpu)

    def test_paired_extra_latency_retains_negative_values_and_violations(self):
        records = (self.records(path="remote_raw", pair=0, gpu=180) +
                   self.records(path="local_raw", pair=0, gpu=20) +
                   self.records(path="remote_raw", pair=1, gpu=10) +
                   self.records(path="local_raw", pair=1, gpu=30))
        summary = h.bench.summarize_records(records, h.COMPARISONS, bootstrap_iterations=0)
        delta = next(item for item in summary["paired_deltas"] if item["clock"] == "gpu_graph_event_interval")
        self.assertEqual(delta["paired_delta_stats"]["min_us"], -20)
        self.assertEqual(delta["paired_delta_stats"]["max_us"], 160)
        self.assertEqual(delta["paired_delta_stats"]["over_threshold_count"], 1)
        self.assertEqual(summary["model_slo"], "NOT_RUN")

    def test_result_creation_exclusive_and_capabilities_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            h.write_new_json(path, {"model_slo": "NOT_RUN"})
            self.assertEqual(json.loads(path.read_text())["model_slo"], "NOT_RUN")
            with self.assertRaises(FileExistsError):
                h.write_new_json(path, {})
            with self.assertRaises(ValueError):
                h.write_new_json(Path(directory) / "secret.json", {"fabric_handle": "secret"})

    def test_failed_cleanup_disables_finalizers_and_retains_resources(self):
        resource = SimpleNamespace(_registration_finalizer=SimpleNamespace(atexit=True),
                                   _lease_finalizer=SimpleNamespace(atexit=True))
        count = len(h._FAILED_RESOURCES)
        h.retain_failed(resource)
        self.assertFalse(resource._registration_finalizer.atexit)
        self.assertFalse(resource._lease_finalizer.atexit)
        self.assertIs(h._FAILED_RESOURCES[-1][0], resource)
        del h._FAILED_RESOURCES[count:]

    def test_top_level_failure_never_prints_sensitive_exception_arguments(self):
        secret = "PRIVATE_FABRIC_CAPABILITY_DO_NOT_LOG"
        error = TypeError("incompatible function arguments; Invoked with: b'" + secret + "'")
        error.code = secret
        argv = []
        for name in ("checkpoint", "source-checkpoint", "bundle", "private-inputs", "output"):
            argv += ["--" + name, "/private/example/" + name]
        output = io.StringIO()
        with patch.object(h, "check_options", return_value=object()), patch.object(h, "run", side_effect=error), redirect_stdout(output):
            self.assertEqual(h.main(argv), 1)
        record = json.loads(output.getvalue())
        self.assertEqual(record["status"], "FAILED")
        self.assertEqual(record["error_type"], "TypeError")
        self.assertNotIn(secret, output.getvalue())
        self.assertNotIn("Invoked with", output.getvalue())
        self.assertNotIn("message", record)
        self.assertNotIn("code", record)

    def local_cleanup(self, status=0, drain_fail=False):
        calls = []
        def drain(device):
            calls.append("drain")
            if drain_fail:
                raise RuntimeError("GPU drain failed")
        def unregister(pointer):
            calls.append("unregister")
            return (status,)
        def detach():
            calls.append("detach_finalizer")
            return (object(), (), {})
        control = SimpleNamespace(device=0, storage=SimpleNamespace(data_ptr=lambda: 123),
                                  runtime=SimpleNamespace(cudaHostUnregister=unregister),
                                  _registration_finalizer=SimpleNamespace(alive=True, detach=detach),
                                  _lease_finalizer=lambda: calls.append("release_lease"))
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=drain, device=lambda _: nullcontext()))
        return control, torch, calls

    def test_baseline_unregister_requires_drain_and_lease_release_is_last(self):
        control, torch, calls = self.local_cleanup()
        h.close_local(control, torch)
        self.assertEqual(calls, ["drain", "unregister", "detach_finalizer", "release_lease"])

    def test_failed_drain_never_unregisters_pinned_baseline(self):
        control, torch, calls = self.local_cleanup(drain_fail=True)
        with self.assertRaisesRegex(RuntimeError, "drain failed"):
            h.close_local(control, torch)
        self.assertEqual(calls, ["drain"])

    def test_failed_unregister_preserves_finalizer_and_lease(self):
        control, torch, calls = self.local_cleanup(status=1)
        with self.assertRaisesRegex(RuntimeError, "Unregister failed"):
            h.close_local(control, torch)
        self.assertEqual(calls, ["drain", "unregister"])


if __name__ == "__main__":
    unittest.main()
