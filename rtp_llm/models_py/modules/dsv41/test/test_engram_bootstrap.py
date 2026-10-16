# SPDX-License-Identifier: Apache-2.0
"""File/bootstrap CPU tests; native mapping is mocked, never model forward.

Hash tests instantiate the real EngramLayout/NgramHashState with an explicit
token map. They need CPU PyTorch, but no CUDA device. File checks run without it.
Actual GPU reads and cross-host imports belong to the integration harness.
"""
import base64
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "engram_bootstrap_cpu_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
b = importlib.import_module(PACKAGE + ".engram_bootstrap")
m = importlib.import_module(PACKAGE + ".engram_manifest")
fixture_spec = importlib.util.spec_from_file_location(
    "engram_bootstrap_manifest_fixture", ROOT / "test" / "test_engram_manifest.py"
)
fixture_module = importlib.util.module_from_spec(fixture_spec)
sys.modules[fixture_spec.name] = fixture_module
fixture_spec.loader.exec_module(fixture_module)
try:
    import torch
except ImportError:
    torch = None


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def real_hash_state():
    e = importlib.import_module(PACKAGE + ".engram")
    config = dict(
        engram_layer_ids=[1, 14], engram_num_embeddings=[128, 256],
        engram_max_ngram_size=3, engram_n_heads=2, engram_head_dim=32,
        engram_compressed_vocab_size=8, engram_pad_token_id=2, engram_vocab_size=5,
    )
    return e.NgramHashState(e.EngramLayout(config), token_map=list(range(8)))


class BundleFiles:
    def prepare_bundle(self, layout=None, hash_abi=None):
        self.temporary = tempfile.TemporaryDirectory(prefix="engram-bootstrap-ut-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "launch.json"
        self.layout = layout or types.SimpleNamespace(
            layer_ids=(1, 14), num_embeddings=(3, 5), head_dim=32
        )
        self.abi = hash_abi or "sha256:" + "b" * 64
        self.document = fixture_module.fixture()
        rows_by_layer = dict(zip(self.layout.layer_ids, self.layout.num_embeddings))
        for tensor in self.document["tensors"]:
            lid = int(tensor["name"].split(".")[1])
            rows = rows_by_layer[lid]
            columns = self.layout.head_dim if tensor["name"].endswith(".weight") else self.layout.head_dim // 32
            tensor["shape"] = [rows, columns]
            part = tensor["shards"][0]
            part.update(row_end=rows, logical_bytes=rows * columns,
                        byte_offset=0 if tensor["name"].endswith(".weight") else rows * self.layout.head_dim,
                        allocation_bytes=((rows * (self.layout.head_dim + self.layout.head_dim // 32) + 4095) // 4096) * 4096)
            tensor["row_bytes"] = columns
        self.document["hash_abi"] = self.abi
        self.document["dataset_id"] = m.compute_dataset_id(self.document)
        self.manifest_raw = json.dumps(self.document).encode()
        self.samples = {
            tensor["name"]: {
                "row_ids": sorted({0, tensor["shape"][0] // 2, tensor["shape"][0] - 1}),
                "sha256": "sha256:" + "c" * 64,
            }
            for tensor in self.document["tensors"]
        }
        self.bundle = dict(
            schema_version=1, manifest="manifest.json", capabilities="handles.json",
            manifest_sha256=digest(self.manifest_raw),
            dataset_id=self.document["dataset_id"], generation=self.document["generation"],
            model_revision=self.document["model_revision"], hash_abi=self.abi,
            samples=self.samples,
        )
        self.capabilities = {
            tensor["shards"][0]["allocation_id"]: base64.b64encode(b"private-handle-bytes").decode()
            for tensor in self.document["tensors"]
        }
        self.write("manifest.json", self.manifest_raw)
        self.write("handles.json", self.capabilities)
        self.write("launch.json", self.bundle)

    def write(self, name, value):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        path = self.root / name
        path.write_bytes(raw)
        path.chmod(0o600)
        return path

    def read(self):
        return b._read_bundle(self.path, self.layout, self.abi, limits=b._configured_limits({}))


class BundleFileTest(BundleFiles, unittest.TestCase):
    def setUp(self):
        self.prepare_bundle()

    def assert_code(self, code, function=None):
        with self.assertRaises((b.BootstrapError, m.ManifestError)) as raised:
            (function or self.read)()
        self.assertEqual(raised.exception.code, code)
        self.assertNotIn("private-handle", str(raised.exception))

    def test_real_private_files_validate_complete_generation(self):
        bundle, raw, expected, handles = self.read()
        self.assertEqual(bundle["dataset_id"], self.document["dataset_id"])
        self.assertEqual(raw, self.manifest_raw)
        self.assertEqual(expected.layers, ((1, 3), (14, 5)))
        self.assertEqual(set(handles), set(self.capabilities))
        self.assertEqual(set(handles.values()), {b"private-handle-bytes"})

    def test_large_metadata_requires_explicit_local_policy(self):
        # Only small JSON files are created: no table memory or native imports.
        rows = ((1 << 30) // 33) + 1
        self.prepare_bundle(types.SimpleNamespace(layer_ids=(1, 14),
                            num_embeddings=(rows, rows), head_dim=32))
        self.assert_code("INVALID_ALLOCATION_SIZE")
        limits = b._configured_limits({"RTP_ENGRAM_MAX_ALLOCATION_BYTES": str(128 << 30),
                                       "RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES": str(256 << 30)})
        bundle, _, _, _ = b._read_bundle(self.path, self.layout, self.abi, limits=limits)
        self.assertEqual(bundle["dataset_id"], self.document["dataset_id"])

    def test_total_capacity_counts_unique_backing_and_cannot_be_in_bundle(self):
        limits = b._configured_limits({"RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES": "8191"})
        self.assert_code("TOTAL_ALLOCATION_LIMIT", lambda: b._read_bundle(
            self.path, self.layout, self.abi, limits=limits))
        self.bundle["max_total_allocation_bytes"] = 256 << 30
        self.write("launch.json", self.bundle)
        self.assert_code("INVALID_BUNDLE_FIELDS")

    def test_real_table_metadata_needs_both_local_capacity_settings(self):
        self.prepare_bundle(types.SimpleNamespace(layer_ids=(1, 14),
                            num_embeddings=(384006168, 384016682), head_dim=256))
        self.assert_code("INVALID_ALLOCATION_SIZE")
        env = {"RTP_ENGRAM_MAX_ALLOCATION_BYTES": str(128 << 30)}
        self.assert_code("TOTAL_ALLOCATION_LIMIT", lambda: b._read_bundle(
            self.path, self.layout, self.abi, limits=b._configured_limits(env)))
        env["RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES"] = str(256 << 30)
        _, _, expected, handles = b._read_bundle(
            self.path, self.layout, self.abi, limits=b._configured_limits(env))
        self.assertEqual(expected.layers, ((1, 384006168), (14, 384016682)))
        self.assertEqual(expected.head_dim, 256)
        self.assertEqual(len(handles), 2)

    def test_invalid_local_policy_is_a_bounded_error(self):
        for value in ("0", "-1", str((128 << 30) + 1), "private-value"):
            self.assert_code("INVALID_CAPACITY_LIMIT", lambda: b._configured_limits(
                {"RTP_ENGRAM_MAX_ALLOCATION_BYTES": value}))

    def test_relative_bundle_rejected(self):
        self.assert_code("BUNDLE_PATH_MUST_BE_ABSOLUTE",
                         lambda: b._read_bundle("launch.json", self.layout, self.abi))

    def test_symlinked_file_rejected(self):
        self.path.unlink()
        self.path.symlink_to(self.root / "manifest.json")
        self.assert_code("BOOTSTRAP_FILE_UNAVAILABLE")

    def test_public_mode_rejected_for_each_file(self):
        for name in ("launch.json", "manifest.json", "handles.json"):
            with self.subTest(file=name):
                path = self.root / name
                path.chmod(0o644)
                self.assert_code("UNSAFE_BOOTSTRAP_FILE")
                path.chmod(0o600)

    def test_wrong_file_owner_rejected(self):
        with mock.patch.object(b.os, "getuid", return_value=os.getuid() + 1):
            self.assert_code("UNSAFE_BOOTSTRAP_FILE")

    def test_directory_and_oversized_bundle_rejected(self):
        self.assert_code("BOOTSTRAP_FILE_UNAVAILABLE",
                         lambda: b._read_bundle(self.root, self.layout, self.abi))
        self.write("launch.json", b" " * ((2 << 20) + 1))
        self.assert_code("BOOTSTRAP_TOO_LARGE")

    def test_duplicate_nonfinite_or_malformed_json_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'\xff', b'{} trailing'):
            with self.subTest(raw=raw):
                self.write("launch.json", raw)
                self.assert_code("INVALID_JSON")

    def test_bundle_fields_and_version_are_strict(self):
        for change, code in (({"unexpected": 1}, "INVALID_BUNDLE_FIELDS"),
                             ({"schema_version": True}, "UNSUPPORTED_BUNDLE_VERSION")):
            self.write("launch.json", {**self.bundle, **change})
            self.assert_code(code)

    def test_path_traversal_and_aliasing_rejected(self):
        for name, code in (("../manifest.json", "INVALID_BUNDLE_FILENAME"),
                           ("/manifest.json", "INVALID_BUNDLE_FILENAME"),
                           ("launch.json", "BOOTSTRAP_FILES_MUST_DIFFER")):
            with self.subTest(name=name):
                self.write("launch.json", {**self.bundle, "manifest": name})
                self.assert_code(code)

    def test_manifest_byte_tamper_rejected(self):
        self.write("manifest.json", self.manifest_raw + b" ")
        self.assert_code("MANIFEST_DIGEST_MISMATCH")

    def test_actual_hash_abi_pinned_before_other_files(self):
        (self.root / "manifest.json").unlink()
        self.assert_code("HASH_ABI_MISMATCH",
                         lambda: b._read_bundle(self.path, self.layout, "sha256:" + "e" * 64))

    def test_dataset_generation_and_layout_pin(self):
        for field, value, code in (
            ("dataset_id", "sha256:" + "0" * 64, "DATASET_MISMATCH"),
            ("generation", fixture_module.uid(999), "GENERATION_MISMATCH"),
            ("generation", "not-a-uuid", "INVALID_GENERATION"),
            ("model_revision", "other-model", "MODEL_CONTRACT_MISMATCH"),
        ):
            with self.subTest(field=field, value=value):
                self.write("launch.json", {**self.bundle, field: value})
                self.assert_code(code)
        self.write("launch.json", self.bundle)
        self.layout.num_embeddings = (4, 5)
        self.assert_code("INCOMPLETE_GENERATION")

    def test_all_tensor_samples_required(self):
        missing = copy.deepcopy(self.bundle)
        missing["samples"].pop(next(iter(missing["samples"])))
        self.write("launch.json", missing)
        self.assert_code("INCOMPLETE_READ_SAMPLES")

    def test_samples_require_exact_sorted_boundary_rows(self):
        name = next(iter(self.samples))
        for rows in ([0], [0, 2, 1], [0, True, 2], [0, 1, 1, 2]):
            changed = copy.deepcopy(self.bundle)
            changed["samples"][name]["row_ids"] = rows
            self.write("launch.json", changed)
            self.assert_code("INVALID_READ_SAMPLE")

    def test_capability_set_must_match_allocations(self):
        changed = dict(self.capabilities)
        changed.pop(next(iter(changed)))
        self.write("handles.json", changed)
        self.assert_code("CAPABILITY_SET_MISMATCH")

    def test_capability_encoding_is_bounded_and_strict(self):
        key = next(iter(self.capabilities))
        for encoded in ("", "?private-handle", "a" * 1025, 42):
            self.write("handles.json", {**self.capabilities, key: encoded})
            self.assert_code("INVALID_CAPABILITY_ENCODING")


@unittest.skipIf(torch is None, "CPU PyTorch required; no GPU is used")
class HashSemanticTest(unittest.TestCase):
    def test_real_hash_state_is_deterministic(self):
        self.assertEqual(b.hash_state_digest(real_hash_state()), b.hash_state_digest(real_hash_state()))

    def test_every_hash_constant_changes_digest(self):
        baseline = b.hash_state_digest(real_hash_state())
        for name in ("token_map", "multipliers", "primes", "offsets"):
            with self.subTest(constant=name):
                state = real_hash_state()
                tensor = state.layout.offsets if name == "offsets" else getattr(state, name)
                tensor.reshape(-1)[-1] += 1
                self.assertNotEqual(b.hash_state_digest(state), baseline)

    def test_pad_and_layout_semantics_change_digest(self):
        baseline = b.hash_state_digest(real_hash_state())
        for name, value in (("pad_id", 3), ("pad_token_id", 3), ("max_ngram_size", 4),
                            ("head_dim", 64), ("n_heads", 3),
                            ("compressed_vocab_size", 9), ("layer_ids", (1, 15)),
                            ("num_embeddings", (129, 256))):
            with self.subTest(field=name):
                state = real_hash_state()
                setattr(state if name == "pad_id" else state.layout, name, value)
                self.assertNotEqual(b.hash_state_digest(state), baseline)

    def test_equal_noncontiguous_constants_have_equal_digest(self):
        state = real_hash_state()
        baseline = b.hash_state_digest(state)
        state.token_map = state.token_map.repeat_interleave(2)[::2]
        self.assertFalse(state.token_map.is_contiguous())
        self.assertEqual(b.hash_state_digest(state), baseline)

    def test_non_cpu_or_wrong_dtype_rejected_without_gpu_allocation(self):
        for device, dtype in (("meta", torch.int64), ("cpu", torch.int32), ("cpu", torch.float32)):
            state = real_hash_state()
            state.token_map = torch.empty(state.token_map.shape, dtype=dtype, device=device)
            with self.assertRaisesRegex(b.BootstrapError, "HASH_CONSTANTS_MUST_BE_CPU_INT64"):
                b.hash_state_digest(state)


@unittest.skipIf(torch is None, "CPU PyTorch required; native/GPU calls are mocked")
class AttachFailureTest(BundleFiles, unittest.TestCase):
    def setUp(self):
        self.hash_state = real_hash_state()
        self.prepare_bundle(self.hash_state.layout, b.hash_state_digest(self.hash_state))
        self.storage = mock.Mock()
        self.storage.verify.return_value = None
        self.storage.close.return_value = None
        self.native = types.SimpleNamespace(max_allocation_bytes=128 << 30)
        self.retained = []
        self.native_patch = mock.patch.object(b, "_load_native", return_value=self.native)
        self.storage_patch = mock.patch.object(b.FabricEngramStorage, "attach", return_value=self.storage)
        self.retain_patch = mock.patch.object(b, "retain_for_process", side_effect=self.retained.append)
        self.native_mock = self.native_patch.start()
        self.attach_mock = self.storage_patch.start()
        self.retain_patch.start()
        self.addCleanup(self.native_patch.stop)
        self.addCleanup(self.storage_patch.stop)
        self.addCleanup(self.retain_patch.stop)

    def do_attach(self, device="cuda:2", environ=None):
        return b.attach_from_bundle(self.path, layout=self.layout,
                                    hash_state=self.hash_state, device=device,
                                    environ={} if environ is None else environ)

    def test_native_capacity_mismatch_rejects_before_first_import(self):
        for maximum in (None, True, 0, 4095):
            self.native.max_allocation_bytes = maximum
            with self.assertRaisesRegex(b.BootstrapError, "NATIVE_ALLOCATION_LIMIT"):
                self.do_attach()
            self.attach_mock.assert_not_called()

    def test_explicit_policy_is_forwarded_as_one_immutable_limits_object(self):
        env = {"RTP_ENGRAM_MAX_ALLOCATION_BYTES": str(128 << 30),
               "RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES": str(256 << 30)}
        self.do_attach(environ=env)
        limits = self.attach_mock.call_args.kwargs["limits"]
        self.assertEqual(limits.max_allocation_bytes, 128 << 30)
        self.assertEqual(limits.max_total_allocation_bytes, 256 << 30)

    def test_bad_policy_fails_before_native_load(self):
        with self.assertRaisesRegex(b.BootstrapError, "INVALID_CAPACITY_LIMIT"):
            self.do_attach(environ={"RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES": "0"})
        self.native_mock.assert_not_called()
        self.attach_mock.assert_not_called()

    def test_success_calls_verify_before_returning_storage(self):
        self.assertIs(self.do_attach(), self.storage)
        self.attach_mock.assert_called_once()
        self.assertEqual(self.attach_mock.call_args.kwargs["device"], 2)
        self.assertIs(self.attach_mock.call_args.kwargs["native"], self.native)
        self.storage.verify.assert_called_once()
        self.storage.close.assert_not_called()
        self.assertEqual(self.retained, [])  # Model-level success owns retention.

    def test_cpu_device_rejected_before_read_or_import(self):
        self.path.unlink()
        with self.assertRaisesRegex(b.BootstrapError, "CUDA_DEVICE_REQUIRED"):
            self.do_attach("cpu")
        self.native_mock.assert_not_called()
        self.attach_mock.assert_not_called()

    def test_hash_mismatch_rejected_before_native_import(self):
        self.hash_state.token_map[-1] -= 1
        with self.assertRaisesRegex(b.BootstrapError, "HASH_ABI_MISMATCH"):
            self.do_attach()
        self.native_mock.assert_not_called()
        self.attach_mock.assert_not_called()

    def test_missing_sample_rejected_before_native_import(self):
        self.bundle["samples"].pop(next(iter(self.bundle["samples"])))
        self.write("launch.json", self.bundle)
        with self.assertRaisesRegex(b.BootstrapError, "INCOMPLETE_READ_SAMPLES"):
            self.do_attach()
        self.attach_mock.assert_not_called()

    def test_import_cleanup_pending_resource_is_retained(self):
        pending = b.CleanupPendingError("injected import cleanup failure", self.storage)
        self.attach_mock.side_effect = pending
        with self.assertRaises(b.CleanupPendingError) as raised:
            self.do_attach()
        self.assertIs(raised.exception, pending)
        self.assertEqual(self.retained, [self.storage])
        self.storage.verify.assert_not_called()

    def test_verify_failure_closes_storage_and_preserves_original_error(self):
        failure = b.BootstrapError("GPU_READ_SAMPLE_MISMATCH")
        self.storage.verify.side_effect = failure
        with self.assertRaises(b.BootstrapError) as raised:
            self.do_attach()
        self.assertIs(raised.exception, failure)
        self.storage.close.assert_called_once()
        self.assertEqual(self.retained, [])

    def test_verify_and_close_failure_retains_resource(self):
        failure = RuntimeError("injected asynchronous verification failure")
        self.storage.verify.side_effect = failure
        self.storage.close.side_effect = RuntimeError("injected drain failure")
        with self.assertRaises(b.CleanupPendingError) as raised:
            self.do_attach()
        self.assertIs(raised.exception.resource, self.storage)
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(self.retained, [self.storage])

    def test_sample_callback_compares_bytes_and_synchronizes(self):
        validated = m.validate_for_attach(m.parse_manifest(self.manifest_raw),
                    m.ExpectedLayout(layers=tuple(zip(self.layout.layer_ids, self.layout.num_embeddings)),
                                     head_dim=32, model_revision=self.bundle["model_revision"], hash_abi=self.abi),
                    self.bundle["dataset_id"], self.bundle["generation"])
        views, samples = {}, {}
        for spec in validated.manifest.tensors:
            view = torch.arange(spec.shape[0] * spec.shape[1], dtype=torch.int64).to(torch.uint8).reshape(spec.shape)
            rows = sorted({0, spec.shape[0] // 2, spec.shape[0] - 1})
            views[spec.name] = view
            samples[spec.name] = dict(row_ids=rows, sha256=digest(view[rows].numpy().tobytes()))
        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            self.assertTrue(b._verify_samples(views, validated, samples, "cpu"))
            synchronize.assert_called_once_with("cpu")
        first = validated.manifest.tensors[0].name
        views[first][0, 0] ^= 1
        with mock.patch.object(torch.cuda, "synchronize"):
            with self.assertRaisesRegex(b.BootstrapError, "GPU_READ_SAMPLE_MISMATCH"):
                b._verify_samples(views, validated, samples, "cpu")


if __name__ == "__main__":
    unittest.main()
