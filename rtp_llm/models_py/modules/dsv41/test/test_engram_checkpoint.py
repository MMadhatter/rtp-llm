# SPDX-License-Identifier: Apache-2.0
"""CPU tests with real torch/safetensors and resource-accounting FakeNative.

The real runtime dependencies are mandatory: missing wheels fail this suite
instead of reporting success after skipping the checkpoint loader tests.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.modules.dsv41 import engram_checkpoint as m


class FakeNative:
    max_allocation_bytes = 1 << 30
    max_load_chunk_bytes = 64 << 20

    def __init__(self, granularity=4096):
        self.granularity = granularity
        self.owners, self.events, self.close_attempts = [], [], []
        self.fail_create = self.fail_load = self.fail_export = None
        self.load_error = RuntimeError
        self.fail_close = set()
        self.load_hook = None
        self.load_calls = self.query_calls = 0

    def host_numa_granularity(self, numa, device):
        self.query_calls += 1
        return self.granularity

    def Owner(self, size, numa, device):
        index = len(self.owners)
        if index == self.fail_create:
            raise RuntimeError("injected create failure")
        result = FakeOwner(self, index, size)
        self.owners.append(result)
        self.events.append(("create", index))
        return result


class FakeOwner:
    def __init__(self, native, index, size):
        self.native, self.index, self.size = native, index, size
        self.padded = (size + native.granularity - 1) // native.granularity * native.granularity
        self.data = bytearray(self.padded)
        self.closed = self.sealed = False

    def info(self):
        return dict(allocation_bytes=self.padded, granularity=self.native.granularity)

    def load(self, data, offset=0):
        assert not self.closed and not self.sealed
        assert len(data) <= self.native.max_load_chunk_bytes
        assert 0 <= offset <= offset + len(data) <= self.size
        self.native.load_calls += 1
        if self.native.load_hook:
            self.native.load_hook(self.native.load_calls)
        if self.native.load_calls == self.native.fail_load:
            raise self.native.load_error("injected load failure")
        self.data[offset:offset + len(data)] = data
        self.native.events.append(("load", self.index, offset, len(data)))

    def export_handle(self):
        assert not self.closed
        if self.index == self.native.fail_export:
            raise RuntimeError("injected export failure")
        self.sealed = True
        self.native.events.append(("export", self.index))
        return f"private-capability-{self.index}".encode()

    def close(self):
        self.native.close_attempts.append(self.index)
        if self.index in self.native.fail_close:
            raise RuntimeError("injected close failure")
        self.closed = True


class DependencyFreeTest(unittest.TestCase):
    def test_duplicate_json_and_nonfinite_constants_are_rejected(self):
        for data in (b'{"weight_map":{},"weight_map":{}}', b'{"x":NaN}'):
            with self.assertRaises(m.CheckpointError):
                m._json(data)

    def test_unsafe_paths_are_rejected(self):
        for relative in ("/tmp/x.safetensors", "../x.safetensors", "a/../x.safetensors",
                         "a//x.safetensors", "./x.safetensors", "x\\y.safetensors",
                         "x\x00.safetensors", "model.bin", ""):
            with self.subTest(relative=relative), self.assertRaises(m.CheckpointError):
                m._parts(relative)

    def test_pending_generation_never_exposes_capabilities_or_repr(self):
        generation = m.LoadedGeneration()
        generation._handles["id"] = b"secret"
        self.assertNotIn("secret", repr(generation))
        with self.assertRaises(m.CheckpointError):
            generation.private_handles
        generation.close()
        generation.close()


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.layout = m.ExpectedLayout(layers=((1, 5), (2, 131)), head_dim=64,
                                       model_revision="loader-test-v1", hash_abi="sha256:" + "a" * 64)
        self.index, self.payloads = {}, {}
        self.write_checkpoint()

    def write_index(self):
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": self.index}))

    def write_checkpoint(self, *, scale_fp8=False, weight_dtype=None):
        shards = ({}, {})
        self.index, self.payloads = {}, {}
        for layer, rows in self.layout.layers:
            for role, columns in (("weight", 64), ("scale", 2)):
                name = f"layers.{layer}.engram.embed.{role}"
                raw = (torch.arange(rows * columns, dtype=torch.int64) % 256).to(torch.uint8).reshape(rows, columns)
                dtype = (weight_dtype or torch.float8_e4m3fn) if role == "weight" else (
                    torch.float8_e8m0fnu if scale_fp8 else torch.uint8)
                byte_dtypes = (torch.uint8, torch.float8_e4m3fn, getattr(torch, "float8_e8m0fnu", None))
                tensor = raw.view(dtype) if dtype in byte_dtypes else raw.to(dtype)
                shards[0 if role == "weight" else 1][name] = tensor
                self.payloads[name] = tensor.view(torch.uint8).numpy().tobytes()
                self.index[name] = "weights.safetensors" if role == "weight" else "scales.safetensors"
        for filename, tensors in zip(("weights.safetensors", "scales.safetensors"), shards):
            save_file(tensors, self.root / filename)
        self.write_index()

    def inspect(self, **changes):
        options = dict(allocation_granularity=4096, budget_bytes=16384)
        options.update(changes)
        return m.inspect_checkpoint(self.root, self.layout, **options)

    def load(self, native=None, plan=None, **changes):
        options = dict(native=native or FakeNative(), device=0, numa=0, chunk_bytes=128)
        options.update(changes)
        return m.load_checkpoint(plan or self.inspect(), **options)

    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(m.CheckpointError) as error:
            function(*args, **kwargs)
        self.assertEqual(error.exception.code, code)

    def test_multifile_complete_generation_full_hashes_and_bounded_chunks(self):
        native = FakeNative()
        plan = self.inspect()
        self.assertEqual(plan.physical_bytes, 16384)
        result = self.load(native, plan)
        self.assertEqual(len(native.owners), 2)
        self.assertEqual(len(result.private_handles), 2)
        self.assertNotIn("private-capability", repr(result))
        with self.assertRaises(TypeError):
            result.private_handles["new"] = b"cannot mutate"
        for tensor in result.manifest.tensors:
            part = tensor.shards[0]
            layer = int(tensor.name.split(".")[1])
            owner = native.owners[layer - 1]
            payload = bytes(owner.data[part.byte_offset:part.byte_offset + part.logical_bytes])
            self.assertEqual(payload, self.payloads[tensor.name])
            self.assertEqual(part.sha256, "sha256:" + hashlib.sha256(payload).hexdigest())
        self.assertTrue(all(event[3] <= 128 for event in native.events if event[0] == "load"))
        kinds = [event[0] for event in native.events]
        self.assertLess(max(i for i, kind in enumerate(kinds) if kind == "load"), kinds.index("export"))
        result.close()
        self.assertEqual(native.close_attempts, [1, 0])
        result.close()
        self.assertTrue(all(owner.closed for owner in native.owners))
        self.assert_code("GENERATION_NOT_READY", lambda: result.private_handles)

    def test_e8m0_scale_bits_are_preserved(self):
        self.write_checkpoint(scale_fp8=True)
        result = self.load()
        for tensor in result.manifest.tensors:
            self.assertEqual(tensor.shards[0].sha256,
                             "sha256:" + hashlib.sha256(self.payloads[tensor.name]).hexdigest())
        result.close()

    def test_budget_and_padded_allocation_limits_are_preallocation(self):
        self.assert_code("BUDGET_EXCEEDED", self.inspect, budget_bytes=16383)
        self.assert_code("ALLOCATION_LIMIT", self.inspect, max_allocation_bytes=12287)
        self.assertEqual(self.inspect(max_allocation_bytes=12288).physical_bytes, 16384)
        self.assert_code("INVALID_ARGUMENT", self.inspect, max_allocation_bytes=(128 << 30) + 1)

    def test_explicit_large_policy_preserves_small_fixture_and_default(self):
        default = self.inspect()
        self.assertEqual(default.max_allocation_bytes, 1 << 30)
        large = self.inspect(max_allocation_bytes=128 << 30, budget_bytes=256 << 30)
        self.assertEqual(large.layers, default.layers)
        self.assertEqual(large.physical_bytes, default.physical_bytes)
        native = FakeNative()
        loaded = self.load(native, large)
        self.assertEqual(loaded.physical_bytes, default.physical_bytes)
        loaded.close()

    def test_capacity_policy_rejected_before_source_access_or_owner(self):
        for options in (dict(budget_bytes=(256 << 30) + 1),
                        dict(max_allocation_bytes=True), dict(budget_bytes=False)):
            with patch.object(m, "_path", side_effect=AssertionError("must not access source")):
                self.assert_code("INVALID_ARGUMENT", self.inspect, **options)

    def test_native_granularity_and_limits_rechecked_before_owner(self):
        plan = self.inspect()
        native = FakeNative(granularity=8192)
        with self.assertRaises(m.CheckpointError):
            self.load(native, plan)
        self.assertEqual(native.owners, [])
        native = FakeNative()
        native.max_allocation_bytes = 4096
        self.assert_code("ALLOCATION_LIMIT", self.load, native, plan)
        self.assertEqual(native.owners, [])
        self.assert_code("CHUNK_SMALLER_THAN_ROW", self.load, native, plan, chunk_bytes=63)
        self.assert_code("INVALID_ARGUMENT", self.load, native, plan, chunk_bytes=(64 << 20) + 1)
        self.assertEqual(native.owners, [])

    def test_missing_layer_and_wrong_index_shard_are_rejected(self):
        self.index.pop("layers.2.engram.embed.scale")
        self.write_index()
        self.assert_code("MISSING_TENSOR", self.inspect)
        self.write_checkpoint()
        self.index["layers.1.engram.embed.weight"] = "scales.safetensors"
        self.write_index()
        with self.assertRaises(Exception):
            self.inspect()

    def test_checkpoint_paths_cannot_escape_or_traverse_symlinks(self):
        self.index["layers.1.engram.embed.weight"] = "../weights.safetensors"
        self.write_index()
        self.assert_code("UNSAFE_INDEX_PATH", self.inspect)
        self.write_checkpoint()
        (self.root / "alias.safetensors").symlink_to(self.root / "weights.safetensors")
        self.index["layers.1.engram.embed.weight"] = "alias.safetensors"
        self.write_index()
        self.assert_code("UNSAFE_INDEX_PATH", self.inspect)

    def test_wrong_dtype_and_shape_are_rejected(self):
        self.write_checkpoint(weight_dtype=torch.float16)
        self.assert_code("UNSUPPORTED_DTYPE", self.inspect)
        self.write_checkpoint()
        wrong = m.ExpectedLayout(layers=((1, 6), (2, 131)), head_dim=64,
                                 model_revision=self.layout.model_revision, hash_abi=self.layout.hash_abi)
        self.assert_code("SHAPE_MISMATCH", m.inspect_checkpoint, self.root, wrong,
                         allocation_granularity=4096, budget_bytes=16384)

    def test_lookup_width_and_manifest_tensor_count_fail_in_preflight(self):
        for layout, code in (
                (m.ExpectedLayout(layers=self.layout.layers, head_dim=96,
                                  model_revision=self.layout.model_revision, hash_abi=self.layout.hash_abi),
                 "UNSUPPORTED_LOOKUP_WIDTH"),
                (m.ExpectedLayout(layers=tuple((i, 1) for i in range(513)), head_dim=64,
                                  model_revision=self.layout.model_revision, hash_abi=self.layout.hash_abi),
                 "MANIFEST_LIMIT")):
            self.assert_code(code, m.inspect_checkpoint, self.root, layout,
                             allocation_granularity=4096, budget_bytes=1 << 30)

    def test_inspect_then_same_size_replacement_is_rejected_before_owner(self):
        plan = self.inspect()
        path = self.root / "weights.safetensors"
        replacement = self.root / "replacement"
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        native = FakeNative()
        self.assert_code("CHECKPOINT_OR_GRANULARITY_CHANGED", self.load, native, plan)
        self.assertEqual(native.owners, [])

    def test_same_size_payload_write_with_restored_mtime_is_rejected(self):
        plan = self.inspect()
        path = self.root / "weights.safetensors"
        original = path.stat()
        with path.open("r+b") as source:
            source.seek(-1, 2)
            value = source.read(1)
            source.seek(-1, 2)
            source.write(bytes((value[0] ^ 1,)))
            source.flush()
            os.fsync(source.fileno())
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        native = FakeNative()
        self.assert_code("CHECKPOINT_OR_GRANULARITY_CHANGED", self.load, native, plan)
        self.assertEqual(native.owners, [])

    def test_duplicate_checkpoint_header_keys_rejected_before_owner(self):
        plan = self.inspect()
        path = self.root / "weights.safetensors"
        raw = path.read_bytes()
        length = int.from_bytes(raw[:8], "little")
        header = b'{"__metadata__":{},"__metadata__":{},' + raw[8:8 + length].lstrip()[1:]
        header += b" " * (-len(header) % 8)
        path.write_bytes(len(header).to_bytes(8, "little") + header + raw[8 + length:])
        native = FakeNative()
        self.assert_code("DUPLICATE_JSON_KEY", self.load, native, plan)
        self.assertEqual(native.owners, [])

    def test_corrupt_range_or_truncation_never_allocates_owner(self):
        for corrupt_range in (False, True):
            self.write_checkpoint()
            plan = self.inspect()
            path = self.root / "weights.safetensors"
            raw = path.read_bytes()
            if corrupt_range:
                length = int.from_bytes(raw[:8], "little")
                header = json.loads(raw[8:8 + length])
                header["layers.1.engram.embed.weight"]["data_offsets"][1] = len(raw) + 1
                encoded = json.dumps(header).encode()
                encoded += b" " * (-len(encoded) % 8)
                path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + raw[8 + length:])
            else:
                path.write_bytes(raw[:-1])
            native = FakeNative()
            with self.assertRaises(Exception):
                self.load(native, plan)
            self.assertEqual(native.owners, [])

    def test_index_or_payload_changes_during_load_abort_before_export(self):
        native = FakeNative()
        def change_index(call):
            if call == 1:
                path = self.root / "model.safetensors.index.json"
                path.write_text(path.read_text() + " ")
        native.load_hook = change_index
        self.assert_code("CHECKPOINT_CHANGED", self.load, native)
        self.assertFalse(any(event[0] == "export" for event in native.events))
        self.assertEqual(native.close_attempts, [1, 0])

    def test_second_allocation_failure_rolls_back_first(self):
        native = FakeNative()
        native.fail_create = 1
        with self.assertRaisesRegex(RuntimeError, "create failure"):
            self.load(native)
        self.assertEqual(native.close_attempts, [0])

    def test_read_or_load_failure_rolls_back_without_export(self):
        for exception_type in (RuntimeError, KeyboardInterrupt):
            native = FakeNative()
            native.fail_load, native.load_error = 2, exception_type
            with self.assertRaises(exception_type):
                self.load(native)
            self.assertEqual(native.close_attempts, [0])
            self.assertFalse(any(event[0] == "export" for event in native.events))

    def test_file_open_failure_after_preflight_rolls_back_first_owner(self):
        import safetensors
        original = safetensors.safe_open
        plan = self.inspect()
        native = FakeNative()
        calls = 0
        def safe_open(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:  # Two preflight files; then first streaming read.
                raise RuntimeError("injected file read failure")
            return original(*args, **kwargs)
        with patch.object(safetensors, "safe_open", side_effect=safe_open):
            with self.assertRaisesRegex(RuntimeError, "file read failure"):
                self.load(native, plan)
        self.assertEqual(native.close_attempts, [0])
        self.assertFalse(any(event[0] == "export" for event in native.events))

    def test_second_export_failure_reverses_all_owners_without_publication(self):
        native = FakeNative()
        native.fail_export = 1
        with self.assertRaisesRegex(RuntimeError, "export failure"):
            self.load(native)
        self.assertEqual(native.close_attempts, [1, 0])
        self.assertTrue(all(owner.closed for owner in native.owners))

    def test_cleanup_failures_retain_every_owner_and_can_retry(self):
        native = FakeNative()
        native.fail_export, native.fail_close = 1, {0, 1}
        with self.assertRaises(m.CleanupPendingError) as error:
            self.load(native)
        pending = error.exception.resource
        self.assertEqual(native.close_attempts, [1, 0])
        self.assertEqual(len(pending._owners), 2)
        self.assert_code("GENERATION_NOT_READY", lambda: pending.private_handles)
        self.assertNotIn("private-capability", repr(pending))
        native.fail_close.clear()
        pending.close()
        self.assertTrue(all(owner.closed for owner in native.owners))


if __name__ == "__main__":
    unittest.main()
