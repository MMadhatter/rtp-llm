# SPDX-License-Identifier: Apache-2.0
"""Host NUMA and GPU backing placement and rollback; CPU only."""

import base64
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from uuid import uuid4

from rtp_llm.models_py.modules.dsv41.engram_distributed import (
    backing,
    placement,
    publish,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.config import (
    ShardConfig,
    memory_kind_from_environ,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.manifest import parse_manifest
from rtp_llm.models_py.modules.dsv41.engram_distributed.mapping import (
    ShardedLayerMapping,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.placement import (
    check_gpu_capacity,
    segment_rows,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.shard import (
    mapping_requests,
    plan_shards,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.storage import (
    CleanupPendingError,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.test.test_engram_manifest import (
    build,
    descriptors,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.test.test_engram_sharding import (
    FakeOps,
)


def split_documents(p=2):
    docs = descriptors(p)
    for doc in docs:
        doc["schema_version"] = 3
        for name, old in list(doc["tensors"].items()):
            rows, columns = old["shape"]
            dim = columns if name.endswith("weight") else columns * 32
            plan = plan_shards(rows, dim, p, granularity=4096)
            pieces = []
            for node, begin, end, padded in placement.segment_rows(
                plan, doc["shard"], (0, 1)
            ):
                piece = dict(
                    old,
                    row_begin=begin,
                    row_end=end,
                    logical_bytes=(end - begin) * columns,
                    allocation_bytes=padded * columns,
                    numa_node=node,
                    allocation_id=str(uuid4()),
                    handle=base64.b64encode(uuid4().bytes).decode(),
                )
                pieces.append(piece)
            doc["tensors"][name] = dict(shape=old["shape"], segments=pieces)
    return docs


class PlacementTest(unittest.TestCase):
    def test_publisher_loads_each_row_once_and_rolls_back_partial_segments(self):
        import sys
        from contextlib import ExitStack, nullcontext
        from types import SimpleNamespace as NS
        from unittest.mock import patch

        from rtp_llm.models_py.modules.dsv41.engram_distributed import backing, publish

        layout = NS(layer_ids=[1], num_embeddings=[32768], head_dim=64)
        tensors = [
            NS(
                name=f"layers.1.engram.embed.{role}",
                shape=(32768, columns),
                relative_path="weights",
            )
            for role, columns in (("weight", 64), ("scale", 2))
        ]
        checkpoint = NS(
            inspect_checkpoint=lambda *args: NS(
                layers=[NS(layer_id=1, tensors=tensors)],
                files=[NS(relative_path="weights")],
                index=None,
            ),
            _cache_page_size=lambda _: None,
            _discard_consumed_pages=None,
            _open_tensor_source=None,
            _same_snapshot=lambda *args, **kwargs: None,
        )
        for fail in (False, True):
            owners, loaded = [], []

            class Owner:
                def __init__(self, size, numa, device):
                    self.size, self.numa, self.closed = size, numa, False
                    owners.append(self)

                def export_handle(self):
                    return uuid4().bytes

                def close(self):
                    self.closed = True

            native = NS(
                Owner=Owner,
                host_numa_granularity=lambda *args: 4096,
                max_allocation_bytes=1 << 30,
                max_load_chunk_bytes=1 << 20,
            )

            def load(owner, root, tensor, snapshot, begin, end, columns, **kwargs):
                loaded.append((tensor.name, begin, end, owner.numa))
                if fail and len(loaded) == 3:
                    raise OSError("synthetic segment load failure")
                return "sha256:" + "a" * 64

            with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
                Path(root, "config.json").write_text("{}")
                stack.enter_context(
                    patch.dict(
                        sys.modules,
                        {
                            "torch": NS(cuda=NS(device=lambda _: nullcontext())),
                            "rtp_llm.models_py.modules.dsv41.engram": NS(
                                EngramLayout=lambda _: layout,
                                NgramHashState=lambda *args: None,
                            ),
                            "rtp_llm.models_py.modules.dsv41.engram_distributed.bootstrap": NS(
                                _load_native=lambda: native,
                                hash_state_digest=lambda _: "sha256:" + "b" * 64,
                            ),
                            "rtp_llm.models_py.modules.dsv41.engram_distributed.checkpoint": checkpoint,
                        },
                    )
                )
                stack.enter_context(
                    patch.object(backing, "resolve_numas", return_value=(0, 1))
                )
                stack.enter_context(patch.object(publish, "_reclaim_legacy_tables"))
                stack.enter_context(
                    patch.object(publish, "load_rows", side_effect=load)
                )
                generation = str(uuid4())
                if fail:
                    with self.assertRaises(OSError):
                        publish.publish_shard(
                            root, "revision", "host-0", generation, 0, 2, numa=(0, 1)
                        )
                    self.assertTrue(all(owner.closed for owner in owners))
                    self.assertNotIn(generation, publish._KEEPERS)
                else:
                    doc = publish.publish_shard(
                        root, "revision", "host-0", generation, 0, 2, numa=(0, 1)
                    )
                    self.assertEqual(doc["schema_version"], 3)
                    self.assertEqual(sum(owner.size for owner in owners), 16384 * 66)
                    for name in (t.name for t in tensors):
                        self.assertEqual(
                            [
                                (begin, end, numa)
                                for tensor, begin, end, numa in loaded
                                if tensor == name
                            ],
                            [(0, 8192, 0), (8192, 16384, 1)],
                        )
                    for owner in publish._KEEPERS.pop(generation):
                        owner.close()

    def test_partition_has_no_replicas_or_extra_padding(self):
        for p in (2, 3, 18):
            for dim in (32, 64, 256):
                plan = plan_shards(p * 65536 - 7, dim, p, granularity=4096)
                pieces = [
                    part
                    for i in range(p)
                    for part in placement.segment_rows(plan, i, (0, 1))
                ]
                self.assertEqual(sum(part[3] for part in pieces), plan.padded_rows)
                cursor = 0
                for node, begin, end, padded in pieces:
                    self.assertEqual(begin, cursor)
                    self.assertLess(begin, end)
                    self.assertEqual(padded * (dim // 32) % 4096, 0)
                    cursor = end
                self.assertEqual(cursor, plan.total_rows)
                requests = mapping_requests(
                    plan, 0, 0, segment_rows=[part[3] for part in pieces]
                )
                self.assertEqual(
                    sum(r.size for r in requests), plan.weight_span + plan.scale_span
                )

    def test_shared_parser_and_cpuset_guard(self):
        self.assertEqual(placement.parse_numa(None), 0)
        self.assertEqual(placement.parse_numa("0-1"), (0, 1))
        self.assertEqual(placement.parse_numa("1,0"), (0, 1))
        for bad in ("", "-1", "65536", "0-9999999999", "1-0", "２", "0-8"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                placement.parse_numa(bad)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "has_memory").write_text("0-1\n")
            (root / "status").write_text("Mems_allowed_list:\t0-1\n")
            for n in (0, 1):
                d = root / f"node{n}"
                d.mkdir()
                (d / "meminfo").write_text(f"Node {n} MemTotal: 1048576 kB\n")
                (d / "cpulist").write_text(str(n))
            args = dict(sysfs=root, status=root / "status")
            self.assertEqual(
                placement.resolve_numas(
                    "auto", discover=lambda: {0: 1, 1: 0, 2: 1}, **args
                ),
                (0, 1),
            )
            (root / "status").write_text("Mems_allowed_list:\t0\n")
            with self.assertRaisesRegex(ValueError, "OUTSIDE_CPUSET"):
                placement.resolve_numas((0, 1), **args)

    def test_segmented_wire_counts_storage_hosts_not_allocations(self):
        for p in (2, 3):
            manifest = parse_manifest(json.dumps(build(split_documents(p))).encode())
            self.assertEqual(manifest.schema_version, 2)
            ShardConfig(p).validate_manifest(manifest)
            for tensor in manifest.tensors:
                self.assertEqual(len(tensor.shards), p * 2)
                self.assertEqual(len(placement.owner_groups(tensor)), p)

    def test_startup_package_carries_every_physical_capability(self):
        from rtp_llm.models_py.modules.dsv41.engram_distributed.startup import (
            assemble_package,
            verify_package,
        )

        docs = split_documents()
        group = dict(
            generation=docs[0]["generation"],
            model_revision=docs[0]["model_revision"],
            members=[
                dict(shards=[dict(index=i, host_id=f"host-{i}")]) for i in range(2)
            ],
        )
        package = assemble_package(
            group, docs, "unused", samples=lambda _, m: {t.name: {} for t in m.tensors}
        )
        manifest = verify_package(group, package)
        self.assertEqual(len(package["capabilities"]), 16)
        self.assertEqual(len(placement.owner_groups(manifest.tensors[0])), 2)
        package["capabilities"].pop(next(iter(package["capabilities"])))
        with self.assertRaisesRegex(RuntimeError, "CAPABILITIES_MISMATCH"):
            verify_package(group, package)

    def test_malformed_partition_rejected_before_native_import(self):
        for kind in ("gap", "extra", "node", "scale", "shape", "mixed"):
            docs = split_documents()
            name = "layers.1.engram.embed.weight"
            part = docs[0]["tensors"][name]["segments"][0]
            if kind == "gap":
                part["row_end"] -= 1
            if kind == "extra":
                part["allocation_bytes"] += 4096
            if kind == "node":
                part["numa_node"] = 1
            if kind == "scale":
                part["numa_node"] = 2
            if kind == "shape":
                docs[1]["tensors"][name]["shape"][0] += 1
            if kind == "mixed":
                docs[1] = descriptors()[1]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                build(docs)

    def test_variable_physical_segments_map_and_rollback(self):
        plan = plan_shards(6 * 262144 - 7, 256, 2)
        rows = [p[3] for i in range(2) for p in placement.segment_rows(plan, i, (0, 1))]
        handles = [bytes([i + 1]) * 8 for i in range(4)]
        ops = FakeOps()
        mapping = ShardedLayerMapping(
            plan, handles, handles, 0, ops=ops, segment_rows=rows
        ).attach()
        self.assertEqual(len([c for c in ops.calls if c[0] == "map"]), 8)
        mapping.close()
        bad_ops = FakeOps(fail_on="map")
        with self.assertRaises(RuntimeError):
            ShardedLayerMapping(
                plan, handles, handles, 0, ops=bad_ops, segment_rows=rows
            ).attach()
        self.assertEqual(len([c for c in bad_ops.calls if c[0] == "free"]), 2)
        for bad in ([1], ["bad"], [True], [0]):
            ops = FakeOps()
            with self.assertRaises(ValueError):
                ShardedLayerMapping(
                    plan, handles, handles, 0, ops=ops, segment_rows=bad
                )
            self.assertEqual(ops.calls, [])


def gpu_documents():
    docs = split_documents()
    for doc in docs:
        locations = {d: str(uuid4()) for d in range(4)}
        doc["memory_kind"] = "gpu"
        for name, tensor in doc["tensors"].items():
            rows, columns = 131072 - 7, tensor["shape"][1]
            dim = columns if name.endswith("weight") else columns * 32
            plan = plan_shards(rows, dim, len(docs), granularity=4096)
            template = tensor["segments"][0]
            pieces = []
            for device, begin, end, padded in segment_rows(
                plan, doc["shard"], tuple(locations)
            ):
                piece = dict(
                    template,
                    shape=[rows, columns],
                    row_begin=begin,
                    row_end=end,
                    logical_bytes=(end - begin) * columns,
                    allocation_bytes=padded * columns,
                    allocation_id=str(uuid4()),
                    handle=base64.b64encode(uuid4().bytes).decode(),
                    device_uuid=locations[device],
                )
                piece.pop("numa_node")
                pieces.append(piece)
            doc["tensors"][name] = dict(shape=[rows, columns], segments=pieces)
    return docs


class GpuBackingTest(unittest.TestCase):
    def test_mode_is_explicit_and_defaults_to_host(self):
        self.assertEqual(memory_kind_from_environ({}), "host")
        self.assertEqual(memory_kind_from_environ({"RTP_ENGRAM_MEMORY": "gpu"}), "gpu")
        for value in ("", "GPU", "cuda", None):
            with self.assertRaises(ValueError):
                memory_kind_from_environ({"RTP_ENGRAM_MEMORY": value})

    def test_wire_keeps_logical_host_count(self):
        manifest = parse_manifest(json.dumps(build(gpu_documents())).encode())
        self.assertEqual(manifest.backend, "fabric_gpu")
        self.assertEqual(manifest.schema_version, 2)
        ShardConfig(2).validate_manifest(manifest)
        self.assertEqual(len(manifest.tensors[0].shards), 8)
        self.assertTrue(
            all(
                p.device_uuid and p.numa_node is None
                for t in manifest.tensors
                for p in t.shards
            )
        )

    def test_inconsistent_backing_rejected(self):
        for fault in (
            "mixed",
            "host_field",
            "gpu_on_other_host",
            "duplicate_gpu",
            "scale_gpu",
            "bad_uuid",
        ):
            docs = gpu_documents()
            weight = "layers.1.engram.embed.weight"
            first = docs[0]["tensors"][weight]["segments"][0]
            if fault == "mixed":
                docs[1].pop("memory_kind")
            elif fault == "host_field":
                first["numa_node"] = 0
            elif fault == "gpu_on_other_host":
                docs[1]["tensors"][weight]["segments"][0]["device_uuid"] = first[
                    "device_uuid"
                ]
            elif fault == "duplicate_gpu":
                docs[0]["tensors"][weight]["segments"][1]["device_uuid"] = first[
                    "device_uuid"
                ]
            elif fault == "scale_gpu":
                docs[0]["tensors"]["layers.1.engram.embed.scale"]["segments"][0][
                    "device_uuid"
                ] = str(uuid4())
            else:
                first["device_uuid"] = "cuda:0"
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                build(docs)

    def test_capacity_checks_every_selected_gpu(self):
        query = NS(mem_get_info=lambda d: ((100 if d == 0 else 20), 200))
        with patch.dict(sys.modules, {"torch": NS(cuda=query)}):
            check_gpu_capacity({0: 60, 1: 10}, 10)
            with self.assertRaisesRegex(ValueError, "device=1"):
                check_gpu_capacity({0: 60, 1: 11}, 10)
            for bad in (0, True, -1, 2**63):
                with self.assertRaises(ValueError):
                    check_gpu_capacity({0: 1}, bad)

    def test_publisher_distributes_without_replicas_and_rolls_back(self):
        for failure in (None, "capacity", "load"):
            owners, loads = [], []
            tensors = [
                NS(
                    name=f"layers.1.engram.embed.{role}",
                    shape=(65536, width),
                    relative_path="weights",
                )
                for role, width in (("weight", 64), ("scale", 2))
            ]
            checkpoint = NS(
                inspect_checkpoint=lambda *_: NS(
                    layers=[NS(layer_id=1, tensors=tensors)],
                    files=[NS(relative_path="weights")],
                    index=None,
                ),
                _cache_page_size=lambda _: None,
                _discard_consumed_pages=None,
                _open_tensor_source=None,
                _same_snapshot=lambda *_, **kw: None,
            )

            class Owner:
                def __init__(self, size, numa, device, *, memory_kind):
                    self.size, self.numa, self.device, self.kind, self.closed = (
                        size,
                        numa,
                        device,
                        memory_kind,
                        False,
                    )
                    owners.append(self)

                def export_handle(self):
                    return uuid4().bytes

                def close(self):
                    self.closed = True

            native = NS(
                Owner=Owner,
                device_granularity=lambda _: 4096,
                max_allocation_bytes=1 << 30,
                max_load_chunk_bytes=1 << 20,
            )

            def load(owner, root, tensor, snapshot, begin, end, columns, **kwargs):
                loads.append((tensor.name, begin, end, owner.device))
                if failure == "load" and len(loads) == 5:
                    raise OSError("synthetic GPU load failure")
                return "sha256:" + "a" * 64

            def capacity(required, reserve):
                self.assertEqual(owners, [])
                self.assertEqual(required, {i: 8192 * 66 for i in range(4)})
                if failure == "capacity":
                    raise ValueError("synthetic capacity failure")

            with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
                Path(root, "config.json").write_text("{}")
                stack.enter_context(
                    patch.dict(
                        sys.modules,
                        {
                            "torch": NS(cuda=NS(device=lambda _: nullcontext())),
                            "rtp_llm.models_py.modules.dsv41.engram": NS(
                                EngramLayout=lambda _: NS(
                                    layer_ids=[1], num_embeddings=[65536], head_dim=64
                                ),
                                NgramHashState=lambda *_: None,
                            ),
                            "rtp_llm.models_py.modules.dsv41.engram_distributed.bootstrap": NS(
                                _load_native=lambda: native,
                                hash_state_digest=lambda _: "sha256:" + "b" * 64,
                            ),
                            "rtp_llm.models_py.modules.dsv41.engram_distributed.checkpoint": checkpoint,
                        },
                    )
                )
                stack.enter_context(
                    patch.object(
                        backing,
                        "gpu_locations",
                        return_value={d: str(uuid4()) for d in range(4)},
                    )
                )
                stack.enter_context(
                    patch.object(backing, "check_gpu_capacity", side_effect=capacity)
                )
                stack.enter_context(patch.object(publish, "_reclaim_legacy_tables"))
                stack.enter_context(
                    patch.object(publish, "load_rows", side_effect=load)
                )
                generation = str(uuid4())

                def invoke():
                    return publish.publish_shard(
                        root,
                        "revision",
                        "host-0",
                        generation,
                        0,
                        2,
                        memory_kind="gpu",
                        gpu_devices=(0, 1, 2, 3),
                    )

                if failure:
                    with self.assertRaises((ValueError, OSError)):
                        invoke()
                    self.assertTrue(all(o.closed for o in owners))
                    self.assertNotIn(generation, publish._KEEPERS)
                else:
                    doc = invoke()
                    self.assertEqual(doc["memory_kind"], "gpu")
                    self.assertEqual(sum(o.size for o in owners), 32768 * 66)
                    self.assertTrue(
                        all(o.numa == -1 and o.kind == "gpu" for o in owners)
                    )
                    for t in tensors:
                        self.assertEqual(
                            [(a, b, d) for name, a, b, d in loads if name == t.name],
                            [(8192 * d, 8192 * (d + 1), d) for d in range(4)],
                        )
                    for owner in publish._KEEPERS.pop(generation):
                        owner.close()

    def test_rejected_import_retained_when_release_needs_retry(self):
        class Ops(FakeOps):
            fail_release = True

            def validate_allocation(self, handle):
                raise ValueError("wrong backing kind")

            def release(self, handle):
                super().release(handle)
                if self.fail_release:
                    raise RuntimeError("retry release")

        ops = Ops()
        mapping = ShardedLayerMapping(
            plan_shards(1048576, 256, 2),
            [b"a" * 8] * 2,
            [b"b" * 8] * 2,
            0,
            ops=ops,
            memory_kind="gpu",
        )
        with self.assertRaises(CleanupPendingError):
            mapping.attach()
        self.assertEqual(mapping.info()["imported_handles"], 1)
        self.assertFalse(any(call[0] == "map" for call in ops.calls))
        ops.fail_release = False
        mapping.close()
        self.assertEqual(mapping.info()["imported_handles"], 0)
        self.assertEqual(mapping.info()["state"], "CLOSED")


if __name__ == "__main__":
    unittest.main()
