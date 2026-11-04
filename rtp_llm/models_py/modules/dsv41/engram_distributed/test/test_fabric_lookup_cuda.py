# SPDX-License-Identifier: Apache-2.0
"""Opt-in real CUDA Fabric lookup and lifecycle regressions."""

import base64
import gc
import hashlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

_FAILED_RESOURCES = []


@unittest.skipUnless(
    os.environ.get("ENGRAM_RUN_CUDA_TESTS") == "1",
    "explicit CUDA test opt-in required",
)
class FabricEmbeddingLookupTest(unittest.TestCase):
    def test_real_shards_lookup_graph_and_alias_lifetime(self):
        for memory_kind in ("host", "gpu"):
            with self.subTest(memory_kind=memory_kind):
                self._exercise(memory_kind)

    def _exercise(self, memory_kind):
        import torch

        from rtp_llm.models_py.modules.dsv41._engram_decode_prefetch import (
            DecodeEngramPrefetch,
        )
        from rtp_llm.models_py.modules.dsv41.engram import HostEngramEmbedding
        from rtp_llm.models_py.modules.dsv41.engram_distributed.bootstrap import (
            _load_native,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.descriptors import (
            build_manifest,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.manifest import (
            ExpectedLayout,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.mapping import (
            CudaDriverOps,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.shard import plan_shards
        from rtp_llm.models_py.modules.dsv41.engram_distributed.storage import (
            FabricEngramStorage,
            StorageBusyError,
        )

        class CountingOps(CudaDriverOps):
            def __init__(self, device):
                super().__init__(device, memory_kind=memory_kind)
                self.imports = 0
                self.releases = 0

            def import_handle(self, raw):
                handle = super().import_handle(raw)
                self.imports += 1
                return handle

            def release(self, handle):
                super().release(handle)
                self.releases += 1

        device = int(os.environ.get("ENGRAM_TEST_DEVICE", "0"))
        numa = int(os.environ.get("ENGRAM_TEST_NUMA", "0"))
        torch.cuda.set_device(device)
        target = torch.device("cuda", device)
        native = _load_native()
        baseline = native.resource_counters()
        from rtp_llm.models_py.modules.dsv41.engram_distributed.placement import (
            gpu_locations,
            segment_rows,
        )

        gpu = memory_kind == "gpu"
        segment_count = 2 if gpu and torch.cuda.device_count() >= 4 else 1
        locations = gpu_locations(tuple(range(2 * segment_count))) if gpu else {}
        granularity = (
            native.device_granularity(device)
            if gpu
            else native.host_numa_granularity(numa, device)
        )
        plan = plan_shards(
            2 * segment_count * (granularity // 8) - 7, 256, 2, granularity=granularity
        )
        if 2 * plan.shard_bytes > 512 << 20:
            self.fail("test fixture exceeds its 512 MiB backing bound")

        generation = str(uuid4())
        revision = "fabric-lookup-regression-v1"
        hash_abi = "sha256:" + hashlib.sha256(revision.encode()).hexdigest()
        owners, documents, capabilities = [], [], {}
        storage = embedding = graph = alias = model = metadata = None
        completed = False
        try:
            for shard in range(2):
                row_parts = segment_rows(
                    plan,
                    shard,
                    (
                        tuple(range(shard * segment_count, (shard + 1) * segment_count))
                        if gpu
                        else (numa,)
                    ),
                )
                tensors = {}
                for role, width, size, value in (
                    ("weight", 256, plan.weight_bytes, (0x38, 0x40)[shard]),
                    ("scale", 8, plan.scale_bytes, 127 + shard),
                ):
                    pieces = []
                    for location, begin, end, padded in row_parts:
                        owner = native.Owner(
                            padded * width,
                            -1 if gpu else numa,
                            location if gpu else device,
                            **({"memory_kind": "gpu"} if gpu else {}),
                        )
                        owners.append(owner)
                        logical_bytes = (end - begin) * width
                        digest = hashlib.sha256()
                        for offset in range(0, logical_bytes, 8 << 20):
                            payload = bytes([value]) * min(
                                8 << 20, logical_bytes - offset
                            )
                            owner.load(payload, offset)
                            digest.update(payload)
                        allocation_id = str(uuid4())
                        handle = owner.export_handle()
                        capabilities[allocation_id] = handle
                        piece = dict(
                            shape=[plan.total_rows, width],
                            row_begin=begin,
                            row_end=end,
                            logical_bytes=logical_bytes,
                            allocation_bytes=padded * width,
                            allocation_granularity=granularity,
                            allocation_id=allocation_id,
                            sha256="sha256:" + digest.hexdigest(),
                            handle=base64.b64encode(handle).decode("ascii"),
                        )
                        if gpu:
                            piece["device_uuid"] = locations[location]
                        pieces.append(piece)
                    tensors[f"layers.1.engram.embed.{role}"] = (
                        dict(shape=[plan.total_rows, width], segments=pieces)
                        if gpu
                        else pieces[0]
                    )
                documents.append(
                    dict(
                        schema_version=3 if gpu else 2,
                        **({"memory_kind": "gpu"} if gpu else {}),
                        shard=shard,
                        shards=2,
                        generation=generation,
                        model_revision=revision,
                        hash_abi=hash_abi,
                        host_id=f"fixture-host-{shard}",
                        owner_id=f"fixture-host-{shard}",
                        owner_incarnation=str(uuid4()),
                        tensors=tensors,
                    )
                )

            manifest = build_manifest(documents, 2, generation, revision, hash_abi)
            ops = CountingOps(device)
            storage = FabricEngramStorage.attach(
                json.dumps(manifest).encode(),
                expected_layout=ExpectedLayout(
                    layers=((1, plan.total_rows),),
                    head_dim=256,
                    model_revision=revision,
                    hash_abi=hash_abi,
                    backend="fabric_gpu" if gpu else "fabric_egm",
                ),
                dataset_id=manifest["dataset_id"],
                generation=generation,
                handles=capabilities,
                device=device,
                native=native,
                shard_ops=ops,
            )

            boundary = plan.row_range(1)[0]
            sample_rows = sorted(
                {0, boundary - 1, boundary, plan.total_rows - 1}
                | {
                    row
                    for shard in range(2)
                    for _, begin, end, _ in segment_rows(
                        plan, shard, tuple(range(segment_count))
                    )
                    for row in (begin, end - 1)
                }
            )

            def verify(views, _validated):
                ids = torch.tensor(sample_rows, dtype=torch.int64, device=target)
                for role, expected in (
                    (
                        "weight",
                        [0x38 if row < boundary else 0x40 for row in sample_rows],
                    ),
                    ("scale", [127 if row < boundary else 128 for row in sample_rows]),
                ):
                    actual = views[f"layers.1.engram.embed.{role}"].index_select(0, ids)
                    expected = torch.tensor(expected, dtype=torch.uint8)[:, None]
                    self.assertTrue(
                        torch.equal(actual.cpu(), expected.expand(actual.shape))
                    )
                torch.cuda.synchronize(device)
                return True

            storage.verify(verify)

            def reference(host_ids):
                ids = torch.tensor(host_ids, dtype=torch.int64)
                values = torch.where(ids < boundary, 1.0, 4.0)
                values.masked_fill_(ids < 0, 0)
                return values.to(torch.bfloat16)[..., None].expand(*ids.shape, 256)

            patterns = [
                [
                    [0, boundary - 1, boundary, -1],
                    [plan.total_rows - 1, 1, -1, boundary],
                ],
                [
                    [boundary, 0, -1, boundary - 1],
                    [0, boundary, plan.total_rows - 1, -1],
                ],
            ]
            # Exercise mapped scales directly, then the default Blackwell GPU
            # scale copy. The latter must still expose the BF16 __call__ API.
            for optimized in (False, True):
                with patch.dict(
                    os.environ,
                    {
                        "DSV41_ENGRAM_DEVICE_SCALES": str(int(optimized)),
                        "DSV41_ENGRAM_DIRECT_FP8": str(int(optimized)),
                        # Sync Graph ownership uses the production metadata path;
                        # the attention schedule itself needs the full-model E2E.
                        "DSV41_DECODE_ENGRAM_PREFETCH": "0",
                    },
                ):
                    embedding = HostEngramEmbedding.from_storage(storage, 1)
                    self.assertEqual(embedding._quantized_lookup_supported, optimized)
                    for dtype in (torch.int32, torch.int64):
                        ids = torch.tensor(patterns[0], dtype=dtype, device=target)
                        output = embedding(ids, target)
                        self.assertIsInstance(output, torch.Tensor)
                        self.assertEqual(output.dtype, torch.bfloat16)
                        self.assertTrue(
                            torch.equal(output.cpu(), reference(patterns[0]))
                        )

                    # Minimal model container; table snapshots and capture
                    # authorization are real production objects, not mocks.
                    class Model:
                        pass

                    model = Model()
                    model.layers = [
                        SimpleNamespace(engram=SimpleNamespace(embed_tokens=embedding))
                    ]
                    metadata = SimpleNamespace()
                    ids = torch.tensor(patterns[0], dtype=torch.int64, device=target)
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        output = embedding(ids, target)
                    stream.synchronize()
                    storage._pin_graph()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        plan_owner = DecodeEngramPrefetch(model, metadata)
                        try:
                            output = embedding(ids, target)
                        finally:
                            plan_owner.close()

                    with self.assertRaises(StorageBusyError):
                        storage.close()
                    for host_ids in patterns * 2:
                        ids.copy_(
                            torch.tensor(host_ids, dtype=torch.int64, device=target)
                        )
                        graph.replay()
                        self.assertTrue(torch.equal(output.cpu(), reference(host_ids)))

                    torch.cuda.synchronize(device)
                    graph.reset()
                    graph = None
                    storage._unpin_graph()
                    metadata = model = plan_owner = None
                    embedding.release_external_views()
                    embedding = None
                    gc.collect()

            alias = storage.table_views(1)[0][1:2]
            gc.collect()
            with self.assertRaises(StorageBusyError):
                storage.close()
            alias = None
            gc.collect()
            storage.close()
            self.assertEqual(storage.info()["state"], "CLOSED")
            self.assertEqual(
                (ops.imports, ops.releases), (4 * segment_count, 4 * segment_count)
            )
            for owner in reversed(owners):
                owner.close()
            self.assertEqual(native.resource_counters(), baseline)
            completed = True
        finally:
            if not completed:
                _FAILED_RESOURCES.append(
                    (owners, storage, embedding, graph, alias, model, metadata)
                )


@unittest.skipUnless(
    os.environ.get("ENGRAM_RUN_CUDA_TESTS") == "1", "explicit CUDA test opt-in required"
)
class ShardedCudaLifecycleTest(unittest.TestCase):
    def test_real_aliases_graph_and_repeated_consumers(self):
        import torch

        from rtp_llm.models_py.modules.dsv41.engram_distributed.bootstrap import (
            _load_native,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.mapping import (
            CudaDriverOps,
            ShardedLayerMapping,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.shard import plan_shards
        from rtp_llm.models_py.modules.dsv41.engram_distributed.storage import (
            StorageBusyError,
        )

        device = int(os.environ.get("ENGRAM_TEST_DEVICE", "0"))
        peer_device = int(os.environ.get("ENGRAM_TEST_PEER_DEVICE", str(device)))
        numa = int(os.environ.get("ENGRAM_TEST_NUMA", "0"))
        torch.cuda.set_device(device)
        native = _load_native()
        baseline = native.resource_counters()
        granularity = native.host_numa_granularity(numa, device)

        class CountingOps(CudaDriverOps):
            imports = releases = 0

            def import_handle(self, raw):
                result = super().import_handle(raw)
                self.imports += 1
                return result

            def release(self, handle):
                super().release(handle)
                self.releases += 1

        for p in (2, 3):
            plan = plan_shards(
                p * (granularity // 8) - 7, 256, p, granularity=granularity
            )
            owners, weights, scales = [], [], []
            try:
                for shard in range(p):
                    for size, value, handles in (
                        (plan.weight_bytes, shard + 1, weights),
                        (plan.scale_bytes, shard + 101, scales),
                    ):
                        owner = native.Owner(size, numa, device)
                        owners.append(owner)
                        for offset in range(0, size, 8 << 20):
                            owner.load(
                                bytes([value]) * min(8 << 20, size - offset), offset
                            )
                        handles.append(owner.export_handle())
                devices = [device, peer_device]
                ops = [CountingOps(target) for target in devices]
                mappings = [
                    ShardedLayerMapping(plan, weights, scales, target, ops=op).attach()
                    for target, op in zip(devices, ops)
                ]
                for target, mapping in zip(devices, mappings):
                    view = mapping.weight_view()
                    self.assertEqual(view.data_ptr(), mapping._weight_ptr)
                    self.assertEqual(view.device.index, target)
                    alias = view[1:2]
                    del view
                    gc.collect()
                    self.assertEqual(mapping.info()["active_views"], 1)
                    with self.assertRaises(StorageBusyError):
                        mapping.close()
                    del alias
                    gc.collect()
                    self.assertEqual(mapping.info()["active_views"], 0)

                first, second = mappings
                first.close()
                # Closing one independent mapping must not affect the other.
                torch.cuda.set_device(peer_device)
                w, s = second.weight_view(), second.scale_view()
                rows = torch.tensor(
                    [plan.row_range(i)[0] for i in range(p)], device=peer_device
                )
                expected_w = list(range(1, p + 1))
                expected_s = list(range(101, p + 101))
                self.assertEqual(
                    w.index_select(0, rows)[:, 0].cpu().tolist(), expected_w
                )
                self.assertEqual(
                    s.index_select(0, rows)[:, 0].cpu().tolist(), expected_s
                )
                stream = torch.cuda.Stream(device=peer_device)
                stream.wait_stream(torch.cuda.current_stream(peer_device))
                with torch.cuda.stream(stream):
                    output = w.index_select(0, rows)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = w.index_select(0, rows)
                for _ in range(3):
                    graph.replay()
                    self.assertEqual(output[:, 0].cpu().tolist(), expected_w)
                torch.cuda.synchronize(peer_device)
                graph.reset()
                del graph, w, s, output, rows
                gc.collect()
                second.close()
                self.assertTrue(all(op.imports == op.releases == 2 * p for op in ops))
            finally:
                # Failures must terminate this isolated probe rather than force
                # releasing owners while an unknown Graph/view may still exist.
                if not all(m.state == "CLOSED" for m in locals().get("mappings", [])):
                    raise RuntimeError(
                        "probe mappings remain live; terminate probe process"
                    )
                for owner in reversed(owners):
                    owner.close()
            self.assertEqual(native.resource_counters(), baseline)
            torch.cuda.set_device(device)


if __name__ == "__main__":
    unittest.main()
