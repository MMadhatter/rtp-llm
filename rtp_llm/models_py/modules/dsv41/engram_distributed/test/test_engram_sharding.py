# SPDX-License-Identifier: Apache-2.0
"""Shard geometry and mapping ownership; CPU only."""

import hashlib
import unittest
from enum import IntEnum
from unittest import mock

from rtp_llm.models_py.modules.dsv41.engram_distributed import mapping as storage
from rtp_llm.models_py.modules.dsv41.engram_distributed.mapping import (
    ShardedLayerMapping,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.shard import (
    DEFAULT_GRANULARITY,
    ShardPlan,
    local_rows,
    mapping_requests,
    plan_shards,
    split_rows,
)

REAL_ROWS = (384006168, 384016682)
DIM = 256
ALIGN_ROWS = 262144


class PlanShardsTest(unittest.TestCase):
    def test_real_geometry_is_mappable(self):
        for total in REAL_ROWS:
            plan = plan_shards(total, DIM, 4)
            self.assertEqual(plan.shard_count, 4)
            self.assertEqual(plan.shard_rows % ALIGN_ROWS, 0)
            self.assertGreaterEqual(plan.padded_rows, total)
            self.assertEqual(plan.weight_bytes % DEFAULT_GRANULARITY, 0)
            self.assertEqual(plan.scale_bytes % DEFAULT_GRANULARITY, 0)
            self.assertEqual(plan.shard_bytes % DEFAULT_GRANULARITY, 0)

    def test_padding_is_bounded(self):
        for total in REAL_ROWS:
            for count in (1, 2, 4, 8):
                plan = plan_shards(total, DIM, count)
                overhead = plan.padded_rows - total
                self.assertLess(overhead, count * ALIGN_ROWS)
                self.assertLess(overhead / total, 5e-3)

    def test_single_shard(self):
        plan = plan_shards(1000, DIM, 1)
        self.assertEqual(plan.shard_count, 1)
        self.assertEqual(plan.row_range(0), (0, 1000))
        self.assertEqual(plan.weight_offset(0), 0)
        self.assertEqual(plan.scale_offset(0), 0)

    def test_rows_cover_table_exactly_once(self):
        total = 1000
        plan = plan_shards(total, DIM, 4, granularity=8)
        ranges = split_rows(total, plan)
        covered = []
        for start, end in ranges:
            covered.extend(range(start, end))
        self.assertEqual(covered, list(range(total)))

    def test_row_to_shard_boundaries(self):
        plan = plan_shards(REAL_ROWS[0], DIM, 4)
        step = plan.shard_rows
        self.assertEqual(plan.shard_of_row(0), 0)
        self.assertEqual(plan.shard_of_row(step - 1), 0)
        self.assertEqual(plan.shard_of_row(step), 1)
        self.assertEqual(plan.shard_of_row(step * 2), 2)
        self.assertEqual(plan.shard_of_row(step * 3), 3)
        with self.assertRaises(IndexError):
            plan.shard_of_row(plan.padded_rows)
        with self.assertRaises(IndexError):
            plan.shard_of_row(-1)

    def test_va_offsets_are_granularity_aligned(self):
        plan = plan_shards(REAL_ROWS[0], DIM, 8)
        for shard in range(plan.shard_count):
            self.assertEqual(plan.weight_offset(shard) % DEFAULT_GRANULARITY, 0)
            self.assertEqual(plan.scale_offset(shard) % DEFAULT_GRANULARITY, 0)

    def test_local_rows(self):
        plan = plan_shards(REAL_ROWS[0], DIM, 4)
        self.assertEqual(local_rows(plan, -1), (0, 0))
        self.assertEqual(local_rows(plan, 0), (0, plan.shard_rows))
        start, end = local_rows(plan, 3)
        self.assertEqual(start, plan.shard_rows * 3)
        self.assertEqual(end, REAL_ROWS[0])

    def test_last_shard_holds_padding(self):
        total = 1001
        plan = plan_shards(total, DIM, 4, granularity=8)
        start, end = plan.row_range(3)
        self.assertLessEqual(end, total)
        self.assertLess(end - start, plan.shard_rows)

    def test_rejects_bad_geometry(self):
        with self.assertRaises(ValueError):
            plan_shards(0, DIM, 4)
        with self.assertRaises(ValueError):
            plan_shards(100, 250, 4)  # dim not a multiple of block
        with self.assertRaises(ValueError):
            plan_shards(100, DIM, 0)
        with self.assertRaises(ValueError):
            plan_shards(100, DIM, 4, granularity=3)

    def test_plan_is_hashable_and_immutable(self):
        plan = plan_shards(1000, DIM, 4, granularity=8)
        self.assertIsInstance(plan, ShardPlan)
        self.assertEqual(hash(plan), hash(plan_shards(1000, DIM, 4, granularity=8)))
        with self.assertRaises(Exception):
            plan.shard_rows = 1

    def test_non_power_of_two_host_counts_cover_exactly(self):
        for count in (2, 3, 5, 7, 18, 36, 64):
            for total in REAL_ROWS:
                plan = plan_shards(total, DIM, count)
                cursor = 0
                for shard in range(count):
                    begin, end = plan.row_range(shard)
                    self.assertEqual(begin, cursor)
                    self.assertLess(begin, end)
                    self.assertEqual(plan.shard_of_row(begin), shard)
                    self.assertEqual(plan.shard_of_row(end - 1), shard)
                    cursor = end
                self.assertEqual(cursor, total)
                with self.assertRaises(IndexError):
                    plan.shard_of_row(total)

    def test_empty_shards_rejected_before_allocating(self):
        with self.assertRaisesRegex(ValueError, "empty shards"):
            plan_shards(1000, DIM, 4)

    def test_invalid_integer_geometry(self):
        for key in ("total_rows", "dim", "shard_count", "block", "granularity"):
            for bad in (0, -1, True, 1.5, "2"):
                args = dict(
                    total_rows=1048576,
                    dim=256,
                    shard_count=2,
                    block=32,
                    granularity=4096,
                )
                args[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    plan_shards(**args)
        plan = plan_shards(1048576, 256, 2)
        with self.assertRaises(ValueError):
            split_rows(10, plan)
        with self.assertRaises(IndexError):
            local_rows(plan, -2)


class MappingRequestsTest(unittest.TestCase):
    def test_covers_both_spans_without_overlap(self):
        plan = plan_shards(REAL_ROWS[0], DIM, 4)
        weight_base = 1 << 40
        scale_base = weight_base + plan.weight_span
        requests = mapping_requests(plan, weight_base, scale_base)
        self.assertEqual(len(requests), plan.shard_count * 2)

        weights = [r for r in requests if r.kind == "weight"]
        scales = [r for r in requests if r.kind == "scale"]
        for group, base, span in (
            (weights, weight_base, plan.weight_span),
            (scales, scale_base, plan.scale_span),
        ):
            group.sort(key=lambda r: r.va)
            cursor = base
            for request in group:
                self.assertEqual(request.va, cursor)
                self.assertEqual(request.va % DEFAULT_GRANULARITY, 0)
                cursor += request.size
            self.assertEqual(cursor, base + span)

    def test_handle_offsets_are_zero(self):
        # Fabric handles cannot map a sub-range, so weight and scale must be
        # separate Owner allocations and every map uses offset 0.
        plan = plan_shards(REAL_ROWS[0], DIM, 4)
        requests = mapping_requests(plan, 0, plan.weight_span)
        for request in requests:
            self.assertEqual(request.handle_offset, 0)

    def test_rejects_misaligned_bases(self):
        plan = plan_shards(1000, DIM, 4, granularity=8)
        with self.assertRaises(ValueError):
            mapping_requests(plan, 1, plan.weight_span)
        with self.assertRaises(ValueError):
            mapping_requests(plan, 0, -1)


ROWS = 1_048_576  # 4 x 262144, no padding


class FakeOps:
    def __init__(self, fail_on: str = ""):
        self.calls = []
        self.fail_on = fail_on
        self._next = 0x1000_0000_0000
        self.views = []

    def reserve(self, size, alignment):
        self.calls.append(("reserve", size))
        pointer = self._next
        self._next += 1 << 34
        return pointer

    def import_handle(self, raw):
        self.calls.append(("import", raw))
        if self.fail_on == "import":
            raise RuntimeError("import failed")
        return int.from_bytes(raw[:8], "little")

    def map(self, va, size, offset, handle):
        self.calls.append(("map", va, size, offset, handle))
        if self.fail_on == "map":
            raise RuntimeError("map failed")

    def set_access(self, va, size):
        self.calls.append(("set_access", va, size))
        if getattr(self, "fail_on", "") == "set_access":
            raise RuntimeError("set_access failed")

    def release(self, handle):
        self.calls.append(("release", handle))

    def synchronize(self):
        self.calls.append(("synchronize",))

    def unmap(self, va, size):
        self.calls.append(("unmap", va, size))

    def free(self, va, size):
        self.calls.append(("free", va, size))

    def view(self, va, shape, owner):
        self.views.append((va, shape))
        return (va, shape)


class ShardedLayerMappingTest(unittest.TestCase):
    def setUp(self):
        self.plan = plan_shards(ROWS, DIM, 4)
        self.handles = [bytes([i + 1]) * 8 for i in range(4)]

    def test_attach_maps_every_shard(self):
        ops = FakeOps()
        mapping = ShardedLayerMapping(
            self.plan, self.handles, self.handles, 0, ops=ops
        ).attach()
        self.assertEqual(mapping.state, "MAPPED")
        reserves = [c for c in ops.calls if c[0] == "reserve"]
        self.assertEqual(
            [c[1] for c in reserves], [self.plan.weight_span, self.plan.scale_span]
        )
        maps = [c for c in ops.calls if c[0] == "map"]
        self.assertEqual(len(maps), 8)
        for _, va, size, offset, _ in maps:
            self.assertEqual(va % self.plan.granularity, 0)
            self.assertEqual(size % self.plan.granularity, 0)
            self.assertEqual(offset, 0)  # fabric handles: offset must be zero

    def test_views_cover_padded_rows(self):
        ops = FakeOps()
        mapping = ShardedLayerMapping(
            self.plan, self.handles, self.handles, 0, ops=ops
        ).attach()
        weight = mapping.weight_view()
        scale = mapping.scale_view()
        self.assertEqual(weight[1], (self.plan.padded_rows, DIM))
        self.assertEqual(scale[1], (self.plan.padded_rows, DIM // 32))

    def test_close_unmaps_and_frees(self):
        ops = FakeOps()
        mapping = ShardedLayerMapping(
            self.plan, self.handles, self.handles, 0, ops=ops
        ).attach()
        mapping.close()
        self.assertEqual(mapping.state, "CLOSED")
        unmaps = [c for c in ops.calls if c[0] == "unmap"]
        frees = [c for c in ops.calls if c[0] == "free"]
        self.assertEqual(len(unmaps), 8)
        self.assertEqual(len(frees), 2)
        self.assertEqual(
            sorted(c[2] for c in frees),
            sorted([self.plan.weight_span, self.plan.scale_span]),
        )

    def test_failed_attach_rolls_back(self):
        ops = FakeOps(fail_on="map")
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops)
        with self.assertRaises(RuntimeError):
            mapping.attach()
        self.assertEqual(mapping.state, "CLOSED")
        # Whatever was reserved must have been freed.
        self.assertEqual(len([c for c in ops.calls if c[0] == "free"]), 2)

    def test_failed_shard_logs_safe_identity_and_preserves_original_error(self):
        generation = "00000000-0000-0000-0000-000000000007"
        owners = tuple(f"private-owner-{index}\n/private/path" for index in range(4))
        context = storage.LayerMappingContext(generation, 7, owners, owners)
        owner_digest = hashlib.sha256(owners[1].encode()).hexdigest()[:16]
        for operation in ("import_handle", "map", "set_access"):
            with self.subTest(operation=operation):
                ops = FakeOps()
                original_call = getattr(ops, operation)
                calls = 0
                error = RuntimeError("capability-secret /private/checkpoint/path")
                error.cuda_error = 201

                def fail_at(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 4:  # weight/scale pairs, then scale shard 1
                        raise error
                    return original_call(*args)

                setattr(ops, operation, fail_at)
                mapping = ShardedLayerMapping(
                    self.plan, self.handles, self.handles, 3, ops=ops, context=context
                )
                with self.assertLogs(storage._LOGGER, level="ERROR") as logged:
                    with self.assertRaises(RuntimeError) as caught:
                        mapping.attach()
                self.assertIs(caught.exception, error)
                self.assertEqual(mapping.state, "CLOSED")
                self.assertEqual(len(logged.output), 1)
                event = logged.output[0]
                for expected in (
                    "event=mapping_failed",
                    f"generation={generation}",
                    "layer=7",
                    "tensor_kind=scale",
                    "shard=1",
                    "device=3",
                    f"owner_sha256={owner_digest}",
                    f"operation={operation}",
                    "error_class=RuntimeError",
                    "cuda_error=201",
                ):
                    self.assertIn(expected, event)
                for secret in (
                    "capability-secret",
                    "/private",
                    "private-owner",
                    repr(self.handles),
                ):
                    self.assertNotIn(secret, event)

    def test_failed_logger_does_not_change_mapping_error_or_cleanup(self):
        ops = FakeOps(fail_on="map")
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops)
        with mock.patch.object(
            storage._LOGGER, "error", side_effect=RuntimeError("logger failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "map failed"):
                mapping.attach()
        self.assertEqual(mapping.state, "CLOSED")
        self.assertEqual(len([c for c in ops.calls if c[0] == "free"]), 2)

    def test_driver_status_is_available_without_parsing_error_text(self):
        class Status(IntEnum):
            CUDA_ERROR_INVALID_CONTEXT = 201

        driver = storage.CudaDriverOps.__new__(storage.CudaDriverOps)
        with self.assertRaisesRegex(RuntimeError, "map failed: 201") as caught:
            driver._check((Status.CUDA_ERROR_INVALID_CONTEXT,), "map")
        self.assertIs(type(caught.exception), RuntimeError)
        self.assertEqual(caught.exception.cuda_error, 201)
        self.assertEqual(caught.exception.cuda_error_name, "CUDA_ERROR_INVALID_CONTEXT")

    def test_rejects_bad_handles(self):
        with self.assertRaises(ValueError):
            ShardedLayerMapping(
                self.plan, self.handles[:3], self.handles, 0, ops=FakeOps()
            )
        with self.assertRaises(ValueError):
            ShardedLayerMapping(self.plan, [b""] * 4, self.handles, 0, ops=FakeOps())

    def test_double_attach_rejected(self):
        mapping = ShardedLayerMapping(
            self.plan, self.handles, self.handles, 0, ops=FakeOps()
        ).attach()
        with self.assertRaises(RuntimeError):
            mapping.attach()

    def test_view_before_attach_rejected(self):
        mapping = ShardedLayerMapping(
            self.plan, self.handles, self.handles, 0, ops=FakeOps()
        )
        with self.assertRaises(RuntimeError):
            mapping.weight_view()

    def test_failure_at_each_partial_attach_stage_balances_resources(self):
        for operation, positions in (
            ("reserve", (1, 2)),
            ("import_handle", (1, 4, 8)),
            ("map", (1, 4, 8)),
            ("set_access", (1, 4, 8)),
        ):
            for position in positions:
                with self.subTest(operation=operation, position=position):
                    ops = FakeOps()
                    original = getattr(ops, operation)
                    calls = 0

                    def fail_at(*args):
                        nonlocal calls
                        calls += 1
                        if calls == position:
                            raise RuntimeError("injected stage failure")
                        return original(*args)

                    setattr(ops, operation, fail_at)
                    mapping = ShardedLayerMapping(
                        self.plan, self.handles, self.handles, 0, ops=ops
                    )
                    with self.assertRaisesRegex(RuntimeError, "injected stage"):
                        mapping.attach()
                    names = [c[0] for c in ops.calls]
                    self.assertEqual(names.count("import"), names.count("release"))
                    self.assertEqual(names.count("map"), names.count("unmap"))
                    self.assertEqual(names.count("reserve"), names.count("free"))
                    self.assertEqual(mapping.info()["state"], "CLOSED")

    def test_authorization_failure_unmaps_and_releases_import(self):
        ops = FakeOps(fail_on="set_access")
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops)
        with self.assertRaisesRegex(RuntimeError, "set_access failed"):
            mapping.attach()
        self.assertEqual(mapping.state, "CLOSED")
        names = [c[0] for c in ops.calls]
        self.assertEqual(names.count("unmap"), 1)
        self.assertEqual(names.count("release"), 1)
        self.assertLess(names.index("synchronize"), names.index("unmap"))
        self.assertLess(names.index("unmap"), names.index("release"))
        self.assertLess(names.index("release"), names.index("free"))

    def test_cleanup_retries_each_operation_without_double_free(self):
        for operation in ("synchronize", "unmap", "release", "free"):
            with self.subTest(operation=operation):
                ops = FakeOps()
                mapping = ShardedLayerMapping(
                    self.plan, self.handles, self.handles, 0, ops=ops
                ).attach()
                original = getattr(ops, operation)
                calls = 0

                def fail_once(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 2 or (operation == "synchronize" and calls == 1):
                        raise RuntimeError("transient cleanup failure")
                    return original(*args)

                setattr(ops, operation, fail_once)
                with self.assertRaises(RuntimeError):
                    mapping.close()
                setattr(ops, operation, original)
                mapping.close()
                before = len(ops.calls)
                mapping.close()
                self.assertEqual(len(ops.calls), before)
                names = [c[0] for c in ops.calls]
                self.assertEqual(names.count("unmap"), 8)
                self.assertEqual(names.count("release"), 8)
                self.assertEqual(names.count("free"), 2)

    def test_failed_attach_retains_cleanup_resource_and_original_cause(self):
        from rtp_llm.models_py.modules.dsv41.engram_distributed.storage import (
            CleanupPendingError,
        )

        ops = FakeOps(fail_on="set_access")
        original = ops.unmap

        def fail(*args):
            raise RuntimeError("unmap failed")

        ops.unmap = fail
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops)
        with self.assertRaises(CleanupPendingError) as caught:
            mapping.attach()
        self.assertIs(caught.exception.resource, mapping)
        self.assertIn("set_access", str(caught.exception.__cause__))
        ops.unmap = original
        mapping.close()
        self.assertEqual(mapping.state, "CLOSED")

    def test_exporter_lease_blocks_close_until_last_alias_is_gone(self):
        from rtp_llm.models_py.modules.dsv41.engram_distributed.mapping import (
            ArrayViewLease,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.storage import (
            StorageBusyError,
        )

        ops = FakeOps()
        mapping = ShardedLayerMapping(
            self.plan, self.handles, self.handles, 0, ops=ops
        ).attach()
        lease = ArrayViewLease(mapping._weight_ptr, (1, 256), mapping)
        alias = lease
        del lease
        with self.assertRaises(StorageBusyError):
            mapping.close()
        self.assertFalse(any(c[0] == "unmap" for c in ops.calls))
        del alias
        mapping.close()
        self.assertEqual(mapping.info()["active_views"], 0)


if __name__ == "__main__":
    unittest.main()
