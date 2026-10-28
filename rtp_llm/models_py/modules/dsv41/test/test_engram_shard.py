# SPDX-License-Identifier: Apache-2.0
import unittest

from rtp_llm.models_py.modules.dsv41.engram_shard import (
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
        plan = plan_shards(total, DIM, 4)
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
        total = 1000
        plan = plan_shards(total, DIM, 4)
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
        plan = plan_shards(1000, DIM, 4)
        self.assertIsInstance(plan, ShardPlan)
        self.assertEqual(hash(plan), hash(plan_shards(1000, DIM, 4)))
        with self.assertRaises(Exception):
            plan.shard_rows = 1


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

    def test_handle_offsets(self):
        plan = plan_shards(REAL_ROWS[0], DIM, 4)
        requests = mapping_requests(plan, 0, plan.weight_span)
        for request in requests:
            expected = 0 if request.kind == "weight" else plan.weight_bytes
            self.assertEqual(request.handle_offset, expected)

    def test_rejects_misaligned_bases(self):
        plan = plan_shards(1000, DIM, 4)
        with self.assertRaises(ValueError):
            mapping_requests(plan, 1, plan.weight_span)
        with self.assertRaises(ValueError):
            mapping_requests(plan, 0, -1)


if __name__ == "__main__":
    unittest.main()
