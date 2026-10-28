# SPDX-License-Identifier: Apache-2.0
import unittest

from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards
from rtp_llm.models_py.modules.dsv41.engram_shard_storage import ShardedLayerMapping

DIM = 256
ROWS = 1_048_576  # 4 x 262144, no padding


class FakeOps:
    def __init__(self, fail_on: str = ""):
        self.calls = []
        self.fail_on = fail_on
        self._next = 0x1000_0000_0000
        self.views = []

    def reserve(self, size):
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

    def unmap(self, va, size):
        self.calls.append(("unmap", va, size))

    def free(self, va, size):
        self.calls.append(("free", va, size))

    def view(self, va, shape):
        self.views.append((va, shape))
        return (va, shape)


class ShardedLayerMappingTest(unittest.TestCase):
    def setUp(self):
        self.plan = plan_shards(ROWS, DIM, 4)
        self.handles = [bytes([i + 1]) * 8 for i in range(4)]

    def test_attach_maps_every_shard(self):
        ops = FakeOps()
        mapping = ShardedLayerMapping(self.plan, self.handles, 0, ops=ops).attach()
        self.assertEqual(mapping.state, "MAPPED")
        reserves = [c for c in ops.calls if c[0] == "reserve"]
        self.assertEqual([c[1] for c in reserves], [self.plan.weight_span, self.plan.scale_span])
        maps = [c for c in ops.calls if c[0] == "map"]
        self.assertEqual(len(maps), 8)
        for _, va, size, offset, _ in maps:
            self.assertEqual(va % self.plan.granularity, 0)
            self.assertEqual(size % self.plan.granularity, 0)
            self.assertIn(offset, (0, self.plan.weight_bytes))

    def test_views_cover_padded_rows(self):
        ops = FakeOps()
        mapping = ShardedLayerMapping(self.plan, self.handles, 0, ops=ops).attach()
        weight = mapping.weight_view()
        scale = mapping.scale_view()
        self.assertEqual(weight[1], (self.plan.padded_rows, DIM))
        self.assertEqual(scale[1], (self.plan.padded_rows, DIM // 32))

    def test_close_unmaps_and_frees(self):
        ops = FakeOps()
        mapping = ShardedLayerMapping(self.plan, self.handles, 0, ops=ops).attach()
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
        mapping = ShardedLayerMapping(self.plan, self.handles, 0, ops=ops)
        with self.assertRaises(RuntimeError):
            mapping.attach()
        self.assertEqual(mapping.state, "CLOSED")
        # Whatever was reserved must have been freed.
        self.assertEqual(len([c for c in ops.calls if c[0] == "free"]), 2)

    def test_rejects_bad_handles(self):
        with self.assertRaises(ValueError):
            ShardedLayerMapping(self.plan, self.handles[:3], 0, ops=FakeOps())
        with self.assertRaises(ValueError):
            ShardedLayerMapping(self.plan, [b""] * 4, 0, ops=FakeOps())

    def test_double_attach_rejected(self):
        mapping = ShardedLayerMapping(self.plan, self.handles, 0, ops=FakeOps()).attach()
        with self.assertRaises(RuntimeError):
            mapping.attach()

    def test_view_before_attach_rejected(self):
        mapping = ShardedLayerMapping(self.plan, self.handles, 0, ops=FakeOps())
        with self.assertRaises(RuntimeError):
            mapping.weight_view()


if __name__ == "__main__":
    unittest.main()
