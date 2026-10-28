# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for real checkpoint weight/scale row partitioning."""
from types import SimpleNamespace
import unittest

from tools.engram_sidecar.framework.sharded_checkpoint_owner import tensor_shard_layout
from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards


class ShardedCheckpointLayoutTest(unittest.TestCase):
    def test_fp8_scale_uses_weight_partition_on_every_owner(self):
        rows, dim, granularity = 384006168, 256, 2 << 20
        tensors = [SimpleNamespace(name="layers.1.engram.embed." + role, shape=(rows, width))
                   for role, width in (("weight", dim), ("scale", dim // 32))]
        for count in (2, 4):
            plan = plan_shards(rows, dim, count, granularity=granularity)
            previous_end = 0
            for shard in range(count):
                weight = tensor_shard_layout(tensors[0], (rows, dim), shard, count, granularity)
                scale = tensor_shard_layout(tensors[1], (rows, dim), shard, count, granularity)
                self.assertEqual(weight[:2], scale[:2])
                self.assertEqual(weight[0], previous_end)
                previous_end = weight[1]
                self.assertEqual(weight[2], plan.weight_bytes)
                self.assertEqual(scale[2], plan.scale_bytes)
                self.assertEqual(weight[2], scale[2] * 32)
                self.assertEqual(scale[2] % granularity, 0)
            self.assertEqual(previous_end, rows)

    def test_mismatched_scale_rows_or_width_rejected(self):
        for shape in ((99, 8), (100, 7)):
            tensor = SimpleNamespace(name="layers.1.engram.embed.scale", shape=shape)
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                tensor_shard_layout(tensor, (100, 256), 0, 2, 4096)


if __name__ == "__main__":
    unittest.main()
