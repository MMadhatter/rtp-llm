# SPDX-License-Identifier: Apache-2.0
import unittest

from rtp_llm.models_py.modules.dsv41.engram_shard_config import (
    ShardConfig,
    ShardConfigError,
)


class ShardConfigTest(unittest.TestCase):
    def test_defaults_disable_sharing(self):
        config = ShardConfig.from_environ({})
        self.assertEqual(config.shard_count, 1)
        self.assertEqual(config.shard_index, 0)
        self.assertTrue(config.holds_shard)
        self.assertFalse(config.sharded)
        self.assertEqual(config.mode, "prefill")
        self.assertEqual(config.endpoints, ())

    def test_sharded_without_endpoints_is_rejected(self):
        with self.assertRaises(ShardConfigError):
            ShardConfig.from_environ({"ENGRAM_SHARD_COUNT": "4"})

    def test_full_config(self):
        env = {
            "ENGRAM_SHARD_COUNT": "4",
            "ENGRAM_SHARD_INDEX": "2",
            "ENGRAM_SHARD_ENDPOINTS": "10.0.0.1:19780, 10.0.0.2:19780,10.0.0.3:19780,10.0.0.4:19780",
            "ENGRAM_SHARD_MODE": "decode",
        }
        config = ShardConfig.from_environ(env)
        self.assertEqual(config.shard_index, 2)
        self.assertEqual(config.endpoint(0), "10.0.0.1:19780")
        self.assertEqual(config.endpoint(3), "10.0.0.4:19780")
        self.assertEqual(config.mode, "decode")

    def test_pure_consumer(self):
        env = {
            "ENGRAM_SHARD_COUNT": "4",
            "ENGRAM_SHARD_INDEX": "-1",
            "ENGRAM_SHARD_ENDPOINTS": "a:1,b:2,c:3,d:4",
        }
        config = ShardConfig.from_environ(env)
        self.assertFalse(config.holds_shard)
        self.assertEqual(config.endpoint(2), "c:3")

    def test_single_shard_needs_no_endpoints(self):
        config = ShardConfig.from_environ({"ENGRAM_SHARD_COUNT": "1", "ENGRAM_SHARD_INDEX": "0"})
        self.assertFalse(config.sharded)
        self.assertTrue(config.holds_shard)

    def test_rejects_bad_values(self):
        bad = [
            {"ENGRAM_SHARD_COUNT": "0"},
            {"ENGRAM_SHARD_COUNT": "x"},
            {"ENGRAM_SHARD_INDEX": "4"},
            {"ENGRAM_SHARD_INDEX": "-2"},
            {"ENGRAM_SHARD_COUNT": "4", "ENGRAM_SHARD_INDEX": "0", "ENGRAM_SHARD_ENDPOINTS": "a:1"},
            {"ENGRAM_SHARD_COUNT": "2", "ENGRAM_SHARD_ENDPOINTS": "a:1,a:1"},
            {"ENGRAM_SHARD_COUNT": "2", "ENGRAM_SHARD_ENDPOINTS": "a:1,b:99999"},
            {"ENGRAM_SHARD_COUNT": "2", "ENGRAM_SHARD_ENDPOINTS": "a,b:1"},
            {"ENGRAM_SHARD_MODE": "training"},
        ]
        for env in bad:
            with self.assertRaises(ShardConfigError, msg=str(env)):
                ShardConfig.from_environ(env)

    def test_describe_has_no_handle_material(self):
        config = ShardConfig.from_environ(
            {"ENGRAM_SHARD_COUNT": "2", "ENGRAM_SHARD_INDEX": "1", "ENGRAM_SHARD_ENDPOINTS": "a:1,b:2"}
        )
        described = config.describe()
        self.assertEqual(described["shard_count"], 2)
        self.assertNotIn("handle", str(described).lower())


if __name__ == "__main__":
    unittest.main()
