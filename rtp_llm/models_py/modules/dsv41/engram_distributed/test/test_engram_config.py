# SPDX-License-Identifier: Apache-2.0
"""Storage policy, shard configuration and capacity limits; CPU only."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv41.engram_distributed import startup
from rtp_llm.models_py.modules.dsv41.engram_distributed.config import (
    ShardConfig,
    ShardConfigError,
    StoragePolicy,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.placement import PlacementPlan

spec = importlib.util.spec_from_file_location(
    "engram_capacity_cpu_test", Path(__file__).resolve().parents[1] / "capacity.py"
)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CapacityTest(unittest.TestCase):
    def test_defaults_are_the_hard_bounds(self):
        self.assertEqual(
            c.capacity_from_environ({}),
            (c.HARD_MAX_ALLOCATION_BYTES, c.HARD_MAX_TOTAL_ALLOCATION_BYTES),
        )

    def test_explicit_local_ceiling_is_supported(self):
        self.assertEqual(
            c.capacity_from_environ(
                {
                    c.MAX_ALLOCATION_ENV: str(128 << 30),
                    c.MAX_TOTAL_ALLOCATION_ENV: str(256 << 30),
                }
            ),
            (128 << 30, 256 << 30),
        )

    def test_each_limit_can_be_lowered_independently(self):
        self.assertEqual(c.validate_capacity(1 << 30, 64 << 20), (1 << 30, 64 << 20))
        self.assertEqual(
            c.capacity_from_environ({c.MAX_ALLOCATION_ENV: "4096"}), (4096, 256 << 30)
        )

    def test_capacity_types_and_hard_bounds(self):
        for allocation, total in (
            (0, 1),
            (1, 0),
            (-1, 1),
            (True, 1),
            (1, False),
            (1.0, 1),
            (1, "1"),
            ((128 << 30) + 1, 256 << 30),
            (128 << 30, (256 << 30) + 1),
            (1 << 64, 1),
        ):
            with self.subTest(allocation=allocation, total=total):
                with self.assertRaisesRegex(c.CapacityError, "INVALID_CAPACITY_LIMIT"):
                    c.validate_capacity(allocation, total)

    def test_environment_rejects_ambiguous_or_unbounded_inputs(self):
        for value in (
            "",
            "0",
            "-1",
            "+1",
            "01",
            " 1",
            "1 ",
            "1\n",
            "1GiB",
            "1e9",
            "0x1000",
            "١",
            "1" * 10000,
            1,
            True,
            None,
        ):
            for key in (c.MAX_ALLOCATION_ENV, c.MAX_TOTAL_ALLOCATION_ENV):
                with self.subTest(key=key, value=str(value)[:20]):
                    with self.assertRaisesRegex(
                        c.CapacityError, "INVALID_CAPACITY_LIMIT"
                    ):
                        c.capacity_from_environ({key: value})

    def test_bundle_like_keys_cannot_change_limits(self):
        self.assertEqual(
            c.capacity_from_environ(
                {"budget_bytes": str(256 << 30), "max_allocation_bytes": str(128 << 30)}
            ),
            (c.HARD_MAX_ALLOCATION_BYTES, c.HARD_MAX_TOTAL_ALLOCATION_BYTES),
        )

    def test_real_layout_padded_capacity_uses_64_bit_offsets(self):
        sizes = [rows * (256 + 8) for rows in (384006168, 384016682)]
        padded = [
            c.checked_allocation_bytes(size, 2 << 20, 128 << 30) for size in sizes
        ]
        self.assertEqual(sizes, [101377628352, 101380404048])
        self.assertEqual(padded, [101378424832, 101380521984])
        self.assertEqual(sum(padded), 202758946816)
        self.assertLess(sum(padded), 256 << 30)

    def test_rounding_and_exact_cap_boundaries(self):
        limit, granularity = 128 << 30, 2 << 20
        for size, expected in (
            (1, granularity),
            (granularity, granularity),
            ((1 << 32) + 1, (1 << 32) + granularity),
            (limit - 1, limit),
            (limit, limit),
        ):
            self.assertEqual(
                c.checked_allocation_bytes(size, granularity, limit), expected
            )
        for size, alignment, cap in (
            (limit + 1, granularity, limit),
            (limit, 3, limit),
            (4097, 4096, 8191),
            ((1 << 63) - 1, granularity, limit),
        ):
            with self.assertRaisesRegex(c.CapacityError, "ALLOCATION_LIMIT"):
                c.checked_allocation_bytes(size, alignment, cap)

    def test_rounding_rejects_invalid_numeric_arguments(self):
        for arguments in (
            (0, 1, 1),
            (True, 1, 1),
            (1, 0, 1),
            (1, True, 1),
            (1, 1, 0),
            (1 << 63, 1, 1),
            (1, (128 << 30) + 1, 1),
            (1, 1, (128 << 30) + 1),
        ):
            with self.assertRaisesRegex(c.CapacityError, "INVALID_CAPACITY_LIMIT"):
                c.checked_allocation_bytes(*arguments)


def manifest(count):
    return NS(
        schema_version=1,
        tensors=[
            NS(
                shards=[
                    NS(owner_id=f"host-{i}", owner_incarnation=f"boot-{i}")
                    for i in range(count)
                ]
            )
            for _ in range(4)
        ],
    )


class ConfigTest(unittest.TestCase):
    def test_p_is_independent_of_model_parallelism_and_instance_count(self):
        for count in (2, 3, 5, 18, 36, 64):
            for instances in (1, 2, 10):
                config = ShardConfig.from_environ(
                    {
                        "ENGRAM_SHARD_COUNT": str(count),
                        "TP_SIZE": "1",
                        "DP_SIZE": "8",
                        "EP_SIZE": "8",
                        "INSTANCES": str(instances),
                    }
                )
                self.assertEqual(config.shard_count, count)
                config.validate_manifest(manifest(count))

    def test_inference_from_manifest_and_explicit_mismatch(self):
        self.assertEqual(ShardConfig.from_environ({}, inferred_count=3).shard_count, 3)
        with self.assertRaisesRegex(ShardConfigError, "COUNT_MISMATCH"):
            ShardConfig(3).validate_manifest(manifest(2))

    def test_single_host_and_invalid_p_are_rejected(self):
        for value in (None, True, 1, 0, -1, 65, 2.0, "2"):
            with self.subTest(value=value), self.assertRaises(ShardConfigError):
                ShardConfig(value)
        for value in ("", "-1", "2.0", "２", " 2", 2):
            with self.subTest(value=value), self.assertRaises(ShardConfigError):
                ShardConfig.from_environ({"ENGRAM_SHARD_COUNT": value})

    def test_cancelled_mode_and_rpc_options_fail_explicitly(self):
        for key in (
            "ENGRAM_SHARD_MODE",
            "ENGRAM_SHARD_INDEX",
            "ENGRAM_SHARD_ENDPOINTS",
        ):
            with self.assertRaisesRegex(ShardConfigError, "REMOVED"):
                ShardConfig.from_environ({key: "x"}, inferred_count=2)

    def test_missing_duplicate_or_replaced_host_is_rejected(self):
        for case in ("missing", "duplicate", "restart"):
            doc = manifest(3)
            if case == "missing":
                doc.tensors[1].shards.pop()
            if case == "duplicate":
                doc.tensors[1].shards[1].owner_id = "host-0"
            if case == "restart":
                doc.tensors[1].shards[1].owner_incarnation = "new-boot"
            with self.subTest(case=case), self.assertRaises(ShardConfigError):
                ShardConfig(3).validate_manifest(doc)


class StoragePolicyTest(unittest.TestCase):
    def test_modes_are_complete_and_invalid_modes_fail_closed(self):
        for mode, shared, memory in (
            ("host", False, "host"),
            ("host_shared", True, "host"),
            ("gpu_shared", True, "gpu"),
        ):
            policy = StoragePolicy.from_environ({"RTP_ENGRAM_STORAGE": mode})
            self.assertEqual((policy.shared, policy.memory_kind), (shared, memory))
        for mode in ("", "gpu", "HOST", None, 1):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                StoragePolicy.from_environ({"RTP_ENGRAM_STORAGE": mode})

    def test_legacy_config_preserves_shared_host_and_gpu(self):
        self.assertFalse(StoragePolicy.from_environ({}).shared)
        for memory in ("host", "gpu"):
            policy = StoragePolicy.from_environ(
                {"ENGRAM_SHARD_COUNT": "2", "RTP_ENGRAM_MEMORY": memory}
            )
            self.assertTrue(policy.shared)
            self.assertEqual(policy.memory_kind, memory)
            self.assertEqual(policy.default_graph_mode, "eager")
        with self.assertRaisesRegex(ValueError, "REQUIRES_SHARED_STORAGE"):
            StoragePolicy.from_environ({"RTP_ENGRAM_MEMORY": "gpu"})

    def test_local_mode_overrides_retained_legacy_topology_without_discovery(self):
        env = {
            "RTP_ENGRAM_STORAGE": "host",
            "ENGRAM_SHARD_COUNT": "2",
            "RTP_ENGRAM_COHORT": "unused",
            "RTP_ENGRAM_MEMORY": "gpu",
        }
        with patch.object(
            startup,
            "_gang_parts",
            side_effect=AssertionError("local mode entered exchange"),
        ):
            self.assertIsNone(startup.build_config(env, 8))
        self.assertEqual(StoragePolicy.from_environ(env).memory_kind, "host")

    def test_shared_mode_uses_gang_without_legacy_enable_flags(self):
        for mode, memory in (("host_shared", "host"), ("gpu_shared", "gpu")):
            env = {"RTP_ENGRAM_STORAGE": mode, startup.REVISION_ENV: "test-revision"}
            with patch.object(startup, "_annotation_cohort", return_value=None):
                config = startup.build_config(
                    env,
                    8,
                    gang_hosts=["10.0.0.1"] * 4 + ["10.0.0.2"] * 4,
                    checkpoint="/unused",
                )
            self.assertEqual(config["memory_kind"], memory)
            self.assertEqual(len(config["group"]["members"]), 2)
            self.assertEqual(config["gpu_devices"], (0, 1, 2, 3))
            self.assertEqual(
                config["storage_policy"].default_graph_mode, "process_lifetime"
            )

    def test_capacity_applies_to_physical_allocations_not_host_extent(self):
        checkpoint = NS(
            layers=[
                NS(
                    layer_id=1,
                    tensors=[
                        NS(name="weight", shape=(65536, 64)),
                        NS(name="scale", shape=(65536, 2)),
                    ],
                )
            ]
        )
        # Production names also carry their role suffix.
        for tensor in checkpoint.layers[0].tensors:
            tensor.name = "layer." + tensor.name
        plan = PlacementPlan.build(checkpoint, 0, 2, 4096, {1: (0, 1, 2, 3)})
        plan.validate_capacity(1 << 20, 4 << 20)
        self.assertEqual(plan.total_bytes, 32768 * 66)
        self.assertEqual(plan.bytes_by_location(), {i: 8192 * 66 for i in range(4)})
        with self.assertRaisesRegex(ValueError, "allocation limit"):
            plan.validate_capacity((512 << 10) - 1, 4 << 20)
        with self.assertRaisesRegex(ValueError, "host budget"):
            plan.validate_capacity(1 << 20, plan.total_bytes - 1)


if __name__ == "__main__":
    unittest.main()
