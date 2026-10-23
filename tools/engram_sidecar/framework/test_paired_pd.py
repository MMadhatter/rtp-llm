"""Regression coverage for cross-host PD configuration drift."""

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.engram_sidecar.framework.paired_pd import (
    COMMON_KEYS,
    check_local_devices,
    launch_environment,
    validate_pair,
)


class PairedPDTest(unittest.TestCase):
    def setUp(self):
        self.pair = {
            "schema": 1,
            "seq_size_per_block": 256,
            "roles": {
                role: {
                    "ip": ip,
                    "port": port,
                    "gpu_uuids": [f"GPU-{role}-{n}" for n in range(4)],
                }
                for role, ip, port in [
                    ("PREFILL", "192.0.2.1", 18640),
                    ("DECODE", "192.0.2.2", 18641),
                ]
            },
        }
        self.args = SimpleNamespace(
            checkpoint=Path("/checkpoint"),
            bundle=Path("/private/bundle.json"),
            storage_mode="external-egm",
            cuda_graph=True,
            max_allocation_bytes=1 << 37,
            budget_bytes=1 << 38,
        )

    def test_both_roles_override_conflicting_inherited_geometry(self):
        p = launch_environment(
            self.pair,
            "PREFILL",
            self.args,
            {
                "SEQ_SIZE_PER_BLOCK": "128",
                "PREFILL_CP_SIZE": "2",
                "CUDA_VISIBLE_DEVICES": "GPU-wrong-host",
                "FP8_KV_CACHE": "0",
            },
        )
        d = launch_environment(self.pair, "DECODE", self.args, {})
        self.assertEqual({k: p[k] for k in COMMON_KEYS}, {k: d[k] for k in COMMON_KEYS})
        self.assertEqual(p["SEQ_SIZE_PER_BLOCK"], "256")
        self.assertEqual(p["REMOTE_SERVER_PORT"], d["START_PORT"])
        self.assertEqual(d["REMOTE_SERVER_PORT"], p["START_PORT"])
        self.assertEqual(p["REMOTE_RPC_SERVER_IP"], "192.0.2.2")
        self.assertEqual(d["REMOTE_RPC_SERVER_IP"], "192.0.2.1")
        self.assertEqual(p["RTP_ENGRAM_PAIR_SHA256"], d["RTP_ENGRAM_PAIR_SHA256"])
        self.assertEqual(p["ENABLE_CUDA_GRAPH"], "0")
        self.assertEqual(d["ENABLE_CUDA_GRAPH"], "1")
        self.assertNotEqual(p["CUDA_VISIBLE_DEVICES"], d["CUDA_VISIBLE_DEVICES"])

    def test_role_specific_geometry_is_rejected(self):
        self.pair["roles"]["PREFILL"]["seq_size_per_block"] = 128
        with self.assertRaises(ValueError):
            validate_pair(self.pair)

    def test_pair_identity_changes_with_geometry(self):
        other = copy.deepcopy(self.pair)
        other["seq_size_per_block"] = 128
        self.assertNotEqual(validate_pair(other), validate_pair(self.pair))

    def test_wrong_host_uuid_fails_before_model_launch(self):
        with patch("subprocess.check_output", return_value="GPU-other\n"):
            with self.assertRaisesRegex(ValueError, "absent from this host"):
                check_local_devices(self.pair["roles"]["PREFILL"]["gpu_uuids"])


if __name__ == "__main__":
    unittest.main()
