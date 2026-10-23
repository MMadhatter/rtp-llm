# SPDX-License-Identifier: Apache-2.0
import json
import unittest

from tools.engram_sidecar.deploy.make_pd_env import render


class PairEnvTest(unittest.TestCase):
    def setUp(self):
        self.pair = {
            "schema": 1,
            "seq_size_per_block": 256,
            "roles": {
                role: {
                    "ip": ip,
                    "port": port,
                    "gpu_uuids": [f"GPU-{role}-{i}" for i in range(4)],
                }
                for role, ip, port in (
                    ("PREFILL", "192.0.2.1", 18640),
                    ("DECODE", "192.0.2.2", 18641),
                )
            },
        }

    def env(self, role, graph=True):
        return dict(
            line.split("=", 1)
            for line in render(self.pair, role, cuda_graph=graph).splitlines()
        )

    def test_routes_and_cache_geometry_agree(self):
        p, d = self.env("PREFILL"), self.env("DECODE")
        for key in (
            "MODEL_SERVICE_CONFIG",
            "SEQ_SIZE_PER_BLOCK",
            "KERNEL_SEQ_SIZE_PER_BLOCK",
            "TP_SIZE",
            "EP_SIZE",
        ):
            self.assertEqual(p[key], d[key])
        self.assertEqual(p["REMOTE_RPC_SERVER_IP"], "192.0.2.2")
        self.assertEqual(d["REMOTE_RPC_SERVER_IP"], "192.0.2.1")
        self.assertEqual(p["REMOTE_SERVER_PORT"], d["START_PORT"])
        endpoints = json.loads(p["MODEL_SERVICE_CONFIG"])["role_endpoints"][0]
        self.assertEqual(endpoints["prefill_endpoint"]["address"], "192.0.2.1:18640")
        self.assertEqual(endpoints["decode_endpoint"]["address"], "192.0.2.2:18641")

    def test_storage_is_selected_by_deployment_and_graph_is_decode_only(self):
        for role in ("PREFILL", "DECODE"):
            env = self.env(role)
            self.assertFalse(any(key.startswith("RTP_ENGRAM_") for key in env))
            self.assertNotIn("PYTHONPATH", env)
            self.assertEqual(env["CHECKPOINT_PATH"], "/checkpoint")
            self.assertEqual(env["ENABLE_CUDA_GRAPH"], "1" if role == "DECODE" else "0")
        self.assertEqual(self.env("DECODE", False)["ENABLE_CUDA_GRAPH"], "0")

    def test_invalid_pair_is_rejected_before_output(self):
        self.pair["roles"]["DECODE"]["ip"] = "192.0.2.1"
        with self.assertRaises(ValueError):
            render(self.pair, "PREFILL")


if __name__ == "__main__":
    unittest.main()
