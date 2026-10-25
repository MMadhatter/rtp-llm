# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from c2_sim import config_for, validate
from k8s_pair import documents, quantity

HERE = Path(__file__).resolve().parent


class PairDeploymentTest(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((HERE / "c2-sim.example.json").read_text())
        self.spec["common"]["dataset_id"] = "sha256:" + "a" * 64
        self.spec["owner_epoch"] = "01234567-1234-1234-1234-0123456789ab"

    def test_rdma_is_explicit_and_only_in_rtp(self):
        m = self.spec["instances"][0]
        m["rdma_devices"] = ["/dev/infiniband/uverbs0", "/dev/infiniband/rdma_cm"]
        validate(self.spec)
        self.assertEqual(
            config_for(self.spec, m, False)["RDMA_DEVICES"], " ".join(m["rdma_devices"])
        )
        self.assertNotIn("RDMA_DEVICES", config_for(self.spec, m, True))
        for bad in (
            ["/dev/infiniband/*"],
            ["/dev/nvidia0"],
            ["/dev/infiniband/../mem"],
            ["/dev/infiniband/uverbs0"] * 2,
            "all",
        ):
            m["rdma_devices"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate(self.spec)

    def test_yaml_pairs_identity_roles_resources_and_gate(self):
        docs = documents(self.spec)
        self.assertEqual(len(docs), 4)
        pods = [d for d in docs if d["kind"] == "Pod"]
        roles = []
        for pod, m in zip(pods, self.spec["instances"]):
            cfg = pod["spec"]
            self.assertTrue(cfg["schedulingGates"])
            self.assertNotIn("nodeName", cfg)
            self.assertNotIn("initContainers", cfg)
            self.assertEqual(cfg["restartPolicy"], "Never")
            side, rtp = cfg["containers"]
            roles.append(
                dict((x["name"], x["value"]) for x in side["env"])[
                    "RTP_ENGRAM_SIDECAR_ROLE"
                ]
            )
            self.assertEqual(side["envFrom"][0], rtp["envFrom"][0])
            rtp_env = {x["name"]: x["value"] for x in rtp["env"]}
            self.assertEqual(
                rtp_env["RTP_ENGRAM_OWNER_EPOCH"], self.spec["owner_epoch"]
            )
            self.assertEqual(rtp_env["RTP_ENGRAM_INSTANCE_ID"], m["instance_id"])
            self.assertNotIn("RTP_ENGRAM_SIDECAR_ROLE", rtp_env)
            self.assertIn("until bash", rtp["args"][-1])
            for container in cfg["containers"]:
                self.assertNotIn("livenessProbe", container)
                self.assertFalse(container["securityContext"]["privileged"])
                self.assertNotIn("nvidia.com/gpu", container["resources"]["limits"])
        self.assertEqual(roles, ["producer", "consumer"])
        side = pods[0]["spec"]["containers"][0]
        self.assertIn("--allowed-client", side["args"])
        self.assertEqual(len(pods[1]["spec"]["containers"][0]["volumeMounts"]), 1)
        self.assertEqual(quantity("32g"), str(32 * 1024**3))

    def test_yaml_requires_existing_assignment(self):
        del self.spec["owner_epoch"]
        with self.assertRaisesRegex(ValueError, "render first"):
            documents(self.spec)

    def test_wrapper_sequence_and_failure_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "calls"
            fake = root / "python"
            fake.write_text(
                '#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nsys.exit(7 if sys.argv[2]==os.environ.get("FAIL") else 0)\n'
            )
            fake.chmod(0o700)
            env = dict(os.environ, PYTHON=str(fake), CALLS=str(capture))
            for action, expected in (
                ("sidecar", ["start-sidecar", "wait-owner"]),
                ("rtp", ["wait-owner", "start-rtp", "wait-rtp"]),
                ("stop-rtp", ["stop-rtp"]),
            ):
                capture.write_text("")
                subprocess.run(
                    [
                        "bash",
                        str(HERE / "pouch-pair.sh"),
                        "/run path",
                        "prefill-a",
                        action,
                    ],
                    env=env,
                    check=True,
                )
                calls = [json.loads(line) for line in capture.read_text().splitlines()]
                self.assertEqual([x[1] for x in calls], expected)
                self.assertTrue(all(x[3] == "/run path" for x in calls))
            capture.write_text("")
            proc = subprocess.run(
                ["bash", str(HERE / "pouch-pair.sh"), "/run", "prefill-a", "rtp"],
                env=dict(env, FAIL="wait-owner"),
            )
            self.assertEqual(proc.returncode, 7)
            self.assertEqual(len(capture.read_text().splitlines()), 1)
