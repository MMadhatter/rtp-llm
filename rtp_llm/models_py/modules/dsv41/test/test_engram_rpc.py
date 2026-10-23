# SPDX-License-Identifier: Apache-2.0
"""Real loopback mTLS transport tests; no GPU or model correctness claim."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from uuid import uuid4

import grpc

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("engram_rpc_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
rpc = importlib.import_module(package.__name__ + ".engram_rpc")
mode = importlib.import_module(package.__name__ + ".engram_mode")


class TransportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pki = Path(cls.tmp.name) / "pki"
        script = ROOT.parents[3] / "tools/engram_sidecar/deploy/create-test-pki.sh"
        subprocess.run(
            ["bash", str(script), str(cls.pki), "127.0.0.1"],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.bundle = dict(
            generation=str(uuid4()),
            model_revision="revision-1",
            dataset_id="sha256:" + "a" * 64,
        )
        self.owner = rpc.OwnerControl("service-a", ["engram-consumer"])
        port = self.owner.start(
            "127.0.0.1:0",
            ca=self.pki / "ca.crt",
            cert=self.pki / "owner.crt",
            key=self.pki / "owner.key",
        )
        self.addCleanup(self.owner.stop)
        self.env = dict(
            RTP_ENGRAM_OWNER_ENDPOINT=f"127.0.0.1:{port}",
            RTP_ENGRAM_SERVICE_ID="service-a",
            RTP_ENGRAM_MODEL_REVISION="revision-1",
            RTP_ENGRAM_DATASET_ID=self.bundle["dataset_id"],
            RTP_ENGRAM_TLS_CA=str(self.pki / "ca.crt"),
            RTP_ENGRAM_TLS_CERT=str(self.pki / "consumer.crt"),
            RTP_ENGRAM_TLS_KEY=str(self.pki / "consumer.key"),
            RTP_ENGRAM_RPC_TIMEOUT_S="2",
            RTP_ENGRAM_MODE="shared",
        )

    def client(self, env=None):
        client = rpc.Client(env or self.env)
        self.addCleanup(client.close)
        return client

    def publish(self):
        self.owner.publish(self.bundle, b'{"tables":[]}', {"allocation-1": "private"})

    def test_not_ready(self):
        with self.assertRaises(grpc.RpcError) as cm:
            self.client().describe()
        self.assertEqual(cm.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

    def test_describe_contains_no_handles(self):
        self.publish()
        value = self.client().describe()
        self.assertEqual(set(value), {"schema_version", "identity", "generation"})
        self.assertNotIn("private", json.dumps(value))

    def test_acquire_in_memory_and_deep_copy(self):
        self.publish()
        self.bundle["model_revision"] = "mutated-after-publish"
        value = self.client().acquire(self.bundle["generation"])
        self.assertEqual(value["capabilities"], {"allocation-1": "private"})
        self.assertEqual(value["bundle"]["model_revision"], "revision-1")
        self.assertEqual(self.owner.acquisitions, 1)

    def test_wrong_service_model_or_dataset_rejected(self):
        self.publish()
        for key, value in [
            ("SERVICE_ID", "other"),
            ("MODEL_REVISION", "other"),
            ("DATASET_ID", "sha256:" + "b" * 64),
        ]:
            with self.subTest(key=key):
                env = {**self.env, "RTP_ENGRAM_" + key: value}
                with self.assertRaises(grpc.RpcError) as cm:
                    self.client(env).describe()
                self.assertEqual(cm.exception.code(), grpc.StatusCode.NOT_FOUND)

    def test_stale_generation_rejected(self):
        self.publish()
        with self.assertRaises(grpc.RpcError) as cm:
            self.client().acquire(str(uuid4()))
        self.assertEqual(cm.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

    def test_no_live_republish(self):
        self.publish()
        with self.assertRaisesRegex(rpc.ControlError, "ALREADY_PUBLISHED"):
            self.publish()

    def test_launcher_pins_generation_without_fetching_handles(self):
        self.publish()
        self.env["RTP_ENGRAM_MODE"] = "auto"
        mode.prepare_engram_startup(self.env)
        self.assertEqual(
            self.env["RTP_ENGRAM_RPC_GENERATION"], self.bundle["generation"]
        )
        self.assertEqual(self.env["RTP_ENGRAM_MODE"], "shared")
        self.assertNotIn("RTP_ENGRAM_EXTERNAL_BUNDLE", self.env)
        self.assertEqual(self.owner.acquisitions, 0)

    def test_local_never_connects(self):
        env = {
            **self.env,
            "RTP_ENGRAM_MODE": "local",
            "RTP_ENGRAM_OWNER_ENDPOINT": "bad",
        }
        mode.prepare_engram_startup(env)
        self.assertNotIn("RTP_ENGRAM_RPC_GENERATION", env)

    def test_file_and_rpc_cannot_mix(self):
        with self.assertRaisesRegex(ValueError, "AMBIGUOUS"):
            mode.prepare_engram_startup(
                {**self.env, "RTP_ENGRAM_EXTERNAL_BUNDLE": "/old"}
            )

    def test_client_certificate_required(self):
        self.publish()
        credentials = grpc.ssl_channel_credentials((self.pki / "ca.crt").read_bytes())
        with grpc.secure_channel(self.env[rpc.ENDPOINT_ENV], credentials) as channel:
            with self.assertRaises(grpc.RpcError):
                channel.unary_unary(
                    "/" + rpc.SERVICE + "/Describe", request_serializer=rpc.encode
                )(
                    {
                        "schema_version": 1,
                        "identity": rpc.configured_identity(self.env),
                    },
                    timeout=1,
                )

    def test_authorized_certificate_name_required(self):
        self.publish()
        self.owner.allowed = {b"different-consumer"}
        with self.assertRaises(grpc.RpcError) as cm:
            self.client().describe()
        self.assertEqual(cm.exception.code(), grpc.StatusCode.PERMISSION_DENIED)

    def test_bad_json_and_oversized_payload(self):
        for raw in (b"[]", b'{"x":1,"x":2}', b"x" * (rpc.MAX_MESSAGE + 1)):
            with self.assertRaises(ValueError):
                rpc.decode(raw)


if __name__ == "__main__":
    unittest.main()
