# SPDX-License-Identifier: Apache-2.0
"""Assigned-owner contract and real mTLS control-plane tests (no GPU claims)."""
import importlib
import os
from pathlib import Path
import stat
import threading
import time
import types
import unittest
from unittest.mock import patch
from uuid import uuid4

import grpc
import test_engram_rpc as transport
from test_engram_rpc import rpc, mode, package

assignment = importlib.import_module(package.__name__ + ".engram_assignment")
producer = importlib.import_module(package.__name__ + ".engram_producer")


def platform_env():
    return dict(
        RTP_ENGRAM_DISCOVERY="platform",
        RTP_ENGRAM_OWNER_ENDPOINT="127.0.0.1:19780",
        RTP_ENGRAM_AUTHORIZATION_GROUP="tenant-a",
        RTP_ENGRAM_FABRIC_DOMAIN="domain-1",
        RTP_ENGRAM_OWNER_INSTANCE_ID="owner-a",
        RTP_ENGRAM_OWNER_EPOCH=str(uuid4()),
        RTP_ENGRAM_INSTANCE_ID="engram-consumer",
        RTP_ENGRAM_SERVICE_ID="service-a",
        RTP_ENGRAM_MODEL_REVISION="revision-1",
        RTP_ENGRAM_DATASET_ID="sha256:" + "a" * 64,
        RTP_ENGRAM_STARTUP_TIMEOUT_S="2",
    )


class AssignmentTest(unittest.TestCase):
    def test_required_fields_and_ambiguous_sources(self):
        env = platform_env()
        self.assertEqual(
            assignment.configured_assignment(env)["authorization_group"], "tenant-a"
        )
        for key in env:
            if key in (
                "RTP_ENGRAM_DISCOVERY",
                "RTP_ENGRAM_SERVICE_ID",
                "RTP_ENGRAM_MODEL_REVISION",
                "RTP_ENGRAM_DATASET_ID",
                "RTP_ENGRAM_STARTUP_TIMEOUT_S",
            ):
                continue
            with self.subTest(key=key), self.assertRaises(ValueError):
                assignment.configured_assignment(
                    {k: v for k, v in env.items() if k != key}
                )
        for key in ("REGISTRY_ENDPOINT", "RESOLVED_OWNER_ENDPOINT", "EXTERNAL_BUNDLE"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "AMBIGUOUS"):
                assignment.configured_assignment({**env, "RTP_ENGRAM_" + key: ""})
        with self.assertRaisesRegex(ValueError, "DISCOVERY_INVALID"):
            assignment.configured_assignment({**env, "RTP_ENGRAM_DISCOVERY": "auto"})
        with self.assertRaisesRegex(ValueError, "EPOCH_INVALID"):
            assignment.configured_assignment({**env, "RTP_ENGRAM_OWNER_EPOCH": "old"})

    def test_local_explicitly_bypasses_assignment(self):
        self.assertEqual(
            mode.resolve_mode(
                {"RTP_ENGRAM_MODE": "local", "RTP_ENGRAM_DISCOVERY": "invalid"}
            )[0],
            "local",
        )
        with self.assertRaises(ValueError):
            mode.resolve_mode(
                {"RTP_ENGRAM_MODE": "auto", "RTP_ENGRAM_DISCOVERY": "platform"}
            )

    def test_only_assigned_producer_and_reserved_capacity(self):
        env = {
            **platform_env(),
            "RTP_ENGRAM_SIDECAR_ROLE": "producer",
            "RTP_ENGRAM_INSTANCE_ID": "owner-a",
            "RTP_ENGRAM_RESERVATION_ID": "reservation-a",
            "RTP_ENGRAM_RESERVED_MEMORY_BYTES": "1024",
        }
        with patch.object(
            producer.RegistryClient,
            "__init__",
            side_effect=AssertionError("Registry must not be contacted"),
        ):
            session = producer.AssignedProducerSession(512, env)
            self.assertTrue(session.live())
            self.assertEqual(
                session.publish_ready(str(uuid4()), 1024)["state"], "READY"
            )
            with self.assertRaisesRegex(rpc.ControlError, "RESERVATION_EXCEEDED"):
                session.ensure_capacity(1025)
            session.close()
            with self.assertRaisesRegex(rpc.ControlError, "CLOSED"):
                session.ensure_capacity(1)
        for update in (
            {"RTP_ENGRAM_INSTANCE_ID": "other"},
            {"RTP_ENGRAM_SIDECAR_ROLE": "auto"},
            {"RTP_ENGRAM_RESERVED_MEMORY_BYTES": "511"},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                producer.AssignedProducerSession(512, {**env, **update})

    def test_exactly_one_accessible_imex_channel(self):
        channels = [Path("/dev/nvidia-caps-imex-channels/channel7")]
        with patch.object(Path, "glob", return_value=channels), patch.object(
            Path, "stat", return_value=types.SimpleNamespace(st_mode=stat.S_IFCHR)
        ), patch("os.access", return_value=True):
            assignment.validate_imex_device({"RTP_ENGRAM_IMEX_CHANNEL": "7"})
            with self.assertRaisesRegex(ValueError, "NOT_EXCLUSIVE"):
                assignment.validate_imex_device({"RTP_ENGRAM_IMEX_CHANNEL": "8"})
            channels.append(Path("/dev/nvidia-caps-imex-channels/channel0"))
            with self.assertRaisesRegex(ValueError, "NOT_EXCLUSIVE"):
                assignment.validate_imex_device({"RTP_ENGRAM_IMEX_CHANNEL": "7"})
        with self.assertRaisesRegex(ValueError, "REQUIRED"):
            assignment.validate_imex_device({})


class AssignedTransportTest(unittest.TestCase):
    setUpClass = classmethod(transport.TransportTest.setUpClass.__func__)
    tearDownClass = classmethod(transport.TransportTest.tearDownClass.__func__)
    client = transport.TransportTest.client
    publish = transport.TransportTest.publish

    def setUp(self):
        transport.TransportTest.setUp(self)
        self.env = {**platform_env(), **self.env}
        self.owner.assignment = assignment.configured_assignment(self.env)

    def test_describe_and_acquire_match_assignment(self):
        self.publish()
        client = self.client()
        self.assertEqual(client.describe()["assignment"], self.owner.assignment)
        self.assertNotIn("capabilities", client.describe())
        self.assertEqual(
            client.acquire(self.bundle["generation"])["capabilities"],
            {"allocation-1": "private"},
        )

    def test_cross_tenant_owner_epoch_and_fabric_rejected(self):
        self.publish()
        for key, value in (
            ("AUTHORIZATION_GROUP", "tenant-b"),
            ("FABRIC_DOMAIN", "domain-2"),
            ("OWNER_INSTANCE_ID", "owner-b"),
            ("OWNER_EPOCH", str(uuid4())),
            ("INSTANCE_ID", "forged-consumer"),
        ):
            with self.subTest(key=key), self.assertRaises(grpc.RpcError) as cm:
                self.client({**self.env, "RTP_ENGRAM_" + key: value}).acquire(
                    self.bundle["generation"]
                )
            self.assertEqual(cm.exception.code(), grpc.StatusCode.PERMISSION_DENIED)
        self.assertEqual(self.owner.acquisitions, 0)

    def test_same_domain_other_service_cannot_acquire(self):
        self.publish()
        with self.assertRaises(grpc.RpcError) as cm:
            self.client(
                {**self.env, "RTP_ENGRAM_SERVICE_ID": "other-business"}
            ).acquire(self.bundle["generation"])
        self.assertEqual(cm.exception.code(), grpc.StatusCode.NOT_FOUND)
        self.assertEqual(self.owner.acquisitions, 0)

    def test_legacy_request_rejected_and_missing_response_assignment_rejected(self):
        self.publish()
        env = {k: v for k, v in self.env.items() if k != "RTP_ENGRAM_DISCOVERY"}
        with self.assertRaises(grpc.RpcError) as cm:
            self.client(env).describe()
        self.assertEqual(cm.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        fake_channel = types.SimpleNamespace(
            unary_unary=lambda *a, **kw: lambda *a, **kw: dict(
                schema_version=1,
                identity=rpc.configured_identity(self.env),
                generation=self.bundle["generation"],
            )
        )
        client = self.client()
        with patch.object(client, "channel", fake_channel), self.assertRaisesRegex(
            rpc.ControlError, "ASSIGNMENT_MISMATCH"
        ):
            client.describe()

    def test_wait_then_pin_without_acquiring_handles(self):
        timer = threading.Timer(0.05, self.publish)
        timer.start()
        self.addCleanup(timer.join)
        rpc.pin_generation(self.env)
        self.assertEqual(
            self.env["RTP_ENGRAM_RPC_GENERATION"], self.bundle["generation"]
        )
        self.assertEqual(self.owner.acquisitions, 0)

    def test_wait_is_bounded_and_auth_errors_do_not_fallback(self):
        self.env["RTP_ENGRAM_STARTUP_TIMEOUT_S"] = "0.05"
        start = time.monotonic()
        with self.assertRaises(grpc.RpcError):
            rpc.pin_generation(self.env)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertNotIn("RTP_ENGRAM_RPC_GENERATION", self.env)
        self.publish()
        self.env["RTP_ENGRAM_AUTHORIZATION_GROUP"] = "tenant-b"
        self.env["RTP_ENGRAM_STARTUP_TIMEOUT_S"] = "300"
        with self.assertRaises(grpc.RpcError) as cm:
            rpc.pin_generation(self.env)
        self.assertEqual(cm.exception.code(), grpc.StatusCode.PERMISSION_DENIED)

    def test_launcher_imex_gate_precedes_rpc(self):
        with patch.object(
            assignment, "validate_imex_device", side_effect=ValueError("wrong channel")
        ), patch.object(rpc, "pin_generation") as pin:
            with self.assertRaisesRegex(ValueError, "wrong channel"):
                mode.prepare_engram_startup(self.env)
            pin.assert_not_called()


if __name__ == "__main__":
    unittest.main()
