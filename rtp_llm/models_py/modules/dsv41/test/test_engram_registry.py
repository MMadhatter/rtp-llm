# SPDX-License-Identifier: Apache-2.0
"""Real mTLS coordination, atomic claims and launcher integration; no GPU mocks."""
import copy
import importlib
import json
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import grpc

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("engram_registry_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
registry = importlib.import_module(package.__name__ + ".engram_registry")
rpc = importlib.import_module(package.__name__ + ".engram_rpc")
mode = importlib.import_module(package.__name__ + ".engram_mode")
producer = importlib.import_module(package.__name__ + ".engram_producer")
IDENTITY = dict(
    service_id="test-service", model_revision="v1", dataset_id="sha256:" + "a" * 64
)
SCOPE = dict(fabric_domain="fabric-1", authorization_group="tenant-a")


def policy(count=10):
    return dict(
        schema_version=1,
        groups=[
            dict(
                scope=SCOPE,
                identity=IDENTITY,
                members=[
                    dict(
                        instance_id=f"sidecar-{i}",
                        principal=f"sidecar-{i}",
                        node_id=f"node-{i}",
                        owner_endpoint=f"127.0.0.1:{19000+i}",
                        reservation_id=f"reservation-{i}",
                        reserved_memory_bytes=4096,
                        can_produce=True,
                    )
                    for i in range(count)
                ],
            )
        ],
    )


def request(i=0, **fields):
    return dict(
        schema_version=1,
        scope=SCOPE,
        identity=IDENTITY,
        instance_id=f"sidecar-{i}",
        **fields,
    )


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "registry.db"
        self.now = 1000.0
        self.policy = policy()
        self.adapter = registry.SqlitePlatformAdapter(
            self.database, self.policy, clock=lambda: self.now
        )
        self.addCleanup(self.adapter.close)
        self.attempt = str(uuid4())

    def call(self, method, i=0, **fields):
        return self.adapter.dispatch(method, request(i, **fields), f"sidecar-{i}")

    def claim(self):
        return self.call(
            "AcquireProducer", attempt_id=self.attempt, required_memory_bytes=2048
        )

    def test_ten_candidates_two_adapters_only_one_grant(self):
        other = registry.SqlitePlatformAdapter(
            self.database, self.policy, clock=lambda: self.now
        )
        self.addCleanup(other.close)
        barrier = threading.Barrier(10)

        def run(i):
            barrier.wait()
            adapter = self.adapter if i % 2 else other
            return adapter.dispatch(
                "AcquireProducer",
                request(i, attempt_id=str(uuid4()), required_memory_bytes=2048),
                f"sidecar-{i}",
            )

        with ThreadPoolExecutor(10) as pool:
            results = list(pool.map(run, range(10)))
        self.assertEqual(sum(x["role"] == "PRODUCER" for x in results), 1)
        self.assertEqual(len({x["owner_epoch"] for x in results}), 1)
        self.assertTrue(all(x["state"] == "LOADING" for x in results))

    def test_claim_retry_idempotent_but_new_process_cannot_adopt(self):
        first = self.claim()
        self.assertEqual(self.claim()["owner_epoch"], first["owner_epoch"])
        with self.assertRaisesRegex(registry.RegistryError, "CLAIM_NOT_HELD"):
            self.call(
                "AcquireProducer", attempt_id=str(uuid4()), required_memory_bytes=2048
            )
        with self.assertRaisesRegex(registry.RegistryError, "IDEMPOTENCY_MISMATCH"):
            self.call(
                "AcquireProducer", attempt_id=self.attempt, required_memory_bytes=1024
            )

    def test_platform_static_assignment(self):
        p = copy.deepcopy(self.policy)
        for member in p["groups"][0]["members"][1:]:
            member.update(
                can_produce=False,
                owner_endpoint=None,
                reservation_id=None,
                reserved_memory_bytes=0,
            )
        other = registry.SqlitePlatformAdapter(Path(self.temp.name) / "static.db", p)
        self.addCleanup(other.close)
        result = other.dispatch(
            "AcquireProducer",
            request(1, attempt_id=str(uuid4()), required_memory_bytes=2048),
            "sidecar-1",
        )
        self.assertEqual((result["role"], result["state"]), ("CONSUMER", "EMPTY"))
        self.assertEqual(
            other.dispatch(
                "AcquireProducer",
                request(0, attempt_id=self.attempt, required_memory_bytes=2048),
                "sidecar-0",
            )["role"],
            "PRODUCER",
        )

    def test_expired_owner_is_sticky_and_never_reallocated(self):
        grant = self.claim()
        self.now += 31
        self.assertEqual(self.call("ResolveProducer")["state"], "STALE")
        result = self.call(
            "AcquireProducer", i=1, attempt_id=str(uuid4()), required_memory_bytes=2048
        )
        self.assertEqual(
            (result["role"], result["owner_epoch"]), ("CONSUMER", grant["owner_epoch"])
        )
        with self.assertRaisesRegex(registry.RegistryError, "CLAIM_EXPIRED"):
            self.call(
                "RenewProducer",
                attempt_id=self.attempt,
                owner_epoch=grant["owner_epoch"],
            )

    def test_capacity_must_already_be_reserved(self):
        with self.assertRaisesRegex(
            registry.RegistryError, "PRE_RESERVED_MEMORY_INSUFFICIENT"
        ):
            self.call(
                "AcquireProducer", attempt_id=self.attempt, required_memory_bytes=8192
            )
        self.assertEqual(self.call("ResolveProducer")["state"], "EMPTY")

    def test_ready_publication_immutable_and_epoch_checked(self):
        grant, generation = self.claim(), str(uuid4())
        fields = dict(
            attempt_id=self.attempt,
            owner_epoch=grant["owner_epoch"],
            generation=generation,
            physical_bytes=1024,
        )
        with self.assertRaisesRegex(registry.RegistryError, "CLAIM_NOT_HELD"):
            self.call("PublishReady", **{**fields, "owner_epoch": str(uuid4())})
        self.call("PublishReady", **fields)
        self.call("PublishReady", **fields)
        resolved = self.call("ResolveProducer", i=1)
        self.assertEqual(
            (resolved["state"], resolved["generation"]), ("READY", generation)
        )
        self.assertNotIn("attempt_id", resolved)
        self.assertNotIn("capabilities", resolved)
        with self.assertRaisesRegex(registry.RegistryError, "GENERATION_IMMUTABLE"):
            self.call("PublishReady", **{**fields, "generation": str(uuid4())})
        with self.assertRaisesRegex(registry.RegistryError, "RESERVATION_EXCEEDED"):
            self.call("PublishReady", **{**fields, "physical_bytes": 8192})

    def test_restart_preserves_claim_and_policy_rebind_fails_closed(self):
        grant = self.claim()
        reopened = registry.SqlitePlatformAdapter(
            self.database, self.policy, clock=lambda: self.now
        )
        self.addCleanup(reopened.close)
        self.assertEqual(
            reopened.dispatch("ResolveProducer", request(1), "sidecar-1")[
                "owner_epoch"
            ],
            grant["owner_epoch"],
        )
        rebound = copy.deepcopy(self.policy)
        rebound["groups"][0]["members"][0]["owner_endpoint"] = "127.0.0.1:19999"
        changed = registry.SqlitePlatformAdapter(
            self.database, rebound, clock=lambda: self.now
        )
        self.addCleanup(changed.close)
        with self.assertRaisesRegex(registry.RegistryError, "PLATFORM_BINDING_CHANGED"):
            changed.dispatch("ResolveProducer", request(1), "sidecar-1")

    def test_identity_scope_and_instance_cannot_be_forged(self):
        for wrong in (
            request(1),
            {**request(), "scope": {**SCOPE, "fabric_domain": "other"}},
            {**request(), "identity": {**IDENTITY, "model_revision": "v2"}},
        ):
            with self.assertRaisesRegex(registry.RegistryError, "NOT_AUTHORIZED"):
                self.adapter.dispatch("ResolveProducer", wrong, "sidecar-0")

    def test_invalid_memory_and_unexpected_fields(self):
        for value in (0, -1, True, 1.5, "1"):
            with self.assertRaises(rpc.ControlError):
                self.call(
                    "AcquireProducer",
                    attempt_id=self.attempt,
                    required_memory_bytes=value,
                )
        with self.assertRaises(rpc.ControlError):
            self.call("ResolveProducer", unexpected="ignored?")


class RpcTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.pki = Path(cls.temp.name) / "pki"
        script = ROOT.parents[3] / "tools/engram_sidecar/deploy/create-test-pki.sh"
        subprocess.run(
            ["bash", str(script), str(cls.pki), "127.0.0.1"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for i in range(10):
            name = f"sidecar-{i}"
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-subj",
                    f"/CN={name}",
                    "-keyout",
                    str(cls.pki / (name + ".key")),
                    "-out",
                    str(cls.pki / (name + ".csr")),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    "openssl",
                    "x509",
                    "-req",
                    "-in",
                    str(cls.pki / (name + ".csr")),
                    "-CA",
                    str(cls.pki / "ca.crt"),
                    "-CAkey",
                    str(cls.pki / "ca.key"),
                    "-CAcreateserial",
                    "-days",
                    "1",
                    "-extfile",
                    str(cls.pki / "consumer.ext"),
                    "-out",
                    str(cls.pki / (name + ".crt")),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.owner = rpc.OwnerControl(
            IDENTITY["service_id"], [f"sidecar-{i}" for i in range(10)]
        )
        self.owner_port = self.owner.start(
            "127.0.0.1:0",
            ca=self.pki / "ca.crt",
            cert=self.pki / "owner.crt",
            key=self.pki / "owner.key",
        )
        self.addCleanup(self.owner.stop)
        self.policy = policy()
        self.policy["groups"][0]["members"][0][
            "owner_endpoint"
        ] = f"127.0.0.1:{self.owner_port}"
        self.adapter = registry.SqlitePlatformAdapter(
            Path(self.tmp.name) / "registry.db", self.policy
        )
        self.addCleanup(self.adapter.close)
        self.server = registry.RegistryServer(self.adapter)
        self.port = self.server.start(
            "127.0.0.1:0",
            ca=self.pki / "ca.crt",
            cert=self.pki / "owner.crt",
            key=self.pki / "owner.key",
        )
        self.addCleanup(self.server.stop)

    def env(self, i=0):
        return dict(
            RTP_ENGRAM_MODE="shared",
            RTP_ENGRAM_REGISTRY_ENDPOINT=f"127.0.0.1:{self.port}",
            RTP_ENGRAM_SERVICE_ID=IDENTITY["service_id"],
            RTP_ENGRAM_MODEL_REVISION=IDENTITY["model_revision"],
            RTP_ENGRAM_DATASET_ID=IDENTITY["dataset_id"],
            RTP_ENGRAM_FABRIC_DOMAIN=SCOPE["fabric_domain"],
            RTP_ENGRAM_AUTHORIZATION_GROUP=SCOPE["authorization_group"],
            RTP_ENGRAM_INSTANCE_ID=f"sidecar-{i}",
            RTP_ENGRAM_TLS_CA=str(self.pki / "ca.crt"),
            RTP_ENGRAM_TLS_CERT=str(self.pki / f"sidecar-{i}.crt"),
            RTP_ENGRAM_TLS_KEY=str(self.pki / f"sidecar-{i}.key"),
            RTP_ENGRAM_RPC_TIMEOUT_S="2",
        )

    def client(self, i=0, env=None):
        c = registry.RegistryClient(env or self.env(i))
        self.addCleanup(c.close)
        return c

    def publish(self):
        c, attempt, generation = self.client(), str(uuid4()), str(uuid4())
        grant = c.acquire(attempt, 2048)
        self.owner.publish(
            dict(
                generation=generation,
                model_revision=IDENTITY["model_revision"],
                dataset_id=IDENTITY["dataset_id"],
            ),
            b'{"tables":[]}',
            {"a": "private"},
        )
        c.publish(attempt, grant["owner_epoch"], generation, 1024)
        return generation

    def test_ten_real_tls_clients_choose_one_producer(self):
        clients = [self.client(i) for i in range(10)]
        barrier = threading.Barrier(10)

        def run(i):
            barrier.wait()
            return clients[i].acquire(str(uuid4()), 2048)

        with ThreadPoolExecutor(10) as pool:
            results = list(pool.map(run, range(10)))
        self.assertEqual(sum(x["role"] == "PRODUCER" for x in results), 1)
        self.assertEqual(sum(x["role"] == "CONSUMER" for x in results), 9)
        self.assertEqual(len({x["owner_epoch"] for x in results}), 1)

    def test_launcher_discovers_endpoint_and_pins_generation_before_spawn(self):
        generation = self.publish()
        env = self.env(1)
        mode.prepare_engram_startup(env)
        self.assertEqual(env["RTP_ENGRAM_RPC_GENERATION"], generation)
        self.assertEqual(
            env[registry.RESOLVED_ENDPOINT_ENV], f"127.0.0.1:{self.owner_port}"
        )
        self.assertNotIn("RTP_ENGRAM_OWNER_ENDPOINT", env)
        self.assertEqual(mode.resolve_mode(env), ("shared", "rpc_registry"))
        c = rpc.Client(env)
        self.addCleanup(c.close)
        self.assertEqual(c.acquire(generation)["generation"], generation)

    def test_mismatched_owner_generation_rejected(self):
        c, attempt = self.client(), str(uuid4())
        grant = c.acquire(attempt, 2048)
        self.owner.publish(
            dict(
                generation=str(uuid4()),
                model_revision=IDENTITY["model_revision"],
                dataset_id=IDENTITY["dataset_id"],
            ),
            b'{"tables":[]}',
            {},
        )
        c.publish(attempt, grant["owner_epoch"], str(uuid4()), 1024)
        with self.assertRaisesRegex(rpc.ControlError, "RPC_GENERATION_CHANGED"):
            mode.prepare_engram_startup(self.env(1))

    def test_certificate_cannot_claim_another_instance(self):
        env = self.env(1)
        env["RTP_ENGRAM_INSTANCE_ID"] = "sidecar-0"
        with self.assertRaises(grpc.RpcError) as error:
            self.client(env=env).acquire(str(uuid4()), 2048)
        self.assertEqual(error.exception.code(), grpc.StatusCode.PERMISSION_DENIED)

    def test_signed_but_unregistered_client_is_rejected(self):
        env = self.env()
        env.update(
            RTP_ENGRAM_TLS_CERT=str(self.pki / "consumer.crt"),
            RTP_ENGRAM_TLS_KEY=str(self.pki / "consumer.key"),
        )
        with self.assertRaises(grpc.RpcError) as error:
            self.client(env=env).resolve()
        self.assertEqual(error.exception.code(), grpc.StatusCode.PERMISSION_DENIED)

    def test_local_does_not_connect_and_ambiguous_sources_fail(self):
        mode.prepare_engram_startup(
            {"RTP_ENGRAM_MODE": "local", registry.REGISTRY_ENV: "invalid"}
        )
        for key in ("RTP_ENGRAM_OWNER_ENDPOINT", "RTP_ENGRAM_EXTERNAL_BUNDLE"):
            with self.assertRaisesRegex(ValueError, "AMBIGUOUS"):
                mode.resolve_mode({**self.env(), key: "invalid"})

    def test_no_ready_does_not_silently_fallback(self):
        env = self.env()
        env.update(RTP_ENGRAM_MODE="auto", RTP_ENGRAM_DISCOVERY_TIMEOUT_S="0.02")
        with self.assertRaisesRegex(rpc.ControlError, "PRODUCER_READY_TIMEOUT"):
            mode.prepare_engram_startup(env)
        self.assertEqual(env["RTP_ENGRAM_MODE"], "auto")
        self.assertNotIn(registry.RESOLVED_ENDPOINT_ENV, env)

    def test_session_capacity_and_admission_stop_without_reclaim(self):
        session, grant = producer.ProducerSession.acquire(2048, self.env())
        self.addCleanup(session.close)
        self.assertTrue(session.live())
        with self.assertRaisesRegex(rpc.ControlError, "RESERVATION_EXCEEDED"):
            session.ensure_capacity(8192)
        self.owner.admission = session.live
        self.owner.publish(
            dict(
                generation=str(uuid4()),
                model_revision=IDENTITY["model_revision"],
                dataset_id=IDENTITY["dataset_id"],
            ),
            b'{"tables":[]}',
            {},
        )
        session._failed.set()
        env = self.env(1)
        env.pop(registry.REGISTRY_ENV)
        env["RTP_ENGRAM_OWNER_ENDPOINT"] = f"127.0.0.1:{self.owner_port}"
        c = rpc.Client(env)
        self.addCleanup(c.close)
        with self.assertRaises(grpc.RpcError) as error:
            c.describe()
        self.assertEqual(error.exception.code(), grpc.StatusCode.UNAVAILABLE)
        self.assertIsNotNone(self.owner._payload)
        self.assertEqual(self.client(1).resolve()["owner_epoch"], grant["owner_epoch"])


if __name__ == "__main__":
    unittest.main()
