# SPDX-License-Identifier: Apache-2.0
import base64
import json
import logging
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from uuid import UUID

from rtp_llm.models_py.modules.dsv41.engram_distributed import bootstrap as b
from rtp_llm.models_py.modules.dsv41.engram_distributed import exchange as e
from rtp_llm.models_py.modules.dsv41.engram_distributed import startup as s
from rtp_llm.models_py.modules.dsv41.engram_distributed.shard import plan_shards


def fixture(n=2, p=3, ranks=2):
    group = dict(
        generation=str(UUID(int=123)), model_revision="test-revision", members=[]
    )
    docs = []
    plan = plan_shards(p * 4096 - 7, 64, p, granularity=4096)
    for instance in range(n):
        shards = [
            dict(index=i, host_id=f"host-{i}", publisher_rank=pos)
            for pos, i in enumerate(range(instance, p, n))
        ]
        group["members"].append(
            dict(instance_id=f"i{instance}", world_size=ranks, shards=shards)
        )
    for index in range(p):
        begin, end = plan.row_range(index)
        tensors = {}
        for role, width, size in [
            ("weight", 64, plan.weight_bytes),
            ("scale", 2, plan.scale_bytes),
        ]:
            aid = str(UUID(int=1000 + 2 * index + (role == "scale")))
            tensors["layers.1.engram.embed." + role] = dict(
                shape=[plan.total_rows, width],
                row_begin=begin,
                row_end=end,
                allocation_bytes=size,
                allocation_granularity=4096,
                logical_bytes=(end - begin) * width,
                allocation_id=aid,
                sha256="sha256:" + "a" * 64,
                handle=base64.b64encode(aid.encode()).decode(),
            )
        docs.append(
            dict(
                schema_version=2,
                shard=index,
                shards=p,
                generation=group["generation"],
                model_revision=group["model_revision"],
                hash_abi="sha256:" + "b" * 64,
                host_id=f"host-{index}",
                owner_id=f"host-{index}",
                owner_incarnation=str(UUID(int=200 + index)),
                tensors=tensors,
            )
        )
    return group, docs


def samples(_, manifest):
    return {
        t.name: dict(
            row_ids=sorted(
                {0, t.shape[0] // 2, t.shape[0] - 1}
                | {r for part in t.shards for r in (part.row_begin, part.row_end - 1)}
            ),
            sha256="sha256:" + "c" * 64,
        )
        for t in manifest.tensors
    }


def package(group, docs):
    return s.assemble_package(group, docs, None, samples=samples)


def cohort_env(p, instances=2, index=0, timeout=3):
    return {
        s.COHORT_ENV: ",".join(
            f"127.0.0.{i + 1}:{29500 + i}" for i in range(instances)
        ),
        s.INDEX_ENV: str(index),
        s.REVISION_ENV: "test-revision",
        s.COUNT_ENV: str(p),
        s.TIMEOUT_ENV: str(timeout),
        s.CHECKPOINT_ENV: "/tmp/checkpoint",
        "RTP_ENGRAM_LOCAL_DEVICE": "0",
        "RTP_ENGRAM_LOCAL_NUMA": "0",
    }


def derived(env, world_size):
    """Descriptors whose host identity matches the group derived from env."""
    config = s.build_group(env, world_size)
    group = config["group"]
    p = sum(len(m["shards"]) for m in group["members"])
    plan = plan_shards(p * 4096 - 7, 64, p, granularity=4096)
    hosts = {sh["index"]: sh["host_id"] for m in group["members"] for sh in m["shards"]}
    docs = []
    for index in range(p):
        begin, end = plan.row_range(index)
        tensors = {}
        for role, width, size in [
            ("weight", 64, plan.weight_bytes),
            ("scale", 2, plan.scale_bytes),
        ]:
            aid = str(UUID(int=1000 + 2 * index + (role == "scale")))
            tensors["layers.1.engram.embed." + role] = dict(
                shape=[plan.total_rows, width],
                row_begin=begin,
                row_end=end,
                allocation_bytes=size,
                allocation_granularity=4096,
                logical_bytes=(end - begin) * width,
                allocation_id=aid,
                sha256="sha256:" + "a" * 64,
                handle=base64.b64encode(aid.encode()).decode(),
            )
        docs.append(
            dict(
                schema_version=2,
                shard=index,
                shards=p,
                generation=group["generation"],
                model_revision=group["model_revision"],
                hash_abi="sha256:" + "b" * 64,
                host_id=hosts[index],
                owner_id=hosts[index],
                owner_incarnation=str(UUID(int=200 + index)),
                tensors=tensors,
            )
        )
    return config, docs


class GroupTest(unittest.TestCase):
    def test_supported_static_shapes(self):
        for n, p, ranks in [(1, 2, 2), (2, 3, 2), (3, 5, 2), (5, 5, 1)]:
            group, docs = fixture(n, p, ranks)
            self.assertEqual(len(e.validate_group(group)), 64)
            s.verify_package(group, package(group, docs))

    def test_invalid_memberships_and_host_assignments(self):
        mutations = [
            lambda g: g["members"][1].update(instance_id="i0"),
            lambda g: g["members"][1].update(shards=[]),
            lambda g: g["members"][0]["shards"][0].update(publisher_rank=True),
            lambda g: g["members"][0]["shards"][0].update(index=1),
            lambda g: g["members"][0]["shards"][0].update(host_id="host-1"),
            lambda g: g["members"][0].update(world_size=0),
        ]
        for mutate in mutations:
            group, _ = fixture()
            mutate(group)
            with self.assertRaises((ValueError, e.ExchangeError)):
                e.validate_group(group)

    def test_cohort_waits_for_all_instances_and_rejects_changes(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: package(group, d), 2)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(
                cohort.submit, "i0", cohort.group_digest, [docs[0], docs[2]]
            )
            time.sleep(0.03)
            self.assertFalse(first.done())
            second = pool.submit(cohort.submit, "i1", cohort.group_digest, [docs[1]])
            result = first.result()
            self.assertEqual(result, second.result())
        self.assertEqual(
            cohort.submit("i0", cohort.group_digest, [docs[0], docs[2]]), result
        )
        changed = deepcopy(docs[0])
        changed["owner_incarnation"] = str(UUID(int=999))
        with self.assertRaisesRegex(e.ExchangeError, "REPUBLICATION"):
            cohort.submit("i0", cohort.group_digest, [changed, docs[2]])
        with self.assertRaisesRegex(e.ExchangeError, "GROUP_MISMATCH"):
            cohort.submit("i1", "wrong", [docs[1]])
        with self.assertRaisesRegex(e.ExchangeError, "UNKNOWN"):
            cohort.submit("outsider", cohort.group_digest, [])
        cohort.ack("i0", e.digest(result))
        cohort.ack("i1", e.digest(result))
        cohort.wait_acknowledged()

    def test_missing_member_and_failed_builder_do_not_hang(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: {}, 0.03)
        with self.assertRaisesRegex(e.ExchangeError, "TIMEOUT"):
            cohort.submit("i0", cohort.group_digest, [docs[0], docs[2]])

        def fail(_):
            raise ValueError("private handle details")

        cohort = e.Cohort(group, fail, 2)
        with ThreadPoolExecutor(2) as pool:
            pending = pool.submit(
                cohort.submit, "i0", cohort.group_digest, [docs[0], docs[2]]
            )
            with self.assertRaisesRegex(e.ExchangeError, "^ASSEMBLY_FAILED$"):
                cohort.submit("i1", cohort.group_digest, [docs[1]])
            with self.assertRaisesRegex(e.ExchangeError, "^ASSEMBLY_FAILED$"):
                pending.result()

    def test_slow_builder_and_missing_ack_are_bounded(self):
        group, docs = fixture(1, 2, 2)
        release = threading.Event()

        def slow(_):
            release.wait(2)
            return {}

        cohort = e.Cohort(group, slow, 0.03)
        try:
            with self.assertRaisesRegex(e.ExchangeError, "TIMEOUT"):
                cohort.submit("i0", cohort.group_digest, docs)
        finally:
            release.set()
        cohort = e.Cohort(group, lambda d: package(group, d), 0.05)
        payload = cohort.submit("i0", cohort.group_digest, docs)
        with self.assertRaisesRegex(e.ExchangeError, "ACK_MISMATCH"):
            cohort.ack("i0", "wrong-payload")
        with self.assertRaisesRegex(e.ExchangeError, "TIMEOUT"):
            cohort.wait_acknowledged()

    def test_bounded_frames_and_invalid_generation(self):
        with self.assertRaisesRegex(e.ExchangeError, "MESSAGE_TOO_LARGE"):
            e.decode(b"x" * (e.MAX_MESSAGE + 1))
        with self.assertRaisesRegex(e.ExchangeError, "MESSAGE_TOO_LARGE"):
            e.encode({"value": "x" * e.MAX_MESSAGE})
        with self.assertRaises(ValueError):
            e.decode(b'{"instance_id":"a","instance_id":"b"}')
        for value in (None, 1, True, "not-a-uuid"):
            group, _ = fixture()
            group["generation"] = value
            with self.assertRaisesRegex(e.ExchangeError, "INVALID_GENERATION"):
                e.validate_group(group)

    def test_memory_package_and_full_identity_checks(self):
        env = cohort_env(2)
        config, docs = derived(env, world_size=2)
        payload = package(config["group"], docs)
        memory = s.to_memory_package(config["group"], payload)
        self.assertEqual(
            set(memory),
            {
                "raw",
                "dataset_id",
                "generation",
                "model_revision",
                "capabilities",
                "samples",
            },
        )
        self.assertIsInstance(memory["raw"], bytes)
        self.assertTrue(
            all(isinstance(v, bytes) and v for v in memory["capabilities"].values())
        )
        changed = deepcopy(payload)
        changed["capabilities"].pop(next(iter(changed["capabilities"])))
        with self.assertRaisesRegex(e.ExchangeError, "CAPABILITIES"):
            s.verify_package(config["group"], changed)
        changed_group = deepcopy(config["group"])
        changed_group["generation"] = str(UUID(int=999))
        with self.assertRaisesRegex(e.ExchangeError, "IDENTITY"):
            s.verify_package(changed_group, payload)


class ExchangeDiagnosticsTest(unittest.TestCase):
    @contextmanager
    def capture_assembly_logs(self):
        recorded = threading.Event()

        class SignalHandler(logging.Handler):
            def emit(self, record):
                if "event=assembly_failed" in record.getMessage():
                    recorded.set()

        logger = logging.getLogger(e.__name__)
        with self.assertLogs(e.__name__, level="ERROR") as captured:
            handler = SignalHandler(level=logging.ERROR)
            # Run after assertLogs has captured the record. The assembly thread
            # notifies waiters before logging, so future.result() is not a fence.
            logger.addHandler(handler)
            try:
                yield captured, recorded
            finally:
                logger.removeHandler(handler)

    def serve(self, server):
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

        self.addCleanup(cleanup)
        return server.server_address

    def request(self, address, group_digest, documents):
        return e.remote_exchange(
            address,
            None,
            "i0",
            group_digest,
            documents,
            time.monotonic() + 2,
            lambda package: self.fail("a failed exchange must not verify or ACK"),
        )

    def test_real_assembly_io_and_descriptor_failures_reach_remote_safely(self):
        for failure in (
            "checkpoint_io",
            "descriptor_validation",
            "manifest_validation",
        ):
            with self.subTest(
                failure=failure
            ), tempfile.TemporaryDirectory() as directory:
                group, docs = fixture(1, 2, 2)
                private_path = str(
                    Path(directory) / "private-checkpoint-handle-details"
                )
                if failure == "checkpoint_io":

                    def builder(documents):
                        return s.assemble_package(
                            group,
                            documents,
                            private_path,
                            samples=lambda path, manifest: Path(path).read_bytes(),
                        )

                    expected = ("file_not_found", "FILE_NOT_FOUND")
                elif failure == "descriptor_validation":
                    docs[1]["hash_abi"] = "sha256:" + "d" * 64
                    builder = lambda documents: package(group, documents)
                    expected = ("validation", "VALIDATION_FAILED")
                else:
                    # Geometry is valid, but manifest identity rejects the owner UUID.
                    docs[0]["owner_incarnation"] = "private-owner-value"
                    builder = lambda documents: package(group, documents)
                    expected = ("manifest", "INVALID_UUID")
                cohort = e.Cohort(group, builder, 2)
                address = self.serve(
                    e.ExchangeServer(("127.0.0.1", 0), cohort, None, None)
                )
                with self.capture_assembly_logs() as (captured, recorded):
                    with self.assertRaisesRegex(
                        e.ExchangeError, "^ASSEMBLY_FAILED$"
                    ) as raised:
                        self.request(address, cohort.group_digest, docs)
                    self.assertTrue(recorded.wait(timeout=2))
                self.assertEqual(
                    (raised.exception.error_class, raised.exception.cause_code),
                    expected,
                )
                self.assertEqual(
                    (cohort.error.error_class, cohort.error.cause_code), expected
                )
                self.assertIsNone(cohort.error.__traceback__)
                self.assertIsNone(cohort.error.__context__)
                assembly_logs = [
                    line for line in captured.output if "event=assembly_failed" in line
                ]
                self.assertEqual(len(assembly_logs), 1)
                self.assertIn("generation=" + group["generation"], assembly_logs[0])
                self.assertIn("cause_code=" + expected[1], assembly_logs[0])
                self.assertTrue(
                    all(record.exc_info is None for record in captured.records)
                )
                for private in (
                    private_path,
                    "private-owner-value",
                    docs[0]["tensors"]["layers.1.engram.embed.weight"]["handle"],
                ):
                    self.assertNotIn(private, "\n".join(captured.output))
                    self.assertNotIn(private, str(raised.exception))

    def test_all_local_waiters_receive_categories_without_original_exception(self):
        group, docs = fixture()

        def builder(_):
            raise ValueError("private capability payload")

        cohort = e.Cohort(group, builder, 2)
        with self.capture_assembly_logs() as (captured, recorded), ThreadPoolExecutor(
            2
        ) as pool:
            futures = [
                pool.submit(
                    cohort.submit, "i0", cohort.group_digest, [docs[0], docs[2]]
                ),
                pool.submit(cohort.submit, "i1", cohort.group_digest, [docs[1]]),
            ]
            failures = []
            for future in futures:
                with self.assertRaisesRegex(
                    e.ExchangeError, "^ASSEMBLY_FAILED$"
                ) as raised:
                    future.result()
                failures.append(raised.exception)
            self.assertTrue(recorded.wait(timeout=2))
        self.assertIsNot(failures[0], failures[1])
        self.assertEqual(len(captured.records), 1)
        for failure in failures:
            self.assertEqual(failure.cause_code, "VALIDATION_FAILED")
            self.assertEqual(failure.error_class, "validation")
            self.assertIsNone(failure.__context__)
        self.assertNotIn("private capability payload", captured.output[0])

    def test_coordinator_group_rejection_preserves_protocol_code(self):
        group, docs = fixture(1, 2, 2)
        cohort = e.Cohort(group, lambda documents: {}, 2)
        address = self.serve(e.ExchangeServer(("127.0.0.1", 0), cohort, None, None))
        with self.assertLogs(e.__name__, level="ERROR"):
            with self.assertRaisesRegex(e.ExchangeError, "^GROUP_MISMATCH$"):
                self.request(address, "0" * 64, docs)
        self.assertEqual(cohort.documents, {})

    def test_broken_logging_handler_cannot_delay_or_replace_assembly_failure(self):
        group, docs = fixture(1, 2, 2)

        def builder(_):
            raise ValueError("private builder detail")

        cohort = e.Cohort(group, builder, 2)
        address = self.serve(e.ExchangeServer(("127.0.0.1", 0), cohort, None, None))
        assembly_logged = threading.Event()
        remote_logged = threading.Event()
        published_before_log = []

        class BrokenHandler(logging.Handler):
            def emit(self, record):
                if "event=assembly_failed" in record.getMessage():
                    published_before_log.append(cohort.error is not None)
                    assembly_logged.set()
                if "event=remote_exchange_failed" in record.getMessage():
                    remote_logged.set()
                raise RuntimeError("private logging backend error")

        logger = logging.getLogger(e.__name__)
        handler = BrokenHandler(level=logging.ERROR)
        logger.addHandler(handler)
        try:
            with self.assertRaisesRegex(e.ExchangeError, "^ASSEMBLY_FAILED$") as raised:
                self.request(address, cohort.group_digest, docs)
            self.assertEqual(raised.exception.cause_code, "VALIDATION_FAILED")
            self.assertTrue(assembly_logged.wait(timeout=2))
            self.assertTrue(remote_logged.is_set())
            self.assertEqual(published_before_log, [True])
            with self.assertRaisesRegex(e.ExchangeError, "^ASSEMBLY_FAILED$") as local:
                cohort.submit("i0", cohort.group_digest, docs)
            self.assertEqual(local.exception.cause_code, "VALIDATION_FAILED")
        finally:
            logger.removeHandler(handler)

    def test_info_logging_failure_does_not_stall_successful_or_failed_exchange(self):
        for builder_fails in (False, True):
            with self.subTest(builder_fails=builder_fails):
                group, docs = fixture(1, 2, 2)

                def builder(documents):
                    if builder_fails:
                        raise ValueError("private builder detail")
                    return package(group, documents)

                cohort = e.Cohort(group, builder, 2)
                messages = []

                class BrokenHandler(logging.Handler):
                    def emit(self, record):
                        messages.append(record.getMessage())
                        raise RuntimeError("logging handler failure")

                logger = logging.getLogger(e.__name__)
                old_level = logger.level
                handler = BrokenHandler(level=logging.INFO)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                try:
                    address = self.serve(
                        e.ExchangeServer(("127.0.0.1", 0), cohort, None, None)
                    )
                    if builder_fails:
                        with self.assertRaisesRegex(
                            e.ExchangeError, "^ASSEMBLY_FAILED$"
                        ) as raised:
                            self.request(address, cohort.group_digest, docs)
                        self.assertEqual(
                            raised.exception.cause_code, "VALIDATION_FAILED"
                        )
                    else:
                        result = e.remote_exchange(
                            address,
                            None,
                            "i0",
                            cohort.group_digest,
                            docs,
                            time.monotonic() + 2,
                            lambda value: s.verify_package(group, value),
                        )
                        self.assertEqual(
                            result["manifest"]["generation"], group["generation"]
                        )
                        self.assertEqual(cohort.acks, cohort.members)
                    self.assertTrue(
                        any("assembling package" in message for message in messages)
                    )
                finally:
                    logger.removeHandler(handler)
                    logger.setLevel(old_level)

    def test_legacy_unknown_and_malformed_responses_are_sanitized(self):
        cases = (
            (b'{"error":"GROUP_MISMATCH"}', "GROUP_MISMATCH"),
            (b'{"error":"REMOTE_EXCHANGE_FAILED"}', "REMOTE_EXCHANGE_FAILED"),
            (b'{"error":"private-handle-details"}', "REMOTE_EXCHANGE_FAILED"),
            (b"private-handle-details", "REMOTE_EXCHANGE_FAILED"),
            (
                b'{"error":"GROUP_MISMATCH","error":"private-handle-details"}',
                "REMOTE_EXCHANGE_FAILED",
            ),
            (
                b'{"error":"GROUP_MISMATCH","error_class":["private-handle-details"],"cause_code":{}}',
                "GROUP_MISMATCH",
            ),
            (
                b'{"error":"ASSEMBLY_FAILED","error_class":"private-handle-details","cause_code":"private-handle-details"}',
                "ASSEMBLY_FAILED",
            ),
        )
        for payload, expected in cases:
            with self.subTest(payload=payload):

                class LegacyHandler(BaseHTTPRequestHandler):
                    def log_message(self, *args):
                        pass

                    def do_POST(self):
                        self.rfile.read(int(self.headers["Content-Length"]))
                        self.send_response(400)
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)

                address = self.serve(
                    ThreadingHTTPServer(("127.0.0.1", 0), LegacyHandler)
                )
                with self.assertLogs(e.__name__, level="ERROR") as captured:
                    with self.assertRaisesRegex(
                        e.ExchangeError, "^" + expected + "$"
                    ) as raised:
                        self.request(address, "0" * 64, [])
                self.assertIsNone(raised.exception.error_class)
                self.assertIsNone(raised.exception.cause_code)
                self.assertNotIn("private-handle-details", "\n".join(captured.output))


class MemoryStore:
    """Shared test Store with real blocking semantics and bounded waits."""

    def __init__(self):
        self.data = {}
        self.condition = threading.Condition()

    def set(self, key, value):
        with self.condition:
            self.data[key] = value
            self.condition.notify_all()

    def get(self, key):
        with self.condition:
            return self.data[key]

    def check(self, keys):
        with self.condition:
            return all(key in self.data for key in keys)

    def compare_set(self, key, expected, desired):
        with self.condition:
            current = self.data.get(key, b"")
            if current == expected:
                self.data[key] = desired
                self.condition.notify_all()
                return desired
            return current

    def wait(self, keys, timeout):
        deadline = time.monotonic() + timeout.total_seconds()
        with self.condition:
            while not all(k in self.data for k in keys):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("store timeout")
                self.condition.wait(remaining)

    def add(self, key, amount):
        with self.condition:
            result = int(self.data.get(key, b"0")) + amount
            self.data[key] = str(result).encode()
            self.condition.notify_all()
            return result

    def delete_key(self, key):
        with self.condition:
            return self.data.pop(key, None) is not None


class IntraInstanceTest(unittest.TestCase):
    def tearDown(self):
        b.clear_staged_package()

    def _mock_publisher(self, env, docs, directory):
        from rtp_llm.models_py.modules.dsv41.engram_distributed import (
            publish as engram_publish,
        )

        by_index = {d["shard"]: d for d in docs}
        stub = mock.patch.object(
            engram_publish,
            "publish_shard",
            lambda checkpoint, revision, host, generation, shard, count, **kw: by_index[
                shard
            ],
        )
        stub.start()
        self.addCleanup(stub.stop)
        return env

    def test_only_rank_zero_exchanges_and_all_ranks_receive_identical_backing(self):
        env = cohort_env(2)
        config, docs = derived(env, world_size=4)
        store = MemoryStore()
        calls = []

        def exchange(cfg, local, deadline):
            calls.append(local)
            return package(cfg["group"], docs)

        with tempfile.TemporaryDirectory() as directory:
            self._mock_publisher(env, docs[:1], directory)
            envs = [dict(env) for _ in range(4)]
            with ThreadPoolExecutor(4) as pool:
                futures = [
                    pool.submit(
                        s.prepare_static_engram,
                        store,
                        rank=i,
                        world_size=4,
                        environ=envs[i],
                        leader_exchange=exchange,
                    )
                    for i in range(4)
                ]
                results = [f.result() for f in futures]
            self.assertEqual(len(calls), 1)
            self.assertTrue(all(r is None for r in results))
            self.assertFalse(
                {
                    k: v
                    for k, v in store.data.items()
                    if not k.startswith("engram/start_count/")
                }
            )
            self.assertTrue(all(item["ENGRAM_SHARD_COUNT"] == "2" for item in envs))
            staged = b.staged_package()
            self.assertIsNotNone(staged)
            self.assertEqual(staged["generation"], config["group"]["generation"])
            # Serving must not materialize any bundle files.
            self.assertEqual(list(Path(directory).glob("**/*.json")), [])

    def test_bad_p_fails_before_any_store_side_effect(self):
        store = MemoryStore()
        env = cohort_env(2)
        env[s.COUNT_ENV] = "1"
        with self.assertRaisesRegex(ValueError, "STORAGE_HOST_COUNT"):
            s.prepare_static_engram(store, rank=0, world_size=2, environ=env)
        self.assertFalse(
            {
                k: v
                for k, v in store.data.items()
                if not k.startswith("engram/start_count/")
            }
        )

    def test_inconsistent_worker_group_never_publishes_package(self):
        env = cohort_env(2, timeout=0.2)
        config, docs = derived(env, world_size=2)
        store = MemoryStore()
        prefix = "engram/" + config["group"]["generation"] + "/attempt/1/i0/"
        store.set(prefix + "config/1", b"wrong-group")
        with tempfile.TemporaryDirectory() as directory:
            self._mock_publisher(env, docs[:1], directory)
            with self.assertRaisesRegex(
                e.ExchangeError, "INSTANCE_GROUP_CONFIG_MISMATCH"
            ):
                s.prepare_static_engram(store, rank=0, world_size=2, environ=env)
        self.assertNotIn(prefix + "package", store.data)

    def test_self_published_shards_describe_their_own_host(self):
        from unittest.mock import patch

        from rtp_llm.models_py.modules.dsv41.engram_distributed import (
            publish as engram_publish,
        )

        env = cohort_env(2)
        env.pop("RTP_ENGRAM_LOCAL_DEVICE")
        env.pop("RTP_ENGRAM_LOCAL_NUMA")
        config, docs = derived(env, world_size=2)
        store = MemoryStore()
        published = []

        def publish(checkpoint, revision, host_id, generation, shard, shards, **kwargs):
            self.assertEqual(checkpoint, env[s.CHECKPOINT_ENV])
            self.assertEqual(revision, config["group"]["model_revision"])
            self.assertEqual(generation, config["group"]["generation"])
            self.assertEqual(shards, 2)
            self.assertEqual(
                host_id, "127.0.0.1"
            )  # the shard's own host, not the member's
            self.assertIsNotNone(kwargs["deadline"])
            self.assertEqual(kwargs["max_allocation_bytes"], 128 << 30)
            self.assertEqual(kwargs["budget_bytes"], 256 << 30)
            published.append((host_id, shard))
            return docs[shard]

        def exchange(cfg, local, deadline):
            return package(cfg["group"], docs)

        with patch.object(engram_publish, "publish_shard", publish), self.assertLogs(
            "rtp_llm.models_py.modules.dsv41.engram_distributed.startup", "WARNING"
        ):
            with ThreadPoolExecutor(2) as pool:
                results = [
                    pool.submit(
                        s.prepare_static_engram,
                        store,
                        rank=i,
                        world_size=2,
                        environ=dict(env),
                        leader_exchange=exchange,
                    )
                    for i in range(2)
                ]
                [result.result() for result in results]
        self.assertEqual(published, [("127.0.0.1", 0)])
        staged = b.staged_package()
        self.assertIsNotNone(staged)
        self.assertEqual(staged["generation"], config["group"]["generation"])
        self.assertFalse(
            {
                k: v
                for k, v in store.data.items()
                if not k.startswith("engram/start_count/")
            }
        )  # descriptors went through Store, not files

    def test_unconfigured_models_do_nothing(self):
        self.assertIsNone(
            s.prepare_static_engram(None, rank=0, world_size=8, environ={})
        )

    def test_publisher_assembly_and_verification_failures_wake_local_waiters(self):
        from rtp_llm.models_py.modules.dsv41.engram_distributed import publish

        secret = "/private/checkpoint secret-capability=" + "x" * 4096
        for phase in ("publish", "assembly", "verify"):
            with self.subTest(phase=phase):
                env = cohort_env(2, timeout=5)
                config, docs = derived(env, world_size=2)
                store = MemoryStore()
                original_error = FileNotFoundError(secret)

                def allocate(*args, **kwargs):
                    if phase == "publish":
                        raise original_error
                    return docs[0]

                def exchange(*args):
                    if phase == "assembly":
                        raise original_error
                    return package(config["group"], docs)

                original_verify = s.verify_package

                def verify(*args):
                    if phase == "verify":
                        raise original_error
                    return original_verify(*args)

                started = time.monotonic()
                with mock.patch.object(
                    publish, "publish_shard", allocate
                ), mock.patch.object(s, "verify_package", verify):
                    with ThreadPoolExecutor(2) as pool:
                        jobs = [
                            pool.submit(
                                s.prepare_static_engram,
                                store,
                                rank=rank,
                                world_size=2,
                                environ=dict(env),
                                leader_exchange=exchange,
                            )
                            for rank in range(2)
                        ]
                        self.assertIs(jobs[0].exception(timeout=2), original_error)
                        peer = jobs[1].exception(timeout=2)
                self.assertLess(time.monotonic() - started, 2)
                self.assertIsInstance(peer, s._StartupPeerFailure)
                self.assertEqual((peer.rank, peer.phase), (0, phase))
                self.assertEqual(peer.error_class, "file_not_found")
                self.assertEqual(peer.cause_code, "FILE_NOT_FOUND")
                self.assertNotIn("private", str(peer))
                prefix = f'engram/{config["group"]["generation"]}/attempt/1/i0/'
                payload = store.get(prefix + "failure")
                self.assertLessEqual(len(payload), 512)
                self.assertNotIn(b"secret-capability", payload)
                self.assertNotIn(prefix + "package", store.data)
                self.assertIsNone(b.staged_package())

    def test_first_failure_is_retained_even_when_success_keys_exist(self):
        store = MemoryStore()
        error = e.ExchangeError(
            "ASSEMBLY_FAILED", error_class="manifest", cause_code="OUT_OF_BOUNDS"
        )
        s._publish_failure(store, "failed", 3, "assembly", error)
        first = store.get("failed")
        s._publish_failure(store, "failed", 0, "package", RuntimeError("secondary"))
        self.assertEqual(store.get("failed"), first)
        store.set("package", b"ready")
        with self.assertRaises(s._StartupPeerFailure) as raised:
            s._wait(store, ["package"], time.monotonic() + 1, failure_key="failed")
        self.assertEqual(raised.exception.rank, 3)
        self.assertEqual(raised.exception.cause_code, "OUT_OF_BOUNDS")

    def test_failure_reporting_cannot_replace_original_error(self):
        store = MemoryStore()
        original = FileNotFoundError("private filename")
        with mock.patch.object(
            store, "compare_set", side_effect=RuntimeError("store disconnected")
        ):
            try:
                try:
                    raise original
                except Exception as error:
                    s._publish_failure(store, "failed", 0, "publish", error)
                    raise
            except FileNotFoundError as actual:
                self.assertIs(actual, original)
        self.assertFalse(store.check(["failed"]))

    def test_failure_diagnostics_are_bounded_and_sanitized(self):
        store = MemoryStore()
        error = e.ExchangeError("secret-capability")
        # A caller can mutate exception attributes after construction; the
        # sender must still apply the shared allowlist at its own boundary.
        error.error_class = "private filename"
        error.cause_code = "secret-capability"
        s._publish_failure(store, "failed", 0, "publish", error)
        self.assertNotIn(b"private", store.get("failed"))
        self.assertNotIn(b"secret-capability", store.get("failed"))
        for payload in (
            b"x" * 513,
            e.encode(
                dict(rank=True, phase="publish", error_class=None, cause_code=None)
            ),
            e.encode(
                dict(rank=0, phase="private-path", error_class=None, cause_code=None)
            ),
        ):
            with self.subTest(payload=payload[:30]):
                store.set("malformed", payload)
                with self.assertRaisesRegex(
                    e.ExchangeError, "INVALID_INSTANCE_FAILURE"
                ):
                    s._wait(
                        store, ["never"], time.monotonic() + 1, failure_key="malformed"
                    )


class MutualTLSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.models_py.modules.dsv41.engram_distributed.test.make_exchange_test_certs import (
            create,
        )

        cls.temp = tempfile.TemporaryDirectory()
        cls.tls = create(Path(cls.temp.name) / "pki", ["i0", "i1", "i2"])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_two_remote_instances_reuse_exactly_two_connections(self):
        group, docs = fixture(3, 3, 1)
        cohort = e.Cohort(group, lambda d: package(group, d), 10)
        server = e.ExchangeServer(
            ("127.0.0.1", 0),
            cohort,
            e.tls_context(self.tls["i0"], True),
            self.tls["i0"]["instance_cert_sha256"],
        )
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        thread.start()
        try:
            with ThreadPoolExecutor(3) as pool:
                futures = [
                    pool.submit(
                        e.remote_exchange,
                        server.server_address,
                        e.tls_context(self.tls[f"i{i}"]),
                        f"i{i}",
                        cohort.group_digest,
                        [docs[i]],
                        time.monotonic() + 8,
                        lambda p: s.verify_package(group, p),
                        self.tls["i0"]["instance_cert_sha256"]["i0"],
                    )
                    for i in (1, 2)
                ]
                payload = cohort.submit("i0", cohort.group_digest, [docs[0]])
                cohort.ack("i0", e.digest(payload))
                cohort.wait_acknowledged()
                results = [f.result() for f in futures]
            self.assertEqual(results[0], results[1])
            self.assertEqual(server.connections, 2)  # exchange + ACK on each connection
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_valid_ca_but_wrong_coordinator_certificate_rejected(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: {}, 3)
        server = e.ExchangeServer(
            ("127.0.0.1", 0),
            cohort,
            e.tls_context(self.tls["i0"], True),
            self.tls["i0"]["instance_cert_sha256"],
        )
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        thread.start()
        try:
            with self.assertRaisesRegex(
                e.ExchangeError, "COORDINATOR_CERTIFICATE_MISMATCH"
            ):
                e.remote_exchange(
                    server.server_address,
                    e.tls_context(self.tls["i1"]),
                    "i1",
                    cohort.group_digest,
                    [docs[1]],
                    time.monotonic() + 2,
                    lambda p: None,
                    self.tls["i0"]["instance_cert_sha256"]["i2"],
                )
            self.assertEqual(cohort.documents, {})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_certificate_cannot_claim_another_instance(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: {}, 3)
        server = e.ExchangeServer(
            ("127.0.0.1", 0),
            cohort,
            e.tls_context(self.tls["i0"], True),
            self.tls["i0"]["instance_cert_sha256"],
        )
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        thread.start()
        try:
            with self.assertRaisesRegex(e.ExchangeError, "UNAUTHORIZED_INSTANCE"):
                e.remote_exchange(
                    server.server_address,
                    e.tls_context(self.tls["i2"]),
                    "i1",
                    cohort.group_digest,
                    [docs[1]],
                    time.monotonic() + 2,
                    lambda p: None,
                )
            self.assertEqual(cohort.documents, {})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class RankSpaceTest(unittest.TestCase):
    """A member is either a whole gang or one host of a gang shared by members."""

    def test_shared_gang_members_are_member_local(self):
        env = cohort_env(p=2, instances=2, index=1)
        env[s.RANKS_ENV] = "4"
        group = s.build_group(env, 8)["group"]
        self.assertEqual([m["world_size"] for m in group["members"]], [4, 4])
        self.assertEqual(
            [sh["publisher_rank"] for m in group["members"] for sh in m["shards"]],
            [0, 0],
        )

    def test_whole_gang_members_keep_global_ranks(self):
        env = cohort_env(p=2, instances=2, index=1)
        group = s.build_group(env, 4)["group"]
        self.assertEqual([m["world_size"] for m in group["members"]], [4, 4])
        self.assertEqual(
            [sh["publisher_rank"] for m in group["members"] for sh in m["shards"]],
            [0, 0],
        )

    def test_ranks_per_instance_must_describe_the_gang(self):
        env = cohort_env(p=2, instances=2, index=1)
        env[s.RANKS_ENV] = "3"
        with self.assertRaises(e.ExchangeError):
            s.build_group(env, 8)


class DeploymentIdentityTest(unittest.TestCase):
    def AssertUuid(self, value):
        self.assertEqual(str(UUID(value)), value)

    def test_physical_p_matches_explicit_hosts(self):
        for p in (2, 3, 5, 18):
            env = cohort_env(p, instances=p)
            group = s.build_group(env, 4)["group"]
            self.assertEqual(
                [m["shards"][0]["host_id"] for m in group["members"]],
                [f"127.0.0.{i+1}" for i in range(p)],
            )
        with self.assertRaisesRegex(e.ExchangeError, "MUST_MATCH_COHORT"):
            s.build_group(cohort_env(4, instances=2), 8)

    def test_bare_hosts_take_the_shared_port(self):
        explicit = s.build_group(cohort_env(2), 4)["group"]
        env = cohort_env(2)
        env[s.COHORT_ENV] = "127.0.0.1,127.0.0.2"
        env[s.PORT_ENV] = "29500"
        self.assertEqual(s.build_group(env, 4)["group"], explicit)
        del env[s.PORT_ENV]
        defaulted = s.build_group(env, 4)["group"]
        self.assertEqual(defaulted["members"][0]["shards"][0]["host_id"], "127.0.0.1")
        self.assertEqual(
            s.build_group(dict(env, **{s.PORT_ENV: "29500"}), 4)["group"], explicit
        )

    def test_instance_index_is_discovered_from_the_local_address(self):
        env = cohort_env(2)
        del env[s.INDEX_ENV]
        # Do not assume the test host's hostname resolves to loopback.
        with mock.patch.object(s, "_local_addresses", lambda: {"127.0.0.1"}):
            self.assertEqual(s.build_group(env, 4)["index"], 0)
        with mock.patch.object(s, "_local_addresses", lambda: {"127.0.0.2"}):
            self.assertEqual(s.build_group(env, 4)["index"], 1)
        with mock.patch.object(s, "_local_addresses", lambda: {"10.9.9.9"}):
            with self.assertRaisesRegex(e.ExchangeError, "INSTANCE_INDEX_REQUIRED"):
                s.build_group(env, 4)
        with mock.patch.object(
            s, "_local_addresses", lambda: {"127.0.0.1", "127.0.0.2"}
        ):
            with self.assertRaisesRegex(e.ExchangeError, "AMBIGUOUS_LOCAL_ADDRESS"):
                s.build_group(env, 4)

    def test_generation_is_not_a_user_input(self):
        env = cohort_env(2)
        expected = s.build_group(env, 4)["group"]
        env["RTP_ENGRAM_GENERATION"] = "ignored-legacy-setting"
        self.assertEqual(s.build_group(env, 4)["group"], expected)
        env[s.COHORT_ENV] = "127.0.0.1:10001,127.0.0.1:10002"
        with self.assertRaisesRegex(e.ExchangeError, "DISTINCT_STORAGE_HOST"):
            s.build_group(env, 4)

    def test_wait_uses_bounded_operation_not_blocking_get(self):
        class Store:
            def get(self, key):
                raise AssertionError("unbounded get must not implement wait")

            def wait(self, keys, timeout):
                self.keys, self.timeout = keys, timeout.total_seconds()
                raise RuntimeError("timeout")

        store = Store()
        with self.assertRaisesRegex(e.ExchangeError, "WAIT_FAILED_OR_TIMEOUT"):
            s._wait(store, ["a", "b"], time.monotonic() + 0.1)
        self.assertEqual(store.keys, ["a", "b"])
        self.assertTrue(0 < store.timeout <= 0.1)


class TopologySourceTest(unittest.TestCase):
    def test_reuse_gang_topology_without_store_or_hostname_lookup(self):
        env = {s.COUNT_ENV: "2", s.REVISION_ENV: "rev", s.CHECKPOINT_ENV: "/model"}
        with mock.patch.object(
            s, "_local_addresses", side_effect=AssertionError("must reuse RTP IP")
        ):
            cfg = s.build_group(
                env, 4, rank=2, gang_hosts=["10.0.0.1"] * 2 + ["10.0.0.2"] * 2
            )
        self.assertEqual(cfg["index"], 1)
        self.assertEqual(cfg["rank_base"], 2)
        self.assertTrue(cfg["single_gang"])
        self.assertEqual([m["world_size"] for m in cfg["group"]["members"]], [2, 2])

    def test_explicit_list_does_not_hide_multihost_gang(self):
        env = cohort_env(4, instances=4, index=3)
        # Second of two independent 2-host instances: local ranks start at zero.
        cfg = s.build_group(
            env, 8, rank=5, gang_hosts=["127.0.0.3"] * 4 + ["127.0.0.4"] * 4
        )
        self.assertEqual(cfg["rank_base"], 4)
        self.assertEqual(cfg["group"]["members"][3]["world_size"], 4)
        self.assertFalse(cfg["single_gang"])
        env[s.INDEX_ENV] = "2"
        with self.assertRaisesRegex(e.ExchangeError, "INDEX_HOST_MISMATCH"):
            s.build_group(
                env, 8, rank=5, gang_hosts=["127.0.0.3"] * 4 + ["127.0.0.4"] * 4
            )

    def test_invalid_rank_layouts_fail_before_allocation(self):
        for hosts, error in [
            (["a", "a", "b"], "UNEVEN"),
            (["a", "b", "a", "b"], "NONCONTIGUOUS"),
        ]:
            with self.assertRaisesRegex(e.ExchangeError, error):
                s._gang_parts(hosts, len(hosts))
        with self.assertRaisesRegex(e.ExchangeError, "INVALID_GANG"):
            s._gang_parts(["a"], 4)
        env = cohort_env(2)
        env[s.RANKS_ENV] = "4"
        with self.assertRaisesRegex(e.ExchangeError, "RANKS_PER_INSTANCE_MISMATCH"):
            s.build_group(env, 4, gang_hosts=["127.0.0.1"] * 2 + ["127.0.0.2"] * 2)

    def test_platform_directory_is_topology_not_an_enable_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "hosts").write_text("10.0.0.1\n10.0.0.2\n")
            env = {s.GANGINFO_ENV: directory}
            self.assertIsNone(s.build_group(env, 4))
            env.update(
                {
                    s.COUNT_ENV: "2",
                    s.INDEX_ENV: "0",
                    s.REVISION_ENV: "rev",
                    s.CHECKPOINT_ENV: "/model",
                }
            )
            cfg = s.build_group(env, 4, gang_hosts=["10.0.0.1"] * 2 + ["10.0.0.2"] * 2)
            self.assertTrue(cfg["single_gang"])
            env[s.COHORT_ENV] = "127.0.0.1,127.0.0.2"
            cfg = s.build_group(env, 4)
            self.assertEqual(cfg["coordinator"]["host"], "127.0.0.1")
        with self.assertRaisesRegex(e.ExchangeError, "INVALID_GANGINFO"):
            s._ganginfo_cohort({s.GANGINFO_ENV: "/nonexistent/ganginfo"})
        self.assertIsNone(s._ganginfo_cohort({}))

    def test_single_gang_needs_no_tls_files_and_legacy_descriptor_mode_is_removed(self):
        env = cohort_env(2)
        env[s.TLS_DIR_ENV] = "/not-present/pki"
        hosts = ["127.0.0.1", "127.0.0.2"]
        self.assertIsNone(s.build_config(env, 2, gang_hosts=hosts)["tls"])
        env["RTP_ENGRAM_LOCAL_DESCRIPTORS"] = "/obsolete/shard.json"
        with self.assertRaisesRegex(e.ExchangeError, "REMOVED_LOCAL_DESCRIPTOR"):
            s.build_config(env, 2, gang_hosts=hosts)

    def test_missing_framework_topology_does_not_guess(self):
        env = {s.COUNT_ENV: "2"}
        with self.assertRaisesRegex(e.ExchangeError, "GANG_TOPOLOGY_REQUIRED"):
            s.build_group(env, 4)
        with self.assertRaisesRegex(e.ExchangeError, "COHORT_REQUIRED"):
            s.build_group(env, 4, gang_hosts=["a"] * 4)

    def _annotation(self, directory, parts):
        payload = {name: {"name": name, "ip": ip} for name, ip in parts}
        body = json.dumps(payload).replace('"', '\\"')
        path = Path(directory) / "annotations"
        path.write_text('app.c2.io/biz-detail-ganginfo="%s"\n' % body)
        return str(path)

    def test_gang_annotation_discovers_single_host_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._annotation(
                directory,
                [
                    ("prefill_part0", "10.0.0.1"),
                    ("prefill_part1", "10.0.0.2"),
                    ("prefill_part2", "10.0.0.3"),
                    ("prefill_part3", "10.0.0.4"),
                ],
            )
            env = {
                s.COUNT_ENV: "4",
                s.REVISION_ENV: "rev",
                s.CHECKPOINT_ENV: "/model",
                s.ANNOTATION_ENV: path,
            }
            cfg = s.build_group(env, 4, rank=0, gang_hosts=["10.0.0.1"] * 4)
            self.assertFalse(cfg["single_gang"])
            self.assertEqual(cfg["index"], 0)
            self.assertEqual(
                cfg["coordinator"], {"host": "10.0.0.1", "port": s.DEFAULT_COHORT_PORT}
            )
            self.assertEqual(
                [m["shards"][0]["host_id"] for m in cfg["group"]["members"]],
                ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
            )
            with self.assertLogs(s.__name__, level="INFO") as logs:
                s.build_group(env, 4, rank=0, gang_hosts=["10.0.0.1"] * 4)
            self.assertIn(
                "source=annotation instances=4 p=4 index=0 "
                "coordinator=10.0.0.1:%d local=10.0.0.1" % s.DEFAULT_COHORT_PORT,
                logs.output[0],
            )
            # Every instance keeps its own rank space; only the address differs.
            cfg = s.build_group(env, 4, rank=0, gang_hosts=["10.0.0.4"] * 4)
            self.assertEqual(cfg["index"], 3)
            self.assertEqual(cfg["rank_base"], 0)
            env[s.PORT_ENV] = "29999"
            cfg = s.build_group(env, 4, rank=0, gang_hosts=["10.0.0.1"] * 4)
            self.assertEqual(cfg["coordinator"]["port"], 29999)

    def test_gang_annotation_keeps_single_instance_multihost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._annotation(
                directory, [("decode_part0", "10.0.0.1"), ("decode_part1", "10.0.0.2")]
            )
            env = {
                s.COUNT_ENV: "2",
                s.REVISION_ENV: "rev",
                s.CHECKPOINT_ENV: "/model",
                s.ANNOTATION_ENV: path,
            }
            cfg = s.build_group(
                env, 8, rank=5, gang_hosts=["10.0.0.1"] * 4 + ["10.0.0.2"] * 4
            )
            self.assertTrue(cfg["single_gang"])
            self.assertEqual(cfg["index"], 1)
            self.assertEqual(cfg["rank_base"], 4)
            self.assertEqual(
                [m["shards"][0]["host_id"] for m in cfg["group"]["members"]],
                ["10.0.0.1", "10.0.0.2"],
            )

    def test_gang_annotation_leaves_multihost_instances_alone(self):
        # A 4-pod gang of two 2-host instances keeps the per-instance layout:
        # the annotation cannot say how pods group, so it must not be guessed.
        with tempfile.TemporaryDirectory() as directory:
            path = self._annotation(
                directory,
                [("prefill_part%d" % i, "10.0.0.%d" % (i + 1)) for i in range(4)],
            )
            env = {
                s.COUNT_ENV: "2",
                s.REVISION_ENV: "rev",
                s.CHECKPOINT_ENV: "/model",
                s.ANNOTATION_ENV: path,
            }
            cfg = s.build_group(
                env, 8, rank=0, gang_hosts=["10.0.0.1"] * 4 + ["10.0.0.2"] * 4
            )
            self.assertTrue(cfg["single_gang"])
            self.assertEqual(
                [m["shards"][0]["host_id"] for m in cfg["group"]["members"]],
                ["10.0.0.1", "10.0.0.2"],
            )

    def test_gang_annotation_must_list_the_local_host(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._annotation(
                directory,
                [("prefill_part0", "10.0.0.1"), ("prefill_part1", "10.0.0.2")],
            )
            env = {
                s.COUNT_ENV: "2",
                s.REVISION_ENV: "rev",
                s.CHECKPOINT_ENV: "/model",
                s.ANNOTATION_ENV: path,
            }
            with self.assertRaisesRegex(e.ExchangeError, "LOCAL_HOST_NOT_IN_GANGINFO"):
                s.build_group(env, 4, gang_hosts=["10.0.0.9"] * 4)

    def test_gang_annotation_is_optional_but_not_ignored_when_configured(self):
        env = {s.COUNT_ENV: "2", s.REVISION_ENV: "rev", s.CHECKPOINT_ENV: "/model"}
        with mock.patch.object(s, "DEFAULT_ANNOTATION", "/nonexistent/annotations"):
            with self.assertRaisesRegex(e.ExchangeError, "COHORT_REQUIRED"):
                s.build_group(env, 4, gang_hosts=["10.0.0.1"] * 4)
        env[s.ANNOTATION_ENV] = "/nonexistent/annotations"
        with self.assertRaisesRegex(e.ExchangeError, "INVALID_GANGINFO"):
            s.build_group(env, 4, gang_hosts=["10.0.0.1"] * 4)

    def test_resolved_checkpoint_wins_over_an_object_store_uri(self):
        # Whale passes CHECKPOINT_PATH as oss://; the launcher resolves it to a
        # local fuse mount before the engine starts, and that is what we read.
        env = {
            s.COUNT_ENV: "2",
            s.REVISION_ENV: "rev",
            "CHECKPOINT_PATH": "oss://bucket/model/?OSS_ENDPOINT=internal",
        }
        hosts = ["10.0.0.1"] * 2 + ["10.0.0.2"] * 2
        cfg = s.build_group(env, 4, gang_hosts=hosts, checkpoint="/mnt/fuse/abc")
        self.assertEqual(cfg["checkpoint"], "/mnt/fuse/abc")
        env[s.CHECKPOINT_ENV] = "/explicit"
        cfg = s.build_group(env, 4, gang_hosts=hosts, checkpoint="/mnt/fuse/abc")
        self.assertEqual(cfg["checkpoint"], "/explicit")

    def test_explicit_cohort_still_overrides_the_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._annotation(
                directory,
                [("prefill_part%d" % i, "10.0.0.%d" % (i + 1)) for i in range(4)],
            )
            env = {
                s.COUNT_ENV: "2",
                s.REVISION_ENV: "rev",
                s.CHECKPOINT_ENV: "/model",
                s.ANNOTATION_ENV: path,
                s.COHORT_ENV: "127.0.0.1,127.0.0.2",
            }
            cfg = s.build_group(env, 4, gang_hosts=["127.0.0.1"] * 4)
            self.assertEqual(cfg["coordinator"]["host"], "127.0.0.1")
            self.assertEqual(len(cfg["group"]["members"]), 2)


class RevisionTest(unittest.TestCase):
    def test_revision_is_derived_from_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "model.safetensors.index.json"
            index.write_text('{"weight_map": {"a": "a.safetensors"}}')
            env = cohort_env(2)
            del env[s.REVISION_ENV]
            env[s.CHECKPOINT_ENV] = directory
            first = s.build_group(env, 4)["group"]["model_revision"]
            self.assertTrue(first.startswith("ckpt@"))
            # Identical weights give identical identity, wherever they are mounted.
            other = cohort_env(2)
            del other[s.REVISION_ENV]
            other[s.CHECKPOINT_ENV] = directory
            self.assertEqual(first, s.build_group(other, 4)["group"]["model_revision"])
            index.write_text('{"weight_map": {"a": "b.safetensors"}}')
            self.assertNotEqual(first, s.build_group(env, 4)["group"]["model_revision"])
            # An explicit revision still wins, and a checkpoint without either
            # index is refused rather than given an accidental identity.
            explicit = dict(env, **{s.REVISION_ENV: "release-1019"})
            self.assertEqual(
                s.build_group(explicit, 4)["group"]["model_revision"], "release-1019"
            )
            empty = Path(directory) / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(e.ExchangeError, "MODEL_REVISION_REQUIRED"):
                s.build_group(dict(env, **{s.CHECKPOINT_ENV: str(empty)}), 4)


class StartupScopeTest(unittest.TestCase):
    def tearDown(self):
        b.clear_staged_package()

    def test_single_gang_uses_store_only_and_does_not_read_stale_package(self):
        from rtp_llm.models_py.modules.dsv41.engram_distributed import (
            publish as engram_publish,
        )

        hosts = ["127.0.0.1"] * 2 + ["127.0.0.2"] * 2
        env = cohort_env(2)
        env.pop(s.COHORT_ENV)
        env.pop(s.INDEX_ENV)
        config = s.build_group(env, 4, gang_hosts=hosts)
        _, docs = derived(cohort_env(2), world_size=2)
        for d in docs:
            d["generation"] = config["group"]["generation"]
        store = MemoryStore()
        for rank in range(4):
            store.add(f"engram/start_count/{rank}", 1)
        old = f'engram/{config["group"]["generation"]}/attempt/1/gang/package'
        store.set(old, b"old invalid capability package")
        old_failure = old.rsplit("/", 1)[0] + "/failure"
        s._publish_failure(
            store, old_failure, 0, "publish", RuntimeError("old failure")
        )
        published = []

        def publish(checkpoint, revision, host_id, generation, shard, shards, **kw):
            published.append(shard)
            return docs[shard]

        def no_rpc(*a):
            raise AssertionError("one gang must not open a new connection")

        with mock.patch.object(
            engram_publish, "publish_shard", publish
        ), mock.patch.object(s, "checkpoint_samples", samples):
            # Default argument binds checkpoint_samples at definition time.
            original = s.assemble_package
            with mock.patch.object(
                s,
                "assemble_package",
                lambda g, d, c: original(g, d, c, samples=samples),
            ):
                with ThreadPoolExecutor(4) as pool:
                    jobs = [
                        pool.submit(
                            s.prepare_static_engram,
                            store,
                            rank=r,
                            world_size=4,
                            environ=dict(env),
                            gang_hosts=hosts,
                            leader_exchange=no_rpc,
                        )
                        for r in range(4)
                    ]
                    [job.result() for job in jobs]
        self.assertEqual(sorted(published), [0, 1])
        self.assertIsNotNone(b.staged_package())
        self.assertEqual(store.get(old), b"old invalid capability package")
        self.assertTrue(store.check([old_failure]))
        self.assertFalse(any("/attempt/2/" in k for k in store.data))
        self.assertTrue(
            all(store.get(f"engram/start_count/{r}") == b"2" for r in range(4))
        )

    def test_partial_restart_cannot_read_previous_attempt(self):
        env = cohort_env(2, timeout=0.04)
        cfg = s.build_group(env, 2)
        store = MemoryStore()
        store.add("engram/start_count/1", 1)
        old = f'engram/{cfg["group"]["generation"]}/attempt/1/i0/package'
        store.set(old, b"stale")
        with self.assertRaisesRegex(e.ExchangeError, "TIMEOUT"):
            s.prepare_static_engram(store, rank=1, world_size=2, environ=env)
        self.assertIsNone(b.staged_package())

    def test_two_multihost_instances_each_publish_their_local_shards(self):
        from rtp_llm.models_py.modules.dsv41.engram_distributed import (
            publish as engram_publish,
        )

        base = cohort_env(4, instances=4)
        base.pop(s.INDEX_ENV)
        group = s.build_group(
            base, 4, gang_hosts=["127.0.0.1"] * 2 + ["127.0.0.2"] * 2
        )["group"]
        _, docs = derived(cohort_env(4, instances=4), world_size=2)
        cohort = e.Cohort(group, lambda d: package(group, d), 3)
        stores = [MemoryStore(), MemoryStore()]
        published, calls = [], []

        def publish(checkpoint, revision, host_id, generation, shard, shards, **kw):
            self.assertEqual(host_id, docs[shard]["host_id"])
            self.assertEqual(generation, docs[shard]["generation"])
            published.append(shard)
            return docs[shard]

        def exchange(cfg, local, deadline):
            calls.append(cfg["instance_id"])
            return e.decode(
                cohort.submit(cfg["instance_id"], e.validate_group(cfg["group"]), local)
            )["package"]

        with mock.patch.object(engram_publish, "publish_shard", publish):
            with ThreadPoolExecutor(8) as pool:
                jobs = []
                for gang in range(2):
                    hosts = [f"127.0.0.{2*gang+1}"] * 2 + [f"127.0.0.{2*gang+2}"] * 2
                    for rank in range(4):
                        jobs.append(
                            pool.submit(
                                s.prepare_static_engram,
                                stores[gang],
                                rank=rank,
                                world_size=4,
                                environ=dict(base),
                                gang_hosts=hosts,
                                leader_exchange=exchange,
                            )
                        )
                [job.result() for job in jobs]
        self.assertEqual(sorted(published), list(range(4)))
        self.assertEqual(sorted(calls), ["i0", "i1", "i2", "i3"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
