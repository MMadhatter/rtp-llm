# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU tests; no RTP package imports or CUDA dependencies."""
import concurrent.futures
import dataclasses
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

_PATH = Path(__file__).resolve().parents[1] / "engram_sessions.py"
_SPEC = importlib.util.spec_from_file_location("engram_sessions_under_test", _PATH)
m = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = m
_SPEC.loader.exec_module(m)


def publication(dataset="dataset-a", generation="generation-a", digest="digest-a", state="READY",
                allocation_ids=None):
    # Test double of S1a ValidatedManifest. Registry deliberately does not rerun
    # schema validation; real callers must supply validate_for_attach's result.
    if allocation_ids is None:
        suffix = "" if generation == "generation-a" else "-" + generation
        allocation_ids = ("a1" + suffix, "a1" + suffix, "a2" + suffix)
    tensors = [SimpleNamespace(shards=[SimpleNamespace(allocation_id=a)]) for a in allocation_ids]
    return SimpleNamespace(manifest=SimpleNamespace(dataset_id=dataset, generation=generation,
                                                    state=state, tensors=tensors), manifest_digest=digest)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class SessionsTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.r = m.GenerationRegistry(lease_seconds=10, clock=self.clock)
        self.r.publish(publication())

    def acquire(self, request="request-a", **overrides):
        args = dict(dataset_id="dataset-a", generation="generation-a", request_id=request,
                    client_id="client-a", worker_incarnation="worker-a", device_ids=("gpu-1", "gpu-0"))
        args.update(overrides)
        return self.r.acquire(**args)

    def ack(self, s, **overrides):
        args = dict(session_id=s.session_id, client_id=s.client_id, worker_incarnation=s.worker_incarnation,
                    manifest_digest="digest-a", allocation_ids=("a1", "a2"))
        args.update(overrides)
        return self.r.attach_ack(**args)

    def release(self, s, **overrides):
        args = dict(session_id=s.session_id, client_id=s.client_id,
                    worker_incarnation=s.worker_incarnation, quiesced=True)
        args.update(overrides)
        return self.r.release(**args)

    def assert_error(self, code, function, *args, **kwargs):
        with self.assertRaises(m.SessionError) as error:
            function(*args, **kwargs)
        self.assertEqual(error.exception.code, code)

    def test_full_lifecycle_and_frozen_snapshots(self):
        s = self.acquire()
        self.assertEqual(s.state, "PENDING")
        self.assertEqual(s.device_ids, ("gpu-0", "gpu-1"))
        self.assertEqual(self.ack(s).state, "ACTIVE")
        self.assertEqual(self.r.drain(s.generation).live_sessions, 1)
        self.assertEqual(self.release(s).state, "RELEASED")
        self.assertEqual(self.r.retire(s.generation).state, "RETIRED")
        self.assertEqual(self.r.retire(s.generation).live_sessions, 0)
        self.assertEqual(self.r.drain(s.generation).state, "RETIRED")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.state = "ACTIVE"

    def test_publish_idempotency_conflict_and_no_reopen(self):
        self.assertEqual(self.r.publish(publication()).allocation_ids, frozenset(("a1", "a2")))
        for p in (publication(dataset="different"), publication(digest="different")):
            self.assert_error("PUBLICATION_CONFLICT", self.r.publish, p)
        self.assert_error("PUBLICATION_NOT_READY", self.r.publish, publication(state="LOADING"))
        p = publication()
        p.manifest.tensors.append(SimpleNamespace(shards=[SimpleNamespace(allocation_id="a3")]))
        self.assert_error("PUBLICATION_CONFLICT", self.r.publish, p)
        self.r.drain("generation-a")
        self.r.retire("generation-a")
        self.assertEqual(self.r.publish(publication()).state, "RETIRED")
        self.assert_error("GENERATION_NOT_READY", self.acquire)

    def test_acquire_idempotency_conflict_and_tombstone(self):
        s = self.acquire()
        self.assertEqual(self.acquire(device_ids=("gpu-0", "gpu-1")).session_id, s.session_id)
        for change in (dict(dataset_id="other"), dict(generation="other"), dict(client_id="other"),
                       dict(worker_incarnation="other"), dict(device_ids=("gpu-2",))):
            self.assert_error("IDEMPOTENCY_CONFLICT", self.acquire, **change)
        self.r.drain(s.generation)
        self.assertEqual(self.acquire().session_id, s.session_id)
        self.assert_error("GENERATION_NOT_READY", self.acquire, "new-request")
        self.release(s)
        self.assertEqual(self.acquire().state, "RELEASED")
        self.assertEqual(self.r.get_generation(s.generation).live_sessions, 0)

    def test_attach_metadata_and_state_failures_preserve_reference(self):
        s = self.acquire()
        for change in (dict(manifest_digest="wrong"), dict(allocation_ids=("a1",)),
                       dict(allocation_ids=("a1", "a2", "a3"))):
            self.assert_error("ATTACH_METADATA_MISMATCH", self.ack, s, **change)
        self.assert_error("DUPLICATE_ID", self.ack, s, allocation_ids=("a1", "a1"))
        self.r.drain(s.generation)
        self.assert_error("GENERATION_NOT_READY", self.ack, s)
        self.assertEqual(self.r.get_generation(s.generation).live_sessions, 1)
        self.release(s)
        self.assert_error("SESSION_RELEASED", self.ack, s)

    def test_lost_ack_retry_during_drain_does_not_reopen_generation(self):
        s = self.ack(self.acquire())
        self.r.drain(s.generation)
        retry = self.ack(s)
        self.assertEqual((retry.state, retry.generation_state), ("ACTIVE", "DRAINING"))

    def test_expiry_and_disconnect_never_release_or_retire(self):
        pending = self.acquire()
        active = self.ack(self.acquire("request-b"))
        self.clock.now = 110.0
        self.assertEqual(len(self.r.expire_sessions()), 2)
        self.assertEqual(self.r.expire_sessions(), ())
        self.assertEqual(self.r.get_session(pending.session_id).state, "SUSPECT")
        self.assertEqual(self.r.get_generation(pending.generation).live_sessions, 2)
        self.assert_error("SESSION_SUSPECT", self.ack, pending)
        renewed = self.r.renew(pending.session_id, pending.client_id, pending.worker_incarnation)
        self.assertEqual((renewed.state, renewed.lease_deadline), ("PENDING", 120.0))
        renewed = self.r.renew(active.session_id, active.client_id, active.worker_incarnation)
        self.assertEqual(renewed.state, "ACTIVE")
        self.assertEqual(self.r.mark_disconnected(active.session_id).state, "SUSPECT")
        self.r.drain(active.generation)
        self.assert_error("LIVE_SESSIONS", self.r.retire, active.generation)
        self.release(pending)
        self.release(active)
        self.assertEqual(self.r.retire(active.generation).state, "RETIRED")

    def test_lease_checked_without_monitor_and_renew_during_drain(self):
        s = self.acquire()
        self.clock.now = 110.0
        self.assertEqual(self.acquire().state, "SUSPECT")
        self.r.drain(s.generation)
        renewed = self.r.renew(s.session_id, s.client_id, s.worker_incarnation)
        self.assertEqual((renewed.state, renewed.generation_state), ("PENDING", "DRAINING"))
        self.assert_error("GENERATION_NOT_READY", self.ack, s)

    def test_release_requires_declaration_is_idempotent_and_stays_released(self):
        s = self.acquire()
        for value in (False, None, 1):
            self.assert_error("QUIESCENCE_REQUIRED", self.release, s, quiesced=value)
        self.assertEqual(self.r.get_generation(s.generation).live_sessions, 1)
        self.assertEqual(self.release(s), self.release(s))
        self.assertEqual(self.r.mark_disconnected(s.session_id).state, "RELEASED")
        self.clock.now = 1000
        self.assertEqual(self.r.expire_sessions(), ())
        self.assert_error("SESSION_RELEASED", self.r.renew, s.session_id, s.client_id, s.worker_incarnation)

    def test_wrong_identity_cannot_mutate_a_session(self):
        s = self.acquire()
        self.assert_error("SESSION_IDENTITY_MISMATCH", self.ack, s, client_id="other")
        self.assert_error("SESSION_IDENTITY_MISMATCH", self.release, s, worker_incarnation="other")
        self.assert_error("SESSION_IDENTITY_MISMATCH", self.r.renew, s.session_id, "other", s.worker_incarnation)
        self.assertEqual(self.r.get_session(s.session_id).state, "PENDING")

    def test_missing_identity_cannot_bypass_client_checks(self):
        s = self.acquire()
        for value in (None, "", False):
            self.assert_error("INVALID_ARGUMENT", self.ack, s, client_id=value)
            self.assert_error("INVALID_ARGUMENT", self.release, s, client_id=value)
            self.assert_error("INVALID_ARGUMENT", self.r.renew, s.session_id, value, s.worker_incarnation)
            self.assert_error("INVALID_ARGUMENT", self.r.renew, s.session_id, s.client_id, value)
        self.assertEqual(self.r.get_generation(s.generation).live_sessions, 1)

    def test_unknown_keys_and_wrong_dataset(self):
        self.assert_error("UNKNOWN_GENERATION", self.r.get_generation, "unknown")
        self.assert_error("UNKNOWN_GENERATION", self.r.drain, "unknown")
        self.assert_error("UNKNOWN_GENERATION", self.r.retire, "unknown")
        self.assert_error("UNKNOWN_GENERATION", self.acquire, generation="unknown")
        self.assert_error("DATASET_MISMATCH", self.acquire, dataset_id="wrong")
        self.assert_error("UNKNOWN_SESSION", self.r.get_session, "unknown")
        self.assert_error("UNKNOWN_SESSION", self.r.mark_disconnected, "unknown")
        self.assert_error("UNKNOWN_SESSION", self.r.renew, "unknown", "client", "worker")
        self.assert_error("UNKNOWN_SESSION", self.r.release, "unknown", "client", "worker", quiesced=True)
        self.assert_error("UNKNOWN_SESSION", self.r.attach_ack, "unknown", "client", "worker", "digest-a", ("a1", "a2"))
        self.assert_error("DRAIN_REQUIRED", self.r.retire, "generation-a")

    def test_caps_preserve_tombstones_and_all_existing_references(self):
        self.r = m.GenerationRegistry(clock=self.clock, max_sessions=1, max_generations=1)
        self.r.publish(publication())
        s = self.acquire()
        self.assert_error("SESSION_LIMIT", self.acquire, "new")
        self.release(s)
        self.assert_error("SESSION_LIMIT", self.acquire, "new")
        self.assertEqual(self.acquire().state, "RELEASED")
        self.r.drain(s.generation)
        self.r.retire(s.generation)
        self.assert_error("GENERATION_LIMIT", self.r.publish, publication(generation="new"))
        self.assertEqual(self.r.publish(publication()).state, "RETIRED")

    def test_invalid_arguments(self):
        for value in (False, 0, -1, float("nan"), float("inf"), "60"):
            self.assert_error("INVALID_ARGUMENT", m.GenerationRegistry, lease_seconds=value)
        for kwargs in (dict(max_sessions=0), dict(max_generations=True), dict(clock=None)):
            self.assert_error("INVALID_ARGUMENT", m.GenerationRegistry, **kwargs)
        for change in (dict(request_id=""), dict(client_id="bad\nname"), dict(device_ids=()),
                       dict(device_ids="gpu0"), dict(device_ids=(False,)), dict(device_ids=("x", "x"))):
            code = "DUPLICATE_ID" if change == dict(device_ids=("x", "x")) else "INVALID_ARGUMENT"
            self.assert_error(code, self.acquire, **change)

    def test_two_generations_have_independent_drain_and_reference_counts(self):
        self.r.publish(publication(generation="generation-b", digest="digest-b"))
        a = self.acquire()
        b = self.acquire("request-b", generation="generation-b")
        self.r.drain(a.generation)
        self.assertEqual(self.r.get_generation(b.generation).state, "READY")
        self.release(a)
        self.r.retire(a.generation)
        self.assertEqual(self.r.get_generation(b.generation).live_sessions, 1)

    def test_allocation_identity_is_exclusive_to_one_generation_even_after_retire(self):
        s = self.acquire()
        reused = publication(generation="generation-b", digest="digest-b", allocation_ids=("a1", "b2"))
        self.assert_error("ALLOCATION_REUSED", self.r.publish, reused)
        self.assert_error("UNKNOWN_GENERATION", self.r.get_generation, "generation-b")
        self.r.drain(s.generation)
        self.release(s)
        self.r.retire(s.generation)
        self.assert_error("ALLOCATION_REUSED", self.r.publish, reused)
        self.assertEqual(self.r.publish(publication()).state, "RETIRED")
        self.assertEqual(self.r.publish(publication(generation="generation-b")).state, "READY")

    def run_race(self, *functions):
        barrier = threading.Barrier(len(functions))

        def run(function):
            barrier.wait(timeout=5)
            try:
                return function()
            except m.SessionError as error:
                return error.code

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(functions)) as pool:
            return list(pool.map(run, functions))

    def test_concurrent_same_request_allocates_one_session(self):
        results = self.run_race(*(lambda: self.acquire() for _ in range(16)))
        self.assertEqual(len({s.session_id for s in results}), 1)
        self.assertEqual(self.r.get_generation("generation-a").live_sessions, 1)

    def test_acquire_drain_race_has_only_serializable_outcomes(self):
        for i in range(40):
            self.r = m.GenerationRegistry(clock=self.clock)
            self.r.publish(publication())
            got, drained = self.run_race(lambda: self.acquire(str(i)), lambda: self.r.drain("generation-a"))
            self.assertEqual(drained.state, "DRAINING")
            if isinstance(got, str):
                self.assertEqual(got, "GENERATION_NOT_READY")
                self.assertEqual(self.r.get_generation("generation-a").live_sessions, 0)
            else:
                self.assertEqual(got.state, "PENDING")
                self.assertEqual(self.r.get_generation("generation-a").live_sessions, 1)
                self.release(got)
            self.r.retire("generation-a")

    def test_ack_drain_race_cannot_activate_after_drain(self):
        for _ in range(40):
            self.r = m.GenerationRegistry(clock=self.clock)
            self.r.publish(publication())
            s = self.acquire()
            ack, _ = self.run_race(lambda: self.ack(s), lambda: self.r.drain(s.generation))
            state = self.r.get_session(s.session_id).state
            if isinstance(ack, str):
                self.assertEqual((ack, state), ("GENERATION_NOT_READY", "PENDING"))
            else:
                self.assertEqual((ack.state, ack.generation_state, state), ("ACTIVE", "READY", "ACTIVE"))
            self.assert_error("LIVE_SESSIONS", self.r.retire, s.generation)
            self.release(s)
            self.r.retire(s.generation)

    def test_concurrent_release_and_retire_are_safe(self):
        s = self.acquire()
        self.r.drain(s.generation)
        results = self.run_race(lambda: self.release(s), lambda: self.release(s),
                                lambda: self.r.retire(s.generation))
        self.assertEqual(results[0].state, "RELEASED")
        self.assertEqual(results[1].state, "RELEASED")
        self.assertIn(results[2] if isinstance(results[2], str) else results[2].state,
                      ("LIVE_SESSIONS", "RETIRED"))
        self.assertEqual(self.r.retire(s.generation).state, "RETIRED")


if __name__ == "__main__":
    unittest.main()
