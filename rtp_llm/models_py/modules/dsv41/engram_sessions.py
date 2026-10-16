# SPDX-License-Identifier: Apache-2.0
"""S1b in-process control registry, with no RPC, CUDA, PID checks or allocator.

publish() accepts the output of S1a validate_for_attach() from a trusted caller.
Session identity checks are consistency checks, not network authentication.
Release(quiesced=True) records a client's declaration, NOT proof of GPU fencing.
The caller must perform actual drain/unmap and arrange owner resource cleanup.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from threading import RLock
import time
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .engram_manifest import ValidatedManifest


class SessionError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class GenerationSnapshot:
    dataset_id: str
    generation: str
    manifest_digest: str
    allocation_ids: frozenset[str]
    state: str
    live_sessions: int


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    request_id: str
    client_id: str
    worker_incarnation: str
    device_ids: tuple[str, ...]
    dataset_id: str
    generation: str
    state: str
    generation_state: str
    lease_deadline: float


@dataclass(frozen=True)
class _Request:
    dataset_id: str
    generation: str
    request_id: str
    client_id: str
    worker_incarnation: str
    device_ids: tuple[str, ...]


@dataclass
class _Generation:
    dataset_id: str
    generation: str
    manifest_digest: str
    allocation_ids: frozenset[str]
    state: str = "READY"


@dataclass
class _Session:
    session_id: str
    request: _Request
    lease_deadline: float
    phase: str = "PENDING"
    suspect: bool = False


def _key(value):
    if not isinstance(value, str) or not 0 < len(value) <= 256 or any(ord(c) < 32 for c in value):
        raise SessionError("INVALID_ARGUMENT")
    return value


def _ids(values):
    if not isinstance(values, (list, tuple, set, frozenset)) or not 0 < len(values) <= 4096:
        raise SessionError("INVALID_ARGUMENT")
    result = tuple(sorted(_key(value) for value in values))
    if len(set(result)) != len(result):
        raise SessionError("DUPLICATE_ID")
    return result


class GenerationRegistry:
    """All transitions serialize under one lock; snapshots cannot mutate state.

    Acquisition tombstones are retained until registry shutdown for idempotency.
    Caps include released/retired entries: exhaustion refuses new work, never
    evicts an in-use reference or silently forgets an idempotency key.
    """

    def __init__(self, lease_seconds=60.0, *, clock=time.monotonic,
                 max_sessions=10000, max_generations=1024):
        if (type(lease_seconds) not in (int, float) or not math.isfinite(lease_seconds)
                or lease_seconds <= 0 or not callable(clock)):
            raise SessionError("INVALID_ARGUMENT")
        if any(type(n) is not int or n < 1 for n in (max_sessions, max_generations)):
            raise SessionError("INVALID_ARGUMENT")
        self._lease = lease_seconds
        self._clock, self._max_sessions, self._max_generations = clock, max_sessions, max_generations
        self._lock = RLock()
        self._generations, self._sessions, self._requests = {}, {}, {}
        self._allocation_generations = {}  # Ownership tombstones survive retirement.

    def _generation(self, generation):
        _key(generation)
        if generation not in self._generations:
            raise SessionError("UNKNOWN_GENERATION")
        return self._generations[generation]

    def _session(self, session_id, client_id=None, worker_incarnation=None):
        _key(session_id)
        if session_id not in self._sessions:
            raise SessionError("UNKNOWN_SESSION")
        s = self._sessions[session_id]
        if client_id is not None and (s.request.client_id, s.request.worker_incarnation) != (
                client_id, worker_incarnation):
            raise SessionError("SESSION_IDENTITY_MISMATCH")
        if s.phase != "RELEASED" and self._clock() >= s.lease_deadline:
            s.suspect = True
        return s

    def _session_snapshot(self, s):
        r = s.request
        state = "SUSPECT" if s.suspect and s.phase != "RELEASED" else s.phase
        return SessionSnapshot(s.session_id, r.request_id, r.client_id, r.worker_incarnation,
                               r.device_ids, r.dataset_id, r.generation, state,
                               self._generations[r.generation].state, s.lease_deadline)

    def _identified_session(self, session_id, client_id, worker_incarnation):
        return self._session(session_id, _key(client_id), _key(worker_incarnation))

    def _generation_snapshot(self, g):
        live = sum(s.request.generation == g.generation and s.phase != "RELEASED"
                   for s in self._sessions.values())
        return GenerationSnapshot(g.dataset_id, g.generation, g.manifest_digest,
                                  g.allocation_ids, g.state, live)

    def publish(self, validated: ValidatedManifest):
        """Publish one validated READY generation; never reopen a retired one."""
        m = validated.manifest
        if m.state != "READY":
            raise SessionError("PUBLICATION_NOT_READY")
        g = _Generation(_key(m.dataset_id), _key(m.generation), _key(validated.manifest_digest),
                        frozenset(_ids(list({s.allocation_id for t in m.tensors for s in t.shards}))))
        with self._lock:
            old = self._generations.get(g.generation)
            if old is not None:
                if (old.dataset_id, old.manifest_digest, old.allocation_ids) != (
                        g.dataset_id, g.manifest_digest, g.allocation_ids):
                    raise SessionError("PUBLICATION_CONFLICT")
                return self._generation_snapshot(old)
            if len(self._generations) >= self._max_generations:
                raise SessionError("GENERATION_LIMIT")
            if any(a in self._allocation_generations for a in g.allocation_ids):
                raise SessionError("ALLOCATION_REUSED")
            self._generations[g.generation] = g
            self._allocation_generations.update((a, g.generation) for a in g.allocation_ids)
            return self._generation_snapshot(g)

    def acquire(self, dataset_id, generation, request_id, client_id, worker_incarnation, device_ids):
        request = _Request(*(_key(v) for v in (dataset_id, generation, request_id,
                                              client_id, worker_incarnation)), _ids(device_ids))
        with self._lock:
            if request_id in self._requests:
                s = self._session(self._requests[request_id])
                if s.request != request:
                    raise SessionError("IDEMPOTENCY_CONFLICT")
                return self._session_snapshot(s)
            g = self._generation(generation)
            if g.dataset_id != dataset_id:
                raise SessionError("DATASET_MISMATCH")
            if g.state != "READY":
                raise SessionError("GENERATION_NOT_READY")
            if len(self._sessions) >= self._max_sessions:
                raise SessionError("SESSION_LIMIT")
            s = _Session(str(uuid4()), request, self._clock() + self._lease)
            self._sessions[s.session_id] = s
            self._requests[request_id] = s.session_id
            return self._session_snapshot(s)

    def attach_ack(self, session_id, client_id, worker_incarnation, manifest_digest, allocation_ids):
        imported = frozenset(_ids(allocation_ids))
        with self._lock:
            s = self._identified_session(session_id, client_id, worker_incarnation)
            g = self._generation(s.request.generation)
            if manifest_digest != g.manifest_digest or imported != g.allocation_ids:
                raise SessionError("ATTACH_METADATA_MISMATCH")
            if s.phase == "RELEASED":
                raise SessionError("SESSION_RELEASED")
            if s.suspect:
                raise SessionError("SESSION_SUSPECT")
            if s.phase == "ACTIVE":
                return self._session_snapshot(s)  # Lost ACK retry, including during drain.
            if g.state != "READY":
                raise SessionError("GENERATION_NOT_READY")
            s.phase, s.lease_deadline = "ACTIVE", self._clock() + self._lease
            return self._session_snapshot(s)

    def renew(self, session_id, client_id, worker_incarnation):
        with self._lock:
            s = self._identified_session(session_id, client_id, worker_incarnation)
            if s.phase == "RELEASED":
                raise SessionError("SESSION_RELEASED")
            s.suspect, s.lease_deadline = False, self._clock() + self._lease
            return self._session_snapshot(s)

    def release(self, session_id, client_id, worker_incarnation, *, quiesced):
        if quiesced is not True:
            raise SessionError("QUIESCENCE_REQUIRED")
        with self._lock:
            s = self._identified_session(session_id, client_id, worker_incarnation)
            s.phase, s.suspect = "RELEASED", False
            return self._session_snapshot(s)

    def mark_disconnected(self, session_id):
        """Trusted monitor hint; never removes a reference or releases memory."""
        with self._lock:
            s = self._session(session_id)
            if s.phase != "RELEASED":
                s.suspect = True
            return self._session_snapshot(s)

    def expire_sessions(self):
        """A monitor may call this periodically; timeouts retain all references."""
        with self._lock:
            changed = []
            for s in self._sessions.values():
                if s.phase != "RELEASED" and not s.suspect and self._clock() >= s.lease_deadline:
                    s.suspect = True
                    changed.append(self._session_snapshot(s))
            return tuple(changed)

    def drain(self, generation):
        with self._lock:
            g = self._generation(generation)
            if g.state == "READY":
                g.state = "DRAINING"
            return self._generation_snapshot(g)

    def retire(self, generation):
        """Retire metadata only; caller owns actual CUDA/OS reference cleanup."""
        with self._lock:
            g = self._generation(generation)
            snapshot = self._generation_snapshot(g)
            if g.state == "RETIRED":
                return snapshot
            if g.state != "DRAINING":
                raise SessionError("DRAIN_REQUIRED")
            if snapshot.live_sessions:
                raise SessionError("LIVE_SESSIONS")
            g.state = "RETIRED"
            return self._generation_snapshot(g)

    def get_generation(self, generation):
        with self._lock:
            return self._generation_snapshot(self._generation(generation))

    def get_session(self, session_id):
        with self._lock:
            return self._session_snapshot(self._session(session_id))
