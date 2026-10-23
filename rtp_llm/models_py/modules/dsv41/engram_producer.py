# SPDX-License-Identifier: Apache-2.0
"""Producer claim lifetime; losing coordination never frees GPU-visible backing."""
import logging
import threading
import time
from uuid import uuid4

from .engram_registry import RegistryClient
from .engram_rpc import ControlError


class ProducerSession:
    def __init__(self, client, attempt_id, grant, request_started):
        self.client, self.attempt_id, self.grant = client, attempt_id, grant
        self.epoch = grant["owner_epoch"]
        self.deadline = request_started + grant["lease_remaining_s"]
        self._stop = threading.Event()
        self._failed = threading.Event()
        self._rpc_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._heartbeat, daemon=True, name="engram-producer-lease"
        )
        self._thread.start()

    @classmethod
    def acquire(cls, required_memory_bytes, environ=None, *, attempt_id=None):
        client = RegistryClient(environ)
        attempt = attempt_id or str(uuid4())
        started = time.monotonic()
        try:
            grant = client.acquire(attempt, required_memory_bytes)
            if grant["role"] != "PRODUCER":
                client.close()
                return None, grant
            return cls(client, attempt, grant, started), grant
        except BaseException:
            client.close()
            raise

    def live(self):
        return (
            not self._failed.is_set()
            and not self._stop.is_set()
            and time.monotonic() < self.deadline
        )

    def ensure_capacity(self, physical_bytes):
        if not self.live():
            raise ControlError("PRODUCER_CLAIM_NOT_LIVE")
        if physical_bytes > self.grant["reserved_memory_bytes"]:
            raise ControlError("RESERVATION_EXCEEDED")

    def _heartbeat(self):
        delay = max(0.1, (self.deadline - time.monotonic()) / 3)
        while not self._stop.wait(delay):
            if not self.live():
                self._failed.set()
                break
            try:
                with self._rpc_lock:
                    started = time.monotonic()
                    reply = self.client.renew(self.attempt_id, self.epoch)
                    if reply["owner_epoch"] != self.epoch:
                        raise ControlError("OWNER_EPOCH_CHANGED")
                    if time.monotonic() >= self.deadline:
                        raise ControlError("LOCAL_LEASE_EXPIRED")
                    self.deadline = started + reply["lease_remaining_s"]
                delay = max(0.1, (self.deadline - time.monotonic()) / 3)
            except Exception:
                # Fail closed on admission, not on existing mapped memory.
                self._failed.set()
                logging.error("ENGRAM_PRODUCER_COORDINATION_LOST backing_retained=true")
                break

    def publish_ready(self, generation, physical_bytes):
        self.ensure_capacity(physical_bytes)
        with self._rpc_lock:
            response = self.client.publish(
                self.attempt_id, self.epoch, generation, physical_bytes
            )
            if response["owner_epoch"] != self.epoch:
                self._failed.set()
                raise ControlError("OWNER_EPOCH_CHANGED")
        return response

    def close(self):
        self._stop.set()
        self._thread.join(timeout=self.client.timeout + 1)
        self.client.close()
        # No release/re-election RPC: platform must fence/drain the old group.


class AssignedProducerSession:
    """A platform decision, not a lease/election or a hardware fencing token.

    The platform must enforce one Owner and retain its resources until all
    importers drain. This process only checks the delivered assignment.
    """

    def __init__(self, required_memory_bytes, environ=None):
        import os
        from types import SimpleNamespace
        from .engram_assignment import configured_assignment, identifier
        from .engram_rpc import configured_identity
        from .engram_registry import endpoint, positive_bytes

        env = os.environ if environ is None else environ
        self.assignment = configured_assignment(env)
        if self.assignment is None or env.get("RTP_ENGRAM_SIDECAR_ROLE") != "producer":
            raise ControlError("PLATFORM_PRODUCER_ROLE_REQUIRED")
        if env["RTP_ENGRAM_INSTANCE_ID"] != self.assignment["owner_instance_id"]:
            raise ControlError("PLATFORM_OWNER_INSTANCE_MISMATCH")
        identifier(env.get("RTP_ENGRAM_RESERVATION_ID"))
        reserved = positive_bytes(int(env.get("RTP_ENGRAM_RESERVED_MEMORY_BYTES", "0")))
        positive_bytes(required_memory_bytes)
        self.client = SimpleNamespace(identity=configured_identity(env))
        self.grant = dict(
            owner_endpoint=endpoint(env["RTP_ENGRAM_OWNER_ENDPOINT"]),
            reserved_memory_bytes=reserved,
        )
        self.epoch = self.assignment["owner_epoch"]
        self._closed = False
        self.ensure_capacity(required_memory_bytes)

    def live(self):
        return not self._closed

    def ensure_capacity(self, physical_bytes):
        if not self.live():
            raise ControlError("PLATFORM_ASSIGNMENT_CLOSED")
        if (
            type(physical_bytes) is not int
            or not 0 < physical_bytes <= self.grant["reserved_memory_bytes"]
        ):
            raise ControlError("RESERVATION_EXCEEDED")

    def publish_ready(self, generation, physical_bytes):
        self.ensure_capacity(physical_bytes)
        return dict(
            state="READY",
            assignment=dict(self.assignment),
            generation=generation,
            physical_bytes=physical_bytes,
        )

    def close(self):
        self._closed = True
