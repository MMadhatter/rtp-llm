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
