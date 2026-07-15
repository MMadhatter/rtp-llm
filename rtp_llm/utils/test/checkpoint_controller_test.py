# type: ignore
import json
import os
import tempfile
import unittest
from unittest import mock

from rtp_llm.utils.checkpoint_controller import (
    CUDA_SUCCESS,
    STATE_CHECKPOINTED,
    STATE_LOCKED,
    STATE_RUNNING,
    CheckpointController,
    CheckpointError,
    checkpoint_deep_sleep_targets,
    clear_deep_sleep_registry,
    deep_sleep_registry_path,
    read_deep_sleep_registry,
    restore_deep_sleep_targets,
    write_deep_sleep_registry,
)

# A non-zero CUDA result code used to simulate a driver failure.
CUDA_ERROR = 999


class FakeDriver:
    """In-memory model of the cuCheckpointProcess* state machine, with fault
    injection. Mirrors the real driver transitions:
      RUNNING --lock--> LOCKED --checkpoint--> CHECKPOINTED
      CHECKPOINTED --restore--> LOCKED --unlock--> RUNNING
    """

    def __init__(self, pids, fail_on=None, fail_pid=None):
        # fail_on: op name ("lock"/"checkpoint"/"restore"/"unlock") to fail once.
        self.states = {pid: STATE_RUNNING for pid in pids}
        self.fail_on = fail_on
        self.fail_pid = fail_pid
        self.calls = []  # ordered (op, pid) log for assertions

    def _maybe_fail(self, op, pid):
        if self.fail_on == op and (self.fail_pid is None or self.fail_pid == pid):
            return CUDA_ERROR
        return CUDA_SUCCESS

    def get_state(self, pid):
        self.calls.append(("get_state", pid))
        return self.states[pid]

    def lock(self, pid, timeout_ms):
        self.calls.append(("lock", pid))
        rc = self._maybe_fail("lock", pid)
        if rc == CUDA_SUCCESS and self.states[pid] == STATE_RUNNING:
            self.states[pid] = STATE_LOCKED
        return rc

    def checkpoint(self, pid):
        self.calls.append(("checkpoint", pid))
        rc = self._maybe_fail("checkpoint", pid)
        if rc == CUDA_SUCCESS and self.states[pid] == STATE_LOCKED:
            self.states[pid] = STATE_CHECKPOINTED
        return rc

    def restore(self, pid):
        self.calls.append(("restore", pid))
        rc = self._maybe_fail("restore", pid)
        if rc == CUDA_SUCCESS and self.states[pid] == STATE_CHECKPOINTED:
            self.states[pid] = STATE_LOCKED
        return rc

    def unlock(self, pid):
        self.calls.append(("unlock", pid))
        rc = self._maybe_fail("unlock", pid)
        if rc == CUDA_SUCCESS and self.states[pid] == STATE_LOCKED:
            self.states[pid] = STATE_RUNNING
        return rc

    def error_string(self, rc):
        return f"fake-error-{rc}"


class CheckpointControllerTest(unittest.TestCase):
    PIDS = [111, 222]

    def test_checkpoint_locks_all_then_checkpoints_all(self):
        drv = FakeDriver(self.PIDS)
        ctl = CheckpointController(driver=drv)
        ctl.checkpoint(self.PIDS)

        for pid in self.PIDS:
            self.assertEqual(drv.states[pid], STATE_CHECKPOINTED)
        # Lockstep: both locks precede either checkpoint.
        ops = [op for (op, _pid) in drv.calls if op in ("lock", "checkpoint")]
        self.assertEqual(ops, ["lock", "lock", "checkpoint", "checkpoint"])

    def test_restore_brings_all_back_to_running(self):
        drv = FakeDriver(self.PIDS)
        ctl = CheckpointController(driver=drv)
        ctl.checkpoint(self.PIDS)
        ctl.restore(self.PIDS)
        for pid in self.PIDS:
            self.assertEqual(drv.states[pid], STATE_RUNNING)

    def test_checkpoint_failure_rolls_back_to_running_and_raises(self):
        # Second pid fails Checkpoint; first was already CHECKPOINTED. Rollback
        # must bring BOTH back to RUNNING and never leave anything LOCKED.
        drv = FakeDriver(self.PIDS, fail_on="checkpoint", fail_pid=222)
        ctl = CheckpointController(driver=drv)
        with self.assertRaises(CheckpointError):
            ctl.checkpoint(self.PIDS)
        for pid in self.PIDS:
            self.assertEqual(
                drv.states[pid], STATE_RUNNING, f"pid {pid} left in {drv.states[pid]}"
            )

    def test_lock_failure_rolls_back_and_raises(self):
        # First pid fails to Lock; the (partially) locked set must be rolled back.
        drv = FakeDriver(self.PIDS, fail_on="lock", fail_pid=111)
        ctl = CheckpointController(driver=drv)
        with self.assertRaises(CheckpointError):
            ctl.checkpoint(self.PIDS)
        for pid in self.PIDS:
            self.assertEqual(drv.states[pid], STATE_RUNNING)
        # No pid may be left LOCKED (the GPU-leak hazard).
        self.assertNotIn(STATE_LOCKED, drv.states.values())

    def test_rollback_from_checkpointed_and_locked(self):
        drv = FakeDriver(self.PIDS)
        ctl = CheckpointController(driver=drv)
        # Manually drive one pid to CHECKPOINTED, one to LOCKED.
        drv.lock(111, 0)
        drv.checkpoint(111)  # 111 CHECKPOINTED
        drv.lock(222, 0)  # 222 LOCKED
        self.assertEqual(drv.states[111], STATE_CHECKPOINTED)
        self.assertEqual(drv.states[222], STATE_LOCKED)

        ctl.rollback(self.PIDS)
        self.assertEqual(drv.states[111], STATE_RUNNING)
        self.assertEqual(drv.states[222], STATE_RUNNING)

    def test_rollback_is_best_effort_across_pids(self):
        # A pid whose restore fails must not prevent rolling back the others.
        drv = FakeDriver(self.PIDS, fail_on="restore", fail_pid=111)
        ctl = CheckpointController(driver=drv)
        drv.lock(111, 0)
        drv.checkpoint(111)  # 111 CHECKPOINTED, restore will fail
        drv.lock(222, 0)  # 222 LOCKED
        ctl.rollback(self.PIDS)  # must not raise
        # 111 stuck (restore failed) but 222 recovered.
        self.assertEqual(drv.states[222], STATE_RUNNING)

    def test_get_states(self):
        drv = FakeDriver(self.PIDS)
        ctl = CheckpointController(driver=drv)
        self.assertEqual(ctl.get_states(self.PIDS), [STATE_RUNNING, STATE_RUNNING])
        ctl.checkpoint(self.PIDS)
        self.assertEqual(
            ctl.get_states(self.PIDS), [STATE_CHECKPOINTED, STATE_CHECKPOINTED]
        )

    def test_registry_roundtrip_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "deep_sleep.json")
            self.assertIsNone(read_deep_sleep_registry(path))
            write_deep_sleep_registry(path, self.PIDS)
            self.assertEqual(read_deep_sleep_registry(path), self.PIDS)
            clear_deep_sleep_registry(path)
            self.assertIsNone(read_deep_sleep_registry(path))

    def test_registry_path_is_order_independent_and_instance_specific(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        self.assertEqual(
            deep_sleep_registry_path(addresses),
            deep_sleep_registry_path(list(reversed(addresses))),
        )
        self.assertNotEqual(
            deep_sleep_registry_path(addresses),
            deep_sleep_registry_path(["127.0.0.1:10017"]),
        )

    def test_registry_rejects_corrupt_or_empty_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "deep_sleep.json")
            for payload in ({}, {"pids": []}, {"pids": [0]}):
                with self.subTest(payload=payload):
                    with open(path, "w") as writer:
                        json.dump(payload, writer)
                    with self.assertRaises(ValueError):
                        read_deep_sleep_registry(path)


class DeepSleepOrchestrationTest(unittest.TestCase):
    """Covers the top-level checkpoint/restore entry points that tie the
    controller to the node-local registry. The CUDA driver is faked and the
    registry path is redirected into a temp dir."""

    PIDS = [111, 222]
    ADDRS = ["127.0.0.1:10001", "127.0.0.1:10009"]

    def _patched(self, directory, driver):
        # Redirect the registry path into `directory` and back the controller
        # with `driver`. Note CheckpointController referenced here is the test's
        # own (real) import, so this replaces the module global without recursion.
        path = os.path.join(directory, "reg.json")
        return path, (
            mock.patch(
                "rtp_llm.utils.checkpoint_controller.deep_sleep_registry_path",
                return_value=path,
            ),
            mock.patch(
                "rtp_llm.utils.checkpoint_controller.CheckpointController",
                side_effect=lambda *a, **k: CheckpointController(driver=driver),
            ),
        )

    def test_checkpoint_then_restore_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            drv = FakeDriver(self.PIDS)
            path, (p_path, p_ctl) = self._patched(directory, drv)
            with p_path, p_ctl:
                checkpoint_deep_sleep_targets(self.ADDRS, self.PIDS)
                # Registry published and all pids checkpointed (GPU released).
                self.assertEqual(read_deep_sleep_registry(path), self.PIDS)
                for pid in self.PIDS:
                    self.assertEqual(drv.states[pid], STATE_CHECKPOINTED)
                # Restore reports the registry existed, clears it, all RUNNING.
                self.assertTrue(restore_deep_sleep_targets(self.ADDRS))
                self.assertIsNone(read_deep_sleep_registry(path))
                for pid in self.PIDS:
                    self.assertEqual(drv.states[pid], STATE_RUNNING)

    def test_restore_without_registry_returns_false(self):
        with tempfile.TemporaryDirectory() as directory:
            drv = FakeDriver(self.PIDS)
            _path, (p_path, p_ctl) = self._patched(directory, drv)
            with p_path, p_ctl:
                self.assertFalse(restore_deep_sleep_targets(self.ADDRS))

    def test_checkpoint_rejects_existing_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            drv = FakeDriver(self.PIDS)
            path, (p_path, p_ctl) = self._patched(directory, drv)
            with p_path, p_ctl:
                write_deep_sleep_registry(path, [999])
                with self.assertRaises(RuntimeError):
                    checkpoint_deep_sleep_targets(self.ADDRS, self.PIDS)
                # Pre-existing registry left untouched; nothing checkpointed.
                self.assertEqual(read_deep_sleep_registry(path), [999])

    def test_checkpoint_failure_clears_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            drv = FakeDriver(self.PIDS, fail_on="checkpoint", fail_pid=222)
            path, (p_path, p_ctl) = self._patched(directory, drv)
            with p_path, p_ctl:
                with self.assertRaises(CheckpointError):
                    checkpoint_deep_sleep_targets(self.ADDRS, self.PIDS)
                # The registry this call published must be cleared on failure so
                # a later wake does not try to restore never-checkpointed pids.
                self.assertIsNone(read_deep_sleep_registry(path))
                # Rollback left every pid RUNNING (never LOCKED -> no GPU leak).
                for pid in self.PIDS:
                    self.assertEqual(drv.states[pid], STATE_RUNNING)


if __name__ == "__main__":
    unittest.main()
