"""Level-3 deep-sleep checkpoint controller.

Drives the CUDA driver ``cuCheckpointProcess*`` state machine on a TARGET
process (the backend engine) from an EXTERNAL process (the launcher / parent).
A process cannot checkpoint itself -- the call would freeze it mid-flight -- so
this runs in the launcher, which is the parent of the backend and stays alive
across sleep.

Per-pid driver state machine::

    RUNNING --Lock--> LOCKED --Checkpoint--> CHECKPOINTED
    CHECKPOINTED --Restore--> LOCKED --Unlock--> RUNNING

For a set of local ranks the controller drives them in lockstep: Lock ALL ->
Checkpoint ALL on sleep, Restore ALL -> Unlock ALL on wake, mirroring the
multi_controller PoC.

Safety (hard-won, see design doc risk C7): a hung/failed Checkpoint must NEVER
leave a target LOCKED and then get hard-killed -- that leaks its GPU memory and
only a GPU reset recovers it. On any failure the controller rolls back via
Restore/Unlock to bring every target back to RUNNING so it can resume or exit
cleanly. Callers must therefore never SIGKILL a target on checkpoint failure;
call :meth:`CheckpointController.rollback` instead.

The CUDA driver marshalling lives entirely in :class:`LibCudaCheckpointDriver`;
the orchestration logic in :class:`CheckpointController` is driver-agnostic and
unit-testable with a fake driver.
"""

import contextlib
import ctypes
import fcntl
import hashlib
import json
import logging
import os
import tempfile
from typing import List, Optional, Protocol, Sequence

# CUprocessState (cuda.h 12.9, CU_PROCESS_STATE_*). Integer values are stable ABI.
STATE_RUNNING = 0
STATE_LOCKED = 1
STATE_CHECKPOINTED = 2
STATE_FAILED = 3

_STATE_NAMES = {
    STATE_RUNNING: "RUNNING",
    STATE_LOCKED: "LOCKED",
    STATE_CHECKPOINTED: "CHECKPOINTED",
    STATE_FAILED: "FAILED",
}

CUDA_SUCCESS = 0


def state_name(state: int) -> str:
    return _STATE_NAMES.get(state, f"UNKNOWN({state})")


class CheckpointError(RuntimeError):
    """Raised when a driver checkpoint operation fails."""


class CudaCheckpointDriver(Protocol):
    """Minimal driver surface the controller depends on. Each op returns a CUDA
    result code (0 == CUDA_SUCCESS)."""

    def get_state(self, pid: int) -> int: ...
    def lock(self, pid: int, timeout_ms: int) -> int: ...
    def checkpoint(self, pid: int) -> int: ...
    def restore(self, pid: int) -> int: ...
    def unlock(self, pid: int) -> int: ...
    def error_string(self, rc: int) -> str: ...


# --- ctypes structs mirroring cuda.h 12.9 CUcheckpoint*Args ------------------


class _CUcheckpointLockArgs(ctypes.Structure):
    _fields_ = [
        ("timeoutMs", ctypes.c_uint),
        ("reserved0", ctypes.c_uint),
        ("reserved1", ctypes.c_uint64 * 7),
    ]


class _CUcheckpointReservedArgs(ctypes.Structure):
    # Shared layout for Checkpoint/Restore/Unlock args: reserved cuuint64_t[8].
    _fields_ = [("reserved", ctypes.c_uint64 * 8)]


class LibCudaCheckpointDriver:
    """Real driver backed by libcuda.so cuCheckpointProcess* entry points."""

    def __init__(self, lib_name: str = "libcuda.so.1"):
        self._lib = ctypes.CDLL(lib_name)
        # cuInit is required before any driver API; safe to call repeatedly.
        rc = self._lib.cuInit(0)
        if rc != CUDA_SUCCESS:
            raise CheckpointError(f"cuInit failed: {self.error_string(rc)}")

    def error_string(self, rc: int) -> str:
        msg = ctypes.c_char_p()
        if (
            self._lib.cuGetErrorString(rc, ctypes.byref(msg)) == CUDA_SUCCESS
            and msg.value
        ):
            return f"{rc} ({msg.value.decode('utf-8', 'replace')})"
        return str(rc)

    def get_state(self, pid: int) -> int:
        state = ctypes.c_int(-1)
        rc = self._lib.cuCheckpointProcessGetState(
            ctypes.c_int(pid), ctypes.byref(state)
        )
        if rc != CUDA_SUCCESS:
            raise CheckpointError(f"GetState pid {pid} failed: {self.error_string(rc)}")
        return state.value

    def lock(self, pid: int, timeout_ms: int) -> int:
        args = _CUcheckpointLockArgs()
        args.timeoutMs = ctypes.c_uint(timeout_ms).value
        return self._lib.cuCheckpointProcessLock(ctypes.c_int(pid), ctypes.byref(args))

    def checkpoint(self, pid: int) -> int:
        args = _CUcheckpointReservedArgs()
        return self._lib.cuCheckpointProcessCheckpoint(
            ctypes.c_int(pid), ctypes.byref(args)
        )

    def restore(self, pid: int) -> int:
        args = _CUcheckpointReservedArgs()
        return self._lib.cuCheckpointProcessRestore(
            ctypes.c_int(pid), ctypes.byref(args)
        )

    def unlock(self, pid: int) -> int:
        args = _CUcheckpointReservedArgs()
        return self._lib.cuCheckpointProcessUnlock(
            ctypes.c_int(pid), ctypes.byref(args)
        )


class CheckpointController:
    """Orchestrates lockstep checkpoint/restore of a set of local target pids.

    Driver-agnostic: pass a fake :class:`CudaCheckpointDriver` in tests.
    """

    def __init__(
        self, driver: CudaCheckpointDriver = None, lock_timeout_ms: int = 60000
    ):
        self._driver = driver if driver is not None else LibCudaCheckpointDriver()
        self._lock_timeout_ms = lock_timeout_ms

    def get_states(self, pids: Sequence[int]) -> List[int]:
        return [self._driver.get_state(pid) for pid in pids]

    def checkpoint(self, pids: Sequence[int]) -> None:
        """Lock ALL then Checkpoint ALL target pids (GPU -> 0). On any failure,
        roll every target back to RUNNING and raise. Never leaves a pid LOCKED.
        """
        pids = list(pids)
        try:
            for pid in pids:
                rc = self._driver.lock(pid, self._lock_timeout_ms)
                if rc != CUDA_SUCCESS:
                    raise CheckpointError(
                        f"Lock pid {pid} failed: {self._driver.error_string(rc)}"
                    )
            for pid in pids:
                rc = self._driver.checkpoint(pid)
                if rc != CUDA_SUCCESS:
                    raise CheckpointError(
                        f"Checkpoint pid {pid} failed: {self._driver.error_string(rc)}"
                    )
            for pid in pids:
                st = self._driver.get_state(pid)
                if st != STATE_CHECKPOINTED:
                    raise CheckpointError(
                        f"pid {pid} not CHECKPOINTED after checkpoint (state={state_name(st)})"
                    )
            logging.info(f"[deep-sleep] checkpointed {len(pids)} target pid(s): {pids}")
        except Exception as e:
            logging.error(
                f"[deep-sleep] checkpoint failed ({e}); rolling back to RUNNING"
            )
            self.rollback(pids)
            raise

    def restore(self, pids: Sequence[int]) -> None:
        """Restore ALL then Unlock ALL target pids back to RUNNING."""
        pids = list(pids)
        for pid in pids:
            if self._driver.get_state(pid) == STATE_CHECKPOINTED:
                rc = self._driver.restore(pid)
                if rc != CUDA_SUCCESS:
                    raise CheckpointError(
                        f"Restore pid {pid} failed: {self._driver.error_string(rc)}"
                    )
        for pid in pids:
            if self._driver.get_state(pid) == STATE_LOCKED:
                rc = self._driver.unlock(pid)
                if rc != CUDA_SUCCESS:
                    raise CheckpointError(
                        f"Unlock pid {pid} failed: {self._driver.error_string(rc)}"
                    )
        for pid in pids:
            st = self._driver.get_state(pid)
            if st != STATE_RUNNING:
                raise CheckpointError(
                    f"pid {pid} not RUNNING after restore (state={state_name(st)})"
                )
        logging.info(f"[deep-sleep] restored {len(pids)} target pid(s): {pids}")

    def rollback(self, pids: Sequence[int]) -> None:
        """Best-effort bring every pid back to RUNNING (Restore if CHECKPOINTED,
        Unlock if LOCKED). Used when a checkpoint hangs/fails. NEVER kills a
        target -- killing a LOCKED process leaks its GPU memory. Swallows
        per-pid errors so one wedged pid does not block rolling back the rest.
        """
        for pid in pids:
            try:
                st = self._driver.get_state(pid)
                if st == STATE_CHECKPOINTED:
                    self._driver.restore(pid)
                    st = self._driver.get_state(pid)
                if st == STATE_LOCKED:
                    self._driver.unlock(pid)
            except Exception as e:  # best-effort per pid
                logging.error(f"[deep-sleep] rollback of pid {pid} failed: {e}")


# --- Node-local deep-sleep registry -----------------------------------------
# When a level-3 sleep checkpoints the local backend pids, the checkpointing
# frontend records them here so that ANY frontend of the same instance on this
# node can, on wake, discover the checkpointed pids and restore them FIRST
# (control-flow inversion: the frozen backend cannot answer gRPC until restored).
# Keyed by the backend control addresses so all frontends of one instance agree.


def deep_sleep_registry_path(control_addresses: Sequence[str]) -> str:
    key = hashlib.sha1("|".join(sorted(control_addresses)).encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"rtp_llm_deep_sleep_{key}.json")


def write_deep_sleep_registry(path: str, pids: Sequence[int]) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump({"pids": [int(p) for p in pids]}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic publish
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def read_deep_sleep_registry(path: str) -> Optional[List[int]]:
    """Return the checkpointed pids, or None if no registry exists."""
    try:
        with open(path) as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("pids"), list):
        raise ValueError(f"invalid deep-sleep registry payload in {path}")
    pids = [int(pid) for pid in payload["pids"]]
    if not pids or any(pid <= 0 for pid in pids):
        raise ValueError(f"invalid deep-sleep pid list in {path}")
    return pids


def clear_deep_sleep_registry(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


@contextlib.contextmanager
def deep_sleep_registry_lock(path: str):
    """Serialize level-3 checkpoint/restore across frontend processes."""
    with open(f"{path}.lock", "a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# --- Deep-sleep checkpoint orchestration ------------------------------------
# Top-level entry points that tie the CheckpointController to the node-local
# registry. Callers (the frontend coordinator) pass the backend control
# addresses; the registry path is an internal detail derived from them. These
# run synchronously and are meant to be dispatched via asyncio.to_thread.


def checkpoint_deep_sleep_targets(
    control_addresses: Sequence[str], pids: Sequence[int]
) -> None:
    """Checkpoint local backend pids and leave durable recovery information.

    Publishes the pid registry BEFORE checkpointing so that if this coordinator
    dies mid-checkpoint, a later wake request can still restore/rollback the
    pids. Clears the registry and re-raises on failure.
    """
    registry_path = deep_sleep_registry_path(control_addresses)
    with deep_sleep_registry_lock(registry_path):
        existing_pids = read_deep_sleep_registry(registry_path)
        if existing_pids is not None:
            raise RuntimeError(
                f"deep-sleep registry already exists for pids {existing_pids}"
            )
        write_deep_sleep_registry(registry_path, pids)
        try:
            CheckpointController().checkpoint(pids)
        except Exception:
            clear_deep_sleep_registry(registry_path)
            raise


def restore_deep_sleep_targets(control_addresses: Sequence[str]) -> bool:
    """Restore registered local backend pids. Return whether a registry existed.

    Called on wake BEFORE any control RPC (a checkpointed backend is frozen and
    cannot answer gRPC until restored).
    """
    registry_path = deep_sleep_registry_path(control_addresses)
    with deep_sleep_registry_lock(registry_path):
        pids = read_deep_sleep_registry(registry_path)
        if pids is None:
            return False
        CheckpointController().restore(pids)
        clear_deep_sleep_registry(registry_path)
        return True
