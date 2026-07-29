import ctypes
import os
import platform
import select
import subprocess
import sys
import unittest
from pathlib import Path

from rtp_llm.utils.multicast_keeper import (
    ENABLE_ENV,
    SYMM_MEM_HANDLE_POLICY_ENV,
    KeeperArtifacts,
    MulticastKeeperRuntime,
)


_FABRIC_BACKING_ORIGIN = r"""
import ctypes
import sys


class Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class AllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", AllocFlags),
    ]


class AccessDesc(ctypes.Structure):
    _fields_ = [("location", Location), ("flags", ctypes.c_uint)]


cuda = ctypes.CDLL("libcuda.so.1")
process = ctypes.CDLL(None)
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
cuda.cuMemCreate.argtypes = [
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.c_size_t,
    ctypes.POINTER(AllocationProp),
    ctypes.c_ulonglong,
]
cuda.cuMemCreate.restype = ctypes.c_int
cuda.cuMemExportToShareableHandle.argtypes = [
    ctypes.c_void_p,
    ctypes.c_ulonglong,
    ctypes.c_int,
    ctypes.c_ulonglong,
]
cuda.cuMemExportToShareableHandle.restype = ctypes.c_int
cuda.cuMemImportFromShareableHandle.argtypes = [
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.c_void_p,
    ctypes.c_int,
]
cuda.cuMemImportFromShareableHandle.restype = ctypes.c_int
cuda.cuMemRelease.argtypes = [ctypes.c_ulonglong]
cuda.cuMemRelease.restype = ctypes.c_int
cuda.cuMemAddressReserve.argtypes = [
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_ulonglong,
    ctypes.c_ulonglong,
]
cuda.cuMemAddressReserve.restype = ctypes.c_int
cuda.cuMemMap.argtypes = [
    ctypes.c_ulonglong,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_ulonglong,
    ctypes.c_ulonglong,
]
cuda.cuMemMap.restype = ctypes.c_int
cuda.cuMemSetAccess.argtypes = [
    ctypes.c_ulonglong,
    ctypes.c_size_t,
    ctypes.POINTER(AccessDesc),
    ctypes.c_size_t,
]
cuda.cuMemSetAccess.restype = ctypes.c_int
cuda.cuMemUnmap.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t]
cuda.cuMemUnmap.restype = ctypes.c_int
cuda.cuMemAddressFree.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t]
cuda.cuMemAddressFree.restype = ctypes.c_int
release_fences = process.rtp_llm_mc_release_fabric_backings
release_fences.argtypes = []
release_fences.restype = ctypes.c_int

assert cuda.cuInit(0) == 0
prop = AllocationProp(
    type=1,
    requestedHandleTypes=8,
    location=Location(type=1, id=0),
)
origin = ctypes.c_ulonglong()
assert cuda.cuMemCreate(
    ctypes.byref(origin), 2 * 1024 * 1024, ctypes.byref(prop), 0
) == 0

fabric = (ctypes.c_ubyte * 64)()
assert (
    cuda.cuMemExportToShareableHandle(
        ctypes.byref(fabric), origin, 8, 0
    )
    == 0
)
assert any(fabric)
print("FABRIC_TOKEN=" + bytes(fabric).hex(), flush=True)

assert sys.stdin.readline().strip() == "rendezvous-complete"
assert release_fences() == 1
address = ctypes.c_ulonglong()
size = 2 * 1024 * 1024
assert cuda.cuMemAddressReserve(ctypes.byref(address), size, 0, 0, 0) == 0
assert cuda.cuMemMap(address, size, 0, origin, 0) == 0
access = AccessDesc(location=Location(type=1, id=0), flags=3)
assert cuda.cuMemSetAccess(address, size, ctypes.byref(access), 1) == 0
assert cuda.cuMemUnmap(address, size) == 0
assert cuda.cuMemAddressFree(address, size) == 0
assert cuda.cuMemRelease(origin) == 0
print("fabric-backing-origin-ok", flush=True)
"""


_FABRIC_BACKING_PEER = r"""
import ctypes
import os


class Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class AccessDesc(ctypes.Structure):
    _fields_ = [("location", Location), ("flags", ctypes.c_uint)]


cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
cuda.cuMemImportFromShareableHandle.argtypes = [
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.c_void_p,
    ctypes.c_int,
]
cuda.cuMemImportFromShareableHandle.restype = ctypes.c_int
cuda.cuMemRelease.argtypes = [ctypes.c_ulonglong]
cuda.cuMemRelease.restype = ctypes.c_int
cuda.cuMemAddressReserve.argtypes = [
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_ulonglong,
    ctypes.c_ulonglong,
]
cuda.cuMemAddressReserve.restype = ctypes.c_int
cuda.cuMemMap.argtypes = [
    ctypes.c_ulonglong,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_ulonglong,
    ctypes.c_ulonglong,
]
cuda.cuMemMap.restype = ctypes.c_int
cuda.cuMemSetAccess.argtypes = [
    ctypes.c_ulonglong,
    ctypes.c_size_t,
    ctypes.POINTER(AccessDesc),
    ctypes.c_size_t,
]
cuda.cuMemSetAccess.restype = ctypes.c_int
cuda.cuMemUnmap.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t]
cuda.cuMemUnmap.restype = ctypes.c_int
cuda.cuMemAddressFree.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t]
cuda.cuMemAddressFree.restype = ctypes.c_int

assert cuda.cuInit(0) == 0
raw = bytes.fromhex(os.environ["RTP_LLM_MC_TEST_FABRIC_TOKEN"])
assert len(raw) == 64
fabric = (ctypes.c_ubyte * 64).from_buffer_copy(raw)
peer = ctypes.c_ulonglong()
assert (
    cuda.cuMemImportFromShareableHandle(
        ctypes.byref(peer), ctypes.byref(fabric), 8
    )
    == 0
)
address = ctypes.c_ulonglong()
size = 2 * 1024 * 1024
assert cuda.cuMemAddressReserve(ctypes.byref(address), size, 0, 0, 0) == 0
assert cuda.cuMemMap(address, size, 0, peer, 0) == 0
access = AccessDesc(location=Location(type=1, id=0), flags=3)
assert cuda.cuMemSetAccess(address, size, ctypes.byref(access), 1) == 0
assert cuda.cuMemUnmap(address, size) == 0
assert cuda.cuMemAddressFree(address, size) == 0
assert cuda.cuMemRelease(peer) == 0
print("fabric-backing-peer-ok", flush=True)
"""


class MulticastKeeperFabricBackingGpuTest(unittest.TestCase):
    @staticmethod
    def _artifact(name: str) -> Path:
        relative = Path("rtp_llm/cpp/cuda_checkpoint/multicast_keeper") / name
        candidates = []
        if os.environ.get("TEST_SRCDIR"):
            candidates.append(
                Path(os.environ["TEST_SRCDIR"])
                / os.environ.get("TEST_WORKSPACE", "__main__")
                / relative
            )
        candidates.append(
            Path(__file__).resolve().parents[3] / "bazel-bin" / relative
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise unittest.SkipTest(f"native keeper artifact is unavailable: {name}")

    def test_fabric_backing_broker_lifetime_and_reexport(self) -> None:
        if platform.machine().lower() not in {"aarch64", "x86_64"}:
            self.skipTest("unsupported ELF architecture")

        cuda = ctypes.CDLL("libcuda.so.1")
        cuda.cuInit.argtypes = [ctypes.c_uint]
        cuda.cuInit.restype = ctypes.c_int
        cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cuda.cuDeviceGetCount.restype = ctypes.c_int
        count = ctypes.c_int()
        if cuda.cuInit(0) != 0 or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            self.skipTest("CUDA driver is unavailable")
        if count.value < 1:
            self.skipTest("no CUDA device is visible")

        artifacts = KeeperArtifacts(
            holder=self._artifact("keeper_lite_holder"),
            creator=self._artifact("keeper_lite_creator"),
            shim=self._artifact("mc_shim_unified.so"),
        )
        env = dict(os.environ)
        env[ENABLE_ENV] = "1"
        runtime = MulticastKeeperRuntime(
            count.value,
            count.value,
            "prefill",
            env=env,
            artifacts=artifacts,
        )
        origin = None
        try:
            runtime.start()
            child_env = runtime.subprocess_env(env)
            child_env[SYMM_MEM_HANDLE_POLICY_ENV] = "fabric"
            origin = subprocess.Popen(
                [sys.executable, "-c", _FABRIC_BACKING_ORIGIN],
                env=child_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert origin.stdout is not None and origin.stdin is not None
            ready, _, _ = select.select([origin.stdout], [], [], 30)
            self.assertTrue(ready, "origin did not export a FABRIC token")
            token_line = origin.stdout.readline().strip()
            self.assertTrue(token_line.startswith("FABRIC_TOKEN="), token_line)

            peer_env = dict(child_env)
            peer_env["RTP_LLM_MC_TEST_FABRIC_TOKEN"] = token_line.split("=", 1)[1]
            peer = subprocess.run(
                [sys.executable, "-c", _FABRIC_BACKING_PEER],
                env=peer_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                0,
                peer.returncode,
                f"peer_stdout={peer.stdout}\npeer_stderr={peer.stderr}\n"
                f"holder={runtime.log_tail(80)}",
            )
            self.assertIn("fabric-backing-peer-ok", peer.stdout)

            origin.stdin.write("rendezvous-complete\n")
            origin.stdin.flush()
            origin_stdout, origin_stderr = origin.communicate(timeout=30)
            self.assertEqual(
                0,
                origin.returncode,
                f"origin_stdout={origin_stdout}\n"
                f"origin_stderr={origin_stderr}\n"
                f"holder={runtime.log_tail(80)}",
            )
            self.assertIn("fabric-backing-origin-ok", origin_stdout)
            self.assertIn("fabric_backing_released", runtime.log_tail(80))
            self.assertEqual(0, runtime.health().entries)
        finally:
            if origin is not None and origin.poll() is None:
                origin.terminate()
                try:
                    origin.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    origin.kill()
                    origin.wait(timeout=5)
            runtime.stop()


if __name__ == "__main__":
    unittest.main()
