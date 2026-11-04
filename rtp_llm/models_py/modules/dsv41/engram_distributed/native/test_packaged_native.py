"""Load the packaged bridge without initializing CUDA or building an extension."""

import importlib
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv41.engram_distributed.bootstrap import _load_native


class PackagedNativeTest(unittest.TestCase):
    def test_packaged_bridge_import_and_abi_without_cuda_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())
        import rtp_llm.ops as ops

        self.assertNotIn("libengram_native", sys.modules)
        self.assertNotIn("librtp_compute_ops", sys.modules)
        self.assertNotIn("libth_transformer", sys.modules)
        native = _load_native()
        self.assertIs(native, ops.load_engram_ops())
        self.assertIs(native, importlib.import_module("libengram_native"))
        self.assertEqual(native.__name__, "libengram_native")
        self.assertNotIn("rtp_llm.libs.libengram_native", sys.modules)
        path = Path(native.__file__)
        self.assertEqual(path.name, "libengram_native.so")
        self.assertEqual(path.parent.name, "libs")
        self.assertTrue(callable(native.device_granularity))
        self.assertTrue(callable(native.host_numa_granularity))
        self.assertTrue(callable(native.Owner))
        self.assertTrue(callable(native.Consumer))
        self.assertGreater(native.fabric_handle_bytes, 0)
        self.assertTrue(
            all(value == 0 for value in native.resource_counters().values())
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_optional_bridge_does_not_block_regular_ops_import(self):
        script = """
import importlib.abc
import sys

class MissingEngram(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "libengram_native":
            raise ModuleNotFoundError("Engram intentionally absent", name=fullname)

sys.meta_path.insert(0, MissingEngram())
import torch
import rtp_llm.ops
from rtp_llm.models_py.modules.dsv41.engram_distributed.bootstrap import (
    BootstrapError, _load_native,
)
assert "libengram_native" not in sys.modules
assert "librtp_compute_ops" not in sys.modules
assert "libth_transformer" not in sys.modules
assert not torch.cuda.is_initialized()
try:
    _load_native()
except BootstrapError as error:
    assert error.code == "PREBUILT_ENGRAM_NATIVE_MODULE_REQUIRED", error.code
else:
    raise AssertionError("missing Engram unexpectedly loaded")
assert not torch.cuda.is_initialized()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
