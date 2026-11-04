"""CPU-only regression coverage for the opt-in Fabric runner's safety gates."""

import fcntl
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv41.engram_distributed.test import (
    run_fabric_ci as runner,
)


class FabricGateTest(unittest.TestCase):
    def test_requires_opt_in_before_any_cuda_or_device_probe(self):
        with patch.dict(os.environ, {"ENGRAM_RUN_CUDA_TESTS": "0"}), patch.object(
            runner, "_wheel_library_environment"
        ) as prepare:
            self.assertEqual(runner.main(), 2)
        prepare.assert_not_called()

    def test_busy_device_is_rejected_and_locks_are_released(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}
        ), patch.object(runner, "_smi", side_effect=["0, GPU-a\n1, GPU-b", "12345"]):
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                with runner.idle_devices(root):
                    self.fail("busy GPU must never be used")
            with (root / "gpu_status" / "0").open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_compatible_existing_file_lock_prevents_gpu_use(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}
        ), patch.object(runner, "_smi", return_value="0, GPU-a\n1, GPU-b") as smi:
            root = Path(directory)
            (root / "gpu_status").mkdir()
            with (root / "gpu_status" / "0").open("a+") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    with runner.idle_devices(root):
                        self.fail("locked GPU must never be used")
            self.assertEqual(smi.call_count, 1)

    def test_uuid_visibility_is_preserved_and_both_gpus_stay_locked(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-c,GPU-b"}
        ), patch.object(
            runner, "_smi", side_effect=["0, GPU-a\n1, GPU-b\n2, GPU-c", "", ""]
        ):
            root = Path(directory)
            with runner.idle_devices(root) as devices:
                self.assertEqual(devices, ["2", "1"])
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-c,GPU-b")
                self.assertEqual(os.environ["ENGRAM_TEST_PEER_DEVICE"], "1")
                for device in devices:
                    with (root / "gpu_status" / device).open("a+") as handle:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_numeric_selection_and_pid_checks_refer_to_nvml_devices(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "2,0"}
        ), patch.object(
            runner, "_smi", side_effect=["0, GPU-a\n1, GPU-b\n2, GPU-c", "", ""]
        ) as smi:
            with runner.idle_devices(Path(directory)) as devices:
                self.assertEqual(devices, ["2", "0"])
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-c,GPU-a")
                self.assertEqual(
                    [call.args for call in smi.call_args_list[1:]],
                    [
                        ("--id=2", "--query-compute-apps=pid"),
                        ("--id=0", "--query-compute-apps=pid"),
                    ],
                )

    def test_real_suite_includes_all_native_cases_with_fixtures(self):
        native, torch = object(), object()
        suite = runner.build_test_suite(native, torch, numa=3)

        def cases(suite):
            for item in suite:
                if isinstance(item, unittest.TestSuite):
                    yield from cases(item)
                else:
                    yield item

        tests = list(cases(suite))
        native_tests = [
            test for test in tests if test.__class__.__module__.endswith(".test_native")
        ]
        from rtp_llm.models_py.modules.dsv41.engram_distributed.test.test_native import (
            NativeStorageTests,
        )

        expected = set(unittest.defaultTestLoader.getTestCaseNames(NativeStorageTests))
        self.assertEqual({t._testMethodName for t in native_tests}, expected)
        self.assertEqual(len(tests), len(expected) + 2)
        for test in native_tests:
            self.assertIs(test.native, native)
            self.assertIs(test.torch, torch)
            self.assertEqual((test.device, test.numa), (0, 3))

    def test_skipped_failed_tests_and_empty_suite_are_not_passes(self):
        class Skipped(unittest.TestCase):
            def runTest(self):
                self.skipTest("missing Fabric")

        class Failed(unittest.TestCase):
            def runTest(self):
                self.fail("native regression")

        for suite in [
            unittest.TestSuite(),
            unittest.TestSuite([Skipped()]),
            unittest.TestSuite([Failed()]),
        ]:
            with self.subTest(suite=suite), patch.object(
                runner,
                "build_test_suite",
                return_value=suite,
            ), patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(runner.run_tests(object(), object(), 0), 1)

    def test_only_executed_passing_tests_report_success(self):
        suite = unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
        with patch.object(runner, "build_test_suite", return_value=suite), patch(
            "sys.stderr", new_callable=io.StringIO
        ):
            self.assertEqual(runner.run_tests(object(), object(), 0), 0)


if __name__ == "__main__":
    unittest.main()
