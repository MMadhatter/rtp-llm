# type: ignore
import os
import unittest
from unittest.mock import MagicMock, patch

import rtp_llm.models_py.distributed.symm_mem as symm_mem_mod
from rtp_llm.models_py.distributed.symm_mem import (
    get_symm_mem_communicator,
    init_symm_mem_communicator,
)


class SymmMemSleepLevelTest(unittest.TestCase):
    """M1b: symm-mem must be skipped under deep-sleep level-3 (SLEEP_MODE_LEVEL=3),
    because its NVLS multicast binding is incompatible with cuda-checkpoint (PoC
    #7/#7b). Every other config must still create it (fall through to real init).

    Hermetic: the level-3 path returns before touching TorchSymmMemCommunicator, and
    the fall-through path mocks it, so no CUDA/distributed is required.
    """

    def setUp(self):
        symm_mem_mod._symm_mem_comm = None
        self._saved = {
            k: os.environ.get(k) for k in ("ENABLE_SLEEP_MODE", "SLEEP_MODE_LEVEL")
        }

    def tearDown(self):
        symm_mem_mod._symm_mem_comm = None
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _set_env(self, enabled, level):
        os.environ["ENABLE_SLEEP_MODE"] = "1" if enabled else "0"
        os.environ["SLEEP_MODE_LEVEL"] = str(level)

    def test_level3_disables_symm_mem(self):
        self._set_env(enabled=True, level=3)
        # Must short-circuit BEFORE constructing TorchSymmMemCommunicator, so a dummy
        # group is fine and no CUDA is touched.
        with patch.object(symm_mem_mod, "TorchSymmMemCommunicator") as mock_comm:
            result = init_symm_mem_communicator(tp_group=None)
            self.assertIsNone(result)
            mock_comm.assert_not_called()
        self.assertIsNone(get_symm_mem_communicator())

    def test_other_configs_do_not_short_circuit(self):
        # (enabled, level): level-3 only disables when sleep mode is actually on;
        # every other combination must fall through to real initialization.
        cases = [
            (True, 1),  # level-1 sleep: keep symm-mem
            (True, 2),  # level-2 sleep: keep symm-mem
            (False, 3),  # sleep off: SLEEP_MODE_LEVEL=3 is inert -> keep symm-mem
            (False, 1),  # no sleep at all
        ]
        for enabled, level in cases:
            with self.subTest(enabled=enabled, level=level):
                symm_mem_mod._symm_mem_comm = None
                self._set_env(enabled=enabled, level=level)
                sentinel = MagicMock()
                sentinel.disabled = False
                with patch.object(
                    symm_mem_mod, "TorchSymmMemCommunicator", return_value=sentinel
                ) as mock_comm, patch("torch.cuda.current_device", return_value=0):
                    result = init_symm_mem_communicator(tp_group=MagicMock())
                    mock_comm.assert_called_once()
                    self.assertIs(result, sentinel)
                self.assertIs(get_symm_mem_communicator(), sentinel)

    def test_missing_env_defaults_to_enabled(self):
        # No sleep env at all -> symm-mem must NOT be disabled.
        os.environ.pop("ENABLE_SLEEP_MODE", None)
        os.environ.pop("SLEEP_MODE_LEVEL", None)
        self.assertFalse(symm_mem_mod._sleep_mode_level_disables_symm_mem())

    def test_malformed_level_is_not_disabled(self):
        os.environ["ENABLE_SLEEP_MODE"] = "1"
        os.environ["SLEEP_MODE_LEVEL"] = "not-an-int"
        self.assertFalse(symm_mem_mod._sleep_mode_level_disables_symm_mem())


if __name__ == "__main__":
    unittest.main()
