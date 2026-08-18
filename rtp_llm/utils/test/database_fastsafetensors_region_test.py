import contextlib
import sys
import types
import unittest
from typing import Iterator, List, Tuple

from rtp_llm.utils.database import CkptDatabase


class _FakeCkptFile:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name


class FastsafetensorsRegionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_fastsafetensors = sys.modules.get("fastsafetensors")
        self._had_fastsafetensors = "fastsafetensors" in sys.modules

    def tearDown(self) -> None:
        if self._had_fastsafetensors:
            sys.modules["fastsafetensors"] = self._saved_fastsafetensors
        else:
            sys.modules.pop("fastsafetensors", None)

    def test_allocation_context_starts_after_auto_loader_init(self) -> None:
        events: List[Tuple[str, bool]] = []
        in_region = False

        def active() -> bool:
            return in_region

        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeAutoLoader:
            def __init__(self, pg, files, device) -> None:
                events.append(("init", active()))

            def iterate_weights(self):
                events.append(("iterate_enter", active()))
                yield "weight", object()
                events.append(("iterate_after_yield", active()))

            def close(self) -> None:
                events.append(("close", active()))

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.__path__ = []
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.AutoLoader = FakeAutoLoader
        sys.modules["fastsafetensors"] = fake_module

        @contextlib.contextmanager
        def allocation_context() -> Iterator[None]:
            nonlocal in_region
            events.append(("context_enter", active()))
            in_region = True
            try:
                yield
            finally:
                in_region = False
                events.append(("context_exit", active()))

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        for _key, _tensor in database.fastsafetensors_weights_iterator(
            "cuda",
            use_tqdm_on_load=False,
            allocation_context=allocation_context,
        ):
            events.append(("consumer", active()))

        self.assertEqual(
            events,
            [
                ("init", False),
                ("context_enter", False),
                ("iterate_enter", True),
                ("consumer", True),
                ("iterate_after_yield", True),
                ("context_exit", False),
                ("close", False),
            ],
        )

    def test_stacked_experts_are_cloned_and_renamed(self) -> None:
        cloned = []

        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeExpertSlice:
            def __init__(self, expert_id: int) -> None:
                self.expert_id = expert_id

            def clone(self):
                result = object()
                cloned.append((self.expert_id, result))
                return result

        class FakeStackedTensor:
            shape = (3, 2)

            def __getitem__(self, expert_id: int):
                return FakeExpertSlice(expert_id)

        class FakeAutoLoader:
            def __init__(self, pg, files, device) -> None:
                pass

            def iterate_weights(self):
                yield "stacked", FakeStackedTensor()
                yield "plain", "plain-tensor"

            def close(self) -> None:
                pass

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.__path__ = []
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.AutoLoader = FakeAutoLoader
        sys.modules["fastsafetensors"] = fake_module

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        result = list(
            database.fastsafetensors_weights_iterator(
                "cuda",
                use_tqdm_on_load=False,
                stacked_key_config={"stacked": "experts.{expert_id}.weight"},
            )
        )

        self.assertEqual(
            [key for key, _ in result],
            [
                "experts.0.weight",
                "experts.1.weight",
                "experts.2.weight",
                "plain",
            ],
        )
        self.assertEqual([expert_id for expert_id, _ in cloned], [0, 1, 2])
        self.assertEqual([tensor for _, tensor in result[:3]], [x[1] for x in cloned])
        self.assertEqual(result[3], ("plain", "plain-tensor"))


if __name__ == "__main__":
    unittest.main()
