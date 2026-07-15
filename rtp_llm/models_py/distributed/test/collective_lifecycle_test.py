# type: ignore
import unittest

import rtp_llm.models_py.distributed.collective_lifecycle as cl


class CollectiveLifecycleTest(unittest.TestCase):
    """M3: level-3 collective teardown/rebuild seam. Hermetic -- participants
    register plain callables, so no CUDA/distributed is required."""

    def setUp(self):
        cl.clear_collectives()

    def tearDown(self):
        cl.clear_collectives()

    def _register_order_probes(self, order):
        # Register in dependency order: process group first (DeepEP runs on it),
        # DeepEP second -- mirrors BackendManager._register_deep_sleep_collectives.
        cl.register_collective(
            "process_group",
            teardown=lambda: order.append("teardown:process_group"),
            rebuild=lambda: order.append("rebuild:process_group"),
        )
        cl.register_collective(
            "deepep",
            teardown=lambda: order.append("teardown:deepep"),
            rebuild=lambda: order.append("rebuild:deepep"),
        )

    def test_teardown_runs_in_reverse_registration_order(self):
        order = []
        self._register_order_probes(order)
        cl.run_teardown()
        # DeepEP must be torn down before the process group it rendezvous on.
        self.assertEqual(order, ["teardown:deepep", "teardown:process_group"])

    def test_rebuild_runs_in_registration_order(self):
        order = []
        self._register_order_probes(order)
        cl.run_rebuild()
        # Process group must be rebuilt before DeepEP.
        self.assertEqual(order, ["rebuild:process_group", "rebuild:deepep"])

    def test_rebuild_without_participants_raises(self):
        self.assertEqual(cl.registered_collectives(), [])
        with self.assertRaises(RuntimeError):
            cl.run_rebuild()

    def test_teardown_without_participants_is_noop(self):
        cl.run_teardown()  # must not raise

    def test_register_same_name_replaces_and_keeps_position(self):
        calls = []
        cl.register_collective(
            "process_group",
            teardown=lambda: None,
            rebuild=lambda: calls.append("stale"),
        )
        cl.register_collective(
            "deepep", teardown=lambda: None, rebuild=lambda: calls.append("deepep")
        )
        # Re-register process_group: replaces the stale rebuild, keeps it first.
        cl.register_collective(
            "process_group",
            teardown=lambda: None,
            rebuild=lambda: calls.append("process_group"),
        )
        self.assertEqual(cl.registered_collectives(), ["process_group", "deepep"])
        cl.run_rebuild()
        self.assertEqual(calls, ["process_group", "deepep"])

    def test_clear_collectives(self):
        cl.register_collective("x", teardown=lambda: None, rebuild=lambda: None)
        self.assertEqual(cl.registered_collectives(), ["x"])
        cl.clear_collectives()
        self.assertEqual(cl.registered_collectives(), [])


if __name__ == "__main__":
    unittest.main()
