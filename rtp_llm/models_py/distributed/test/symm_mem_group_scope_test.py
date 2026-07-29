import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.distributed import symm_mem_group_scope as scope


class SymmMemGroupScopeTest(unittest.TestCase):
    def setUp(self) -> None:
        scope._reset_for_test()
        self.env = patch.dict(
            os.environ,
            {
                scope.KEEPER_ENABLE_ENV: "1",
                scope.LOCAL_GPUS_ENV: "0,1,2,3",
            },
            clear=False,
        )
        self.env.start()

    def tearDown(self) -> None:
        scope._reset_for_test()
        self.env.stop()

    @staticmethod
    def _topology(world_rank=4, local_rank=0, local_world_size=4, world_size=8):
        return SimpleNamespace(
            world_rank=world_rank,
            local_rank=local_rank,
            local_world_size=local_world_size,
            world_size=world_size,
        )

    def _configure(self, ranks, owner="test"):
        group = object()
        with patch.object(scope.dist, "is_initialized", return_value=True), patch.object(
            scope.dist, "get_process_group_ranks", return_value=list(ranks)
        ):
            return scope.configure_group_scope(group, owner=owner)

    def test_complete_local_group_selects_posix_from_actual_ranks(self) -> None:
        scope.configure_rank_topology(self._topology())

        decision = self._configure([4, 5, 6, 7], owner="torch_tp")

        self.assertEqual(decision.scope, scope.SymmMemGroupScope.LOCAL)
        self.assertEqual(decision.policy, scope.SymmMemHandlePolicy.LOCAL_POSIX)
        self.assertEqual(os.environ[scope.HANDLE_POLICY_ENV], "local_posix")

    def test_cross_node_group_preserves_native_handle_selection(self) -> None:
        scope.configure_rank_topology(self._topology())

        decision = self._configure(range(8), owner="dsv4_mega_moe")

        self.assertEqual(decision.scope, scope.SymmMemGroupScope.CROSS_NODE)
        self.assertEqual(decision.policy, scope.SymmMemHandlePolicy.NATIVE)
        self.assertEqual(os.environ[scope.HANDLE_POLICY_ENV], "native")

    def test_local_subgroup_preserves_native_handle_selection(self) -> None:
        scope.configure_rank_topology(self._topology())

        decision = self._configure([4, 5], owner="tp2")

        self.assertEqual(decision.scope, scope.SymmMemGroupScope.LOCAL)
        self.assertEqual(decision.policy, scope.SymmMemHandlePolicy.NATIVE)

    def test_cross_node_subgroup_does_not_assume_fabric(self) -> None:
        scope.configure_rank_topology(self._topology())

        decision = self._configure([0, 1, 4, 5], owner="cp4")

        self.assertEqual(decision.scope, scope.SymmMemGroupScope.CROSS_NODE)
        self.assertEqual(decision.policy, scope.SymmMemHandlePolicy.NATIVE)

    def test_process_rejects_mixed_local_and_cross_node_policies(self) -> None:
        scope.configure_rank_topology(self._topology())
        self._configure([4, 5, 6, 7], owner="torch_tp")

        with self.assertRaisesRegex(RuntimeError, "cannot mix"):
            self._configure(range(8), owner="dsv4_mega_moe")

    def test_disabled_keeper_does_not_touch_process_environment(self) -> None:
        os.environ[scope.KEEPER_ENABLE_ENV] = "0"
        scope.configure_rank_topology(self._topology())

        with patch.object(scope.dist, "is_initialized", return_value=False):
            self.assertIsNone(scope.configure_group_scope(object(), owner="disabled"))
        self.assertNotIn(scope.HANDLE_POLICY_ENV, os.environ)

    def test_cross_node_scope_barriers_before_releasing_fabric_creator(self) -> None:
        scope.configure_rank_topology(self._topology())
        group = object()
        events = []
        with patch.object(
            scope.dist, "is_initialized", return_value=True
        ), patch.object(
            scope.dist, "get_process_group_ranks", return_value=list(range(8))
        ), patch.object(
            scope.dist, "barrier", side_effect=lambda **_: events.append("barrier")
        ), patch.object(
            scope, "_pending_fabric_backing_fences", return_value=1
        ), patch.object(
            scope,
            "_release_fabric_backing_fences",
            side_effect=lambda _owner: events.append("release") or 1,
        ):
            with scope.symm_mem_allocation_scope(group, owner="mega"):
                self.assertEqual(
                    "1", os.environ[scope.BACKING_BROKER_ACTIVE_ENV]
                )
                events.append("allocate")

        self.assertEqual(["allocate", "barrier", "release"], events)
        self.assertNotIn(scope.BACKING_BROKER_ACTIVE_ENV, os.environ)

    def test_non_fabric_cross_node_scope_bypasses_keeper_coordination(self) -> None:
        scope.configure_rank_topology(self._topology())
        group = object()
        with patch.object(
            scope.dist, "is_initialized", return_value=True
        ), patch.object(
            scope.dist, "get_process_group_ranks", return_value=list(range(8))
        ), patch.object(scope.dist, "barrier") as barrier, patch.object(
            scope, "_pending_fabric_backing_fences", return_value=0
        ) as pending, patch.object(
            scope, "_release_fabric_backing_fences"
        ) as release:
            with scope.symm_mem_allocation_scope(group, owner="rdma"):
                self.assertEqual(
                    "1", os.environ[scope.BACKING_BROKER_ACTIVE_ENV]
                )

        pending.assert_called_once_with("rdma")
        barrier.assert_not_called()
        release.assert_not_called()
        self.assertNotIn(scope.BACKING_BROKER_ACTIVE_ENV, os.environ)

    def test_non_fabric_local_subgroup_bypasses_keeper_coordination(self) -> None:
        scope.configure_rank_topology(self._topology())
        group = object()
        with patch.object(
            scope.dist, "is_initialized", return_value=True
        ), patch.object(
            scope.dist, "get_process_group_ranks", return_value=[4, 5]
        ), patch.object(scope.dist, "barrier") as barrier, patch.object(
            scope, "_pending_fabric_backing_fences", return_value=0
        ) as pending, patch.object(
            scope, "_release_fabric_backing_fences"
        ) as release:
            with scope.symm_mem_allocation_scope(group, owner="local_rdma"):
                self.assertEqual(
                    "1", os.environ[scope.BACKING_BROKER_ACTIVE_ENV]
                )

        pending.assert_called_once_with("local_rdma")
        barrier.assert_not_called()
        release.assert_not_called()

    def test_failed_cross_node_allocation_releases_without_barrier(self) -> None:
        scope.configure_rank_topology(self._topology())
        group = object()
        with patch.object(
            scope.dist, "is_initialized", return_value=True
        ), patch.object(
            scope.dist, "get_process_group_ranks", return_value=list(range(8))
        ), patch.object(scope.dist, "barrier") as barrier, patch.object(
            scope, "_pending_fabric_backing_fences", return_value=1
        ), patch.object(
            scope, "_release_fabric_backing_fences", return_value=1
        ) as release:
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                with scope.symm_mem_allocation_scope(group, owner="mega"):
                    raise RuntimeError("allocation failed")

        barrier.assert_not_called()
        release.assert_called_once_with("mega")
        self.assertNotIn(scope.BACKING_BROKER_ACTIVE_ENV, os.environ)

    def test_failed_cross_node_barrier_releases_creator_and_preserves_error(
        self,
    ) -> None:
        scope.configure_rank_topology(self._topology())
        group = object()
        with patch.object(
            scope.dist, "is_initialized", return_value=True
        ), patch.object(
            scope.dist, "get_process_group_ranks", return_value=list(range(8))
        ), patch.object(
            scope.dist, "barrier", side_effect=RuntimeError("barrier failed")
        ), patch.object(
            scope, "_pending_fabric_backing_fences", return_value=1
        ), patch.object(
            scope, "_release_fabric_backing_fences", return_value=1
        ) as release:
            with self.assertRaisesRegex(RuntimeError, "barrier failed"):
                with scope.symm_mem_allocation_scope(group, owner="mega"):
                    pass

        release.assert_called_once_with("mega")


if __name__ == "__main__":
    unittest.main()
