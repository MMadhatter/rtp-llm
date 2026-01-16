"""Unit tests for user_buffers.py

This module tests the UserBufferCommunicator class which provides
GPU-to-GPU communication using CUDA IPC shared buffers.
"""

import logging
import multiprocessing as mp
import os
import unittest
from unittest import mock

logging.basicConfig(level=logging.INFO)

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.user_buffers import (
    UserBufferCommunicator,
    get_user_buffers_communicator,
    init_user_buffers_communicator,
)
from rtp_llm.test.utils.port_util import PortManager


def _init_process_group(rank: int, world_size: int, master_port: int):
    """Initialize distributed process group"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def _cleanup_process_group():
    """Cleanup distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


# Test functions that operate on a communicator instance
def _test_basic_properties(comm: UserBufferCommunicator, rank: int, world_size: int):
    """Test basic properties of the communicator"""
    assert comm.local_rank == rank
    assert comm.world_size == world_size
    assert comm.buffer_size == 1024 * 1024

    # Test device property
    expected_device = torch.device(f"cuda:{rank}")
    assert comm.device == expected_device
    logging.info(f"Rank {rank}: basic properties test passed")


def _test_buffer_internals(comm: UserBufferCommunicator, rank: int):
    """Test that communicator maintains buffer references and streams"""
    # Check buffer pointers and handles are initialized
    assert comm._buffer_ptrs is not None
    assert comm._communicator_ptr is not None
    assert comm._ub_handle is not None

    # Check that streams are created for all ranks
    assert len(comm._communicate_streams) == comm.world_size
    # Check that current stream is set
    assert comm._current_stream is not None
    logging.info(f"Rank {rank}: buffer internals test passed")


def _test_synchronize(comm: UserBufferCommunicator, rank: int):
    """Test synchronize method"""
    # This should not raise any exception
    comm.synchronize()
    logging.info(f"Rank {rank}: synchronize test passed")


def _test_send_valid_tensor(comm: UserBufferCommunicator, rank: int):
    """Test send with valid tensor size (mocked)"""
    # Create tensor within buffer size
    tensor = torch.randn(100, dtype=torch.float32, device=comm.device)

    # Mock the underlying function to avoid actual send
    with mock.patch("rtp_llm.ops.compute_ops.userbuffers_send"):
        # Should not raise an exception
        comm.send(tensor, dst=rank)
    logging.info(f"Rank {rank}: send valid tensor test passed")


def _test_recv_returns_tensor(comm: UserBufferCommunicator, rank: int):
    """Test that recv returns the tensor (mocked)"""
    # Create tensor to receive data into
    recv_tensor = torch.zeros(100, dtype=torch.float32, device=comm.device)

    # Mock the underlying function
    with mock.patch("rtp_llm.ops.compute_ops.userbuffers_recv"):
        result = comm.recv(recv_tensor, src=rank)
        # Verify the returned tensor is the same object
        assert result is recv_tensor
    logging.info(f"Rank {rank}: recv returns tensor test passed")


def _test_all_gather_not_implemented(comm: UserBufferCommunicator, rank: int):
    """Test that all_gather raises NotImplementedError"""
    src_tensor = torch.randn(100, dtype=torch.float32, device=comm.device)
    dst_tensor = torch.zeros(100, dtype=torch.float32, device=comm.device)

    # Should raise NotImplementedError
    try:
        comm.all_gather(src_tensor, dst_tensor)
        assert False, "all_gather should raise NotImplementedError"
    except NotImplementedError:
        pass
    logging.info(f"Rank {rank}: all_gather not implemented test passed")


# Worker functions that create communicator and run all tests
def _test_all_interfaces_worker(
    rank: int, world_size: int, master_port: int, buffer_size: int = 1024 * 1024
):
    """Worker function that creates one communicator and tests all interfaces"""
    logging.info(f"Rank {rank}: starting all interfaces test")
    _init_process_group(rank, world_size, master_port)

    try:
        # Create communicator once
        comm = UserBufferCommunicator(
            group=dist.GroupMember.WORLD,
            local_rank=rank,
            world_size=world_size,
            buffer_size=buffer_size,
        )

        try:
            # Test all interfaces using the same communicator
            _test_basic_properties(comm, rank, world_size)
            _test_buffer_internals(comm, rank)
            _test_synchronize(comm, rank)
            _test_send_valid_tensor(comm, rank)
            _test_recv_returns_tensor(comm, rank)
            _test_all_gather_not_implemented(comm, rank)

            # Synchronize all ranks before cleanup
            if dist.is_initialized():
                dist.barrier()

            logging.info(f"Rank {rank}: all tests passed")
        finally:
            comm.cleanup()
    finally:
        _cleanup_process_group()


def _test_global_communicator_worker(rank: int, world_size: int, master_port: int):
    """Worker function to test global communicator singleton"""
    logging.info(f"Rank {rank}: testing global communicator")
    _init_process_group(rank, world_size, master_port)

    try:
        buffer_size = 1024 * 1024

        # Initialize global communicator
        comm1 = init_user_buffers_communicator(
            group=dist.GroupMember.WORLD,
            world_rank=rank,
            world_size=world_size,
            buffer_size=buffer_size,
        )

        try:
            assert comm1.local_rank == rank
            assert comm1.world_size == world_size

            # Verify we can get the global communicator
            retrieved_comm = get_user_buffers_communicator()
            assert retrieved_comm is comm1

            # Try to initialize again - should return same instance
            comm2 = init_user_buffers_communicator(
                group=dist.GroupMember.WORLD,
                world_rank=rank,
                world_size=world_size,
                buffer_size=buffer_size,
            )
            assert comm1 is comm2

            logging.info(f"Rank {rank}: global communicator test passed")
        finally:
            comm1.cleanup()
    finally:
        _cleanup_process_group()


class TestUserBufferCommunicator(unittest.TestCase):
    """Test UserBufferCommunicator with single process and multiprocess scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Set spawn method for multiprocessing
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set

        self.port_manager = PortManager()

    def tearDown(self):
        """Clean up after tests"""
        pass

    def _run_single_process_test(self, worker_func, test_name: str):
        """Helper to run a single process test"""
        if torch.cuda.device_count() < 1:
            self.skipTest("Not enough GPUs")

        ports, locks = self.port_manager.get_consecutive_ports(1)
        master_port = ports[0]

        try:
            worker_func(rank=0, world_size=1, master_port=master_port)
        except Exception as e:
            raise RuntimeError(f"Test {test_name} failed: {e}")
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    def _run_multi_process_test(self, worker_func, world_size: int, test_name: str):
        """Helper to run a multi-process test"""
        if torch.cuda.device_count() < world_size:
            self.skipTest(f"Need at least {world_size} GPUs")

        ports, locks = self.port_manager.get_consecutive_ports(1)
        master_port = ports[0]

        try:
            processes = []
            for rank in range(world_size):
                p = mp.Process(
                    target=worker_func,
                    args=(rank, world_size, master_port),
                    name=f"rank-{rank}",
                )
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join(timeout=120)
                if p.exitcode != 0:
                    raise RuntimeError(
                        f"Process {p.name} exited with code {p.exitcode}"
                    )
        finally:
            # Release port locks
            for lock in locks:
                lock.__exit__(None, None, None)

    # Main tests - single communicator instance tests all interfaces
    def test_all_interfaces_single_process(self):
        """Test all interfaces with single process"""
        self._run_single_process_test(
            _test_all_interfaces_worker, "all_interfaces_single"
        )

    def test_all_interfaces_multi_process(self):
        """Test all interfaces with multiple processes"""
        self._run_multi_process_test(
            _test_all_interfaces_worker, world_size=2, test_name="all_interfaces_multi"
        )

    def test_global_communicator_single_process(self):
        """Test global communicator singleton pattern"""
        self._run_single_process_test(
            _test_global_communicator_worker, "global_communicator_single"
        )

    def test_global_communicator_multi_process(self):
        """Test global communicator with multiple processes"""
        self._run_multi_process_test(
            _test_global_communicator_worker,
            world_size=2,
            test_name="global_communicator_multi",
        )

    # Custom buffer size test
    def test_custom_buffer_size(self):
        """Test initialization with custom buffer size"""
        if torch.cuda.device_count() < 1:
            self.skipTest("Not enough GPUs")

        ports, locks = self.port_manager.get_consecutive_ports(1)
        master_port = ports[0]

        try:
            custom_size = 512 * 1024  # 512KB
            _test_all_interfaces_worker(
                rank=0, world_size=1, master_port=master_port, buffer_size=custom_size
            )
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    # Error handling tests
    def test_send_oversized_data(self):
        """Test sending data larger than buffer size"""
        if torch.cuda.device_count() < 1:
            self.skipTest("Not enough GPUs")

        ports, locks = self.port_manager.get_consecutive_ports(1)
        master_port = ports[0]

        try:
            _init_process_group(rank=0, world_size=1, master_port=master_port)
            try:
                comm = UserBufferCommunicator(
                    group=dist.GroupMember.WORLD,
                    local_rank=0,
                    world_size=1,
                    buffer_size=1024,  # 1KB buffer
                )

                try:
                    # Create tensor larger than buffer
                    large_tensor = torch.randn(
                        2048, dtype=torch.float32, device="cuda:0"
                    )

                    with self.assertRaises(ValueError) as context:
                        comm.send(large_tensor, dst=0)

                    self.assertIn("exceeds buffer size", str(context.exception))
                finally:
                    comm.cleanup()
            finally:
                _cleanup_process_group()
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    def test_init_invalid_local_rank(self):
        """Test initialization with invalid local rank"""
        ports, locks = self.port_manager.get_consecutive_ports(1)
        master_port = ports[0]

        try:
            _init_process_group(rank=0, world_size=1, master_port=master_port)
            try:
                # Try to create communicator with invalid rank (exceeds GPU count)
                invalid_rank = torch.cuda.device_count() + 1
                with self.assertRaises(RuntimeError) as context:
                    UserBufferCommunicator(
                        group=dist.GroupMember.WORLD,
                        local_rank=invalid_rank,
                        world_size=1,
                    )
                self.assertIn("exceeds available GPU count", str(context.exception))
            finally:
                _cleanup_process_group()
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    def test_cleanup_multiple_times(self):
        """Test that cleanup can be called multiple times safely"""
        if torch.cuda.device_count() < 1:
            self.skipTest("Not enough GPUs")

        ports, locks = self.port_manager.get_consecutive_ports(1)
        master_port = ports[0]

        try:
            _init_process_group(rank=0, world_size=1, master_port=master_port)
            try:
                comm = UserBufferCommunicator(
                    group=dist.GroupMember.WORLD,
                    local_rank=0,
                    world_size=1,
                )

                # Calling cleanup multiple times should not raise
                comm.cleanup()
                comm.cleanup()
                comm.cleanup()
            finally:
                _cleanup_process_group()
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)


class TestUserBufferCommunicatorNoCUDA(unittest.TestCase):
    """Test behavior when CUDA is not available"""

    def test_init_without_cuda_raises_error(self):
        """Test that initialization fails gracefully without CUDA"""
        if torch.cuda.is_available():
            self.skipTest("CUDA is available, skipping no-CUDA test")

        with self.assertRaises(RuntimeError) as context:
            UserBufferCommunicator(
                group=None,  # Won't be used since CUDA check comes first
                local_rank=0,
                world_size=1,
            )
        self.assertIn("CUDA is not available", str(context.exception))


if __name__ == "__main__":
    unittest.main()
