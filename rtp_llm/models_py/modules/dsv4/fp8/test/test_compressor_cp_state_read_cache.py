from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    _cp_state_read_needed_mask,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace


class CompressorCPStateReadCacheTest(unittest.TestCase):
    def test_needed_mask_selects_only_prefix_tail_blocks(self) -> None:
        block_table = torch.arange(12, dtype=torch.int64).view(2, 6)
        mask = _cp_state_read_needed_mask(
            block_table,
            seq_start_per_req=torch.tensor([0, 65], dtype=torch.int32),
            token_count=8,
            state_tokens_per_block=16,
        )

        expected = torch.zeros_like(block_table, dtype=torch.bool)
        # Request 0 has no prefix. Request 1 reads suffix positions [58, 64],
        # which touches state-block columns 3 and 4.
        expected[1, 3] = True
        expected[1, 4] = True
        self.assertTrue(torch.equal(mask, expected))

    def test_needed_mask_wraps_ring_block_columns(self) -> None:
        block_table = torch.ones((1, 4), dtype=torch.int64)
        mask = _cp_state_read_needed_mask(
            block_table,
            seq_start_per_req=torch.tensor([65], dtype=torch.int64),
            token_count=8,
            state_tokens_per_block=16,
        )

        expected = torch.zeros_like(block_table, dtype=torch.bool)
        # Positions [58, 64] touch columns 3 and 4 % 4 == 0.
        expected[0, 3] = True
        expected[0, 0] = True
        self.assertTrue(torch.equal(mask, expected))

    def test_workspace_state_read_buffers_are_reused_and_grow(self) -> None:
        ws = PrefillWorkspace(
            torch.device("cpu"), q_rows=1, q_dim=1, reserve_cp=False, align_bytes=1
        )

        first = ws.state_read_cache(2, 3, 4, torch.float32)
        second = ws.state_read_cache(1, 3, 4, torch.float32)
        self.assertEqual(tuple(second.shape), (1, 3, 4))
        self.assertEqual(first.data_ptr(), second.data_ptr())

        bigger = ws.state_read_cache(4, 3, 4, torch.float32)
        self.assertEqual(tuple(bigger.shape), (4, 3, 4))
        self.assertGreaterEqual(bigger.numel(), first.numel())


if __name__ == "__main__":
    unittest.main()
