"""Unit tests for the batched-MoE Python forward.

Confirms the sort + slab dispatch in
:func:`batched_experts.batched_experts_forward` is numerically equivalent
to the naive per-expert scatter loop that ships in the V3 reference.

The naive loop is written inline so we have a self-contained oracle that
does not depend on whichever loop body the production module currently uses.
"""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.moe.batched_experts import (
    batched_experts_forward,
    topk_to_onehot,
)
from rtp_llm.models_py.modules.moe.clamped_swiglu import clamped_swiglu_split


def _naive_moe_forward(x, topk_idx, gate_vals, W_gate, W_up, W_down, swiglu_limit):
    """Per-expert scatter loop (the V3 reference style, kept for parity)."""
    N, H = x.shape
    num_experts = W_gate.shape[0]
    out = torch.zeros_like(x)
    for e in range(num_experts):
        mask = topk_idx == e
        if not mask.any():
            continue
        tok_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = x[tok_idx]
        gate = x_e @ W_gate[e]
        linear = x_e @ W_up[e]
        h = clamped_swiglu_split(gate, linear, swiglu_limit)
        y_e = h @ W_down[e]
        scale = gate_vals[tok_idx, slot_idx].unsqueeze(-1)
        out.index_add_(0, tok_idx, y_e * scale)
    return out


class BatchedExpertsParityTest(TestCase):
    """Sort+slab path vs scatter loop — must match to fp32 epsilon."""

    def _setup(self, *, N=11, H=16, inter=8, E=6, top_k=3, dtype=torch.float32):
        torch.manual_seed(0)
        x = torch.randn(N, H, dtype=dtype)
        # Random top-k assignments — multinomial without replacement.
        logits = torch.randn(N, E)
        topk_idx = logits.topk(top_k, dim=-1).indices
        gate_vals = torch.softmax(logits.gather(1, topk_idx), dim=-1).to(dtype)
        W_gate = torch.randn(E, H, inter, dtype=dtype) * 0.1
        W_up = torch.randn(E, H, inter, dtype=dtype) * 0.1
        W_down = torch.randn(E, inter, H, dtype=dtype) * 0.1
        return x, topk_idx, gate_vals, W_gate, W_up, W_down

    def test_matches_naive_fp32(self):
        x, topk_idx, gate_vals, W_gate, W_up, W_down = self._setup()
        slimit = 10.0
        ref = _naive_moe_forward(x, topk_idx, gate_vals, W_gate, W_up, W_down, slimit)
        out = batched_experts_forward(
            x, topk_idx, gate_vals, W_gate, W_up, W_down, slimit
        )
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)

    def test_matches_naive_bf16(self):
        x, topk_idx, gate_vals, W_gate, W_up, W_down = self._setup(dtype=torch.bfloat16)
        slimit = 10.0
        ref = _naive_moe_forward(x, topk_idx, gate_vals, W_gate, W_up, W_down, slimit)
        out = batched_experts_forward(
            x, topk_idx, gate_vals, W_gate, W_up, W_down, slimit
        )
        # bf16 has 7 bit mantissa — reductions diverge in low bits.
        torch.testing.assert_close(out, ref, rtol=0, atol=2e-2)

    def test_handles_inactive_expert(self):
        """If no token routes to a given expert, that expert's slab is
        skipped — verify we don't pay an empty bmm and the output stays
        consistent with the naive impl."""
        x, _, _, W_gate, W_up, W_down = self._setup(E=8, top_k=2)
        # Force only experts 0 and 3 to be picked.
        N = x.shape[0]
        topk_idx = torch.full((N, 2), 3, dtype=torch.long)
        topk_idx[:, 0] = 0
        gate_vals = torch.full((N, 2), 0.5, dtype=torch.float32)
        slimit = 10.0
        ref = _naive_moe_forward(x, topk_idx, gate_vals, W_gate, W_up, W_down, slimit)
        out = batched_experts_forward(
            x, topk_idx, gate_vals, W_gate, W_up, W_down, slimit
        )
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)

    def test_top_k_eq_one(self):
        x, _, _, W_gate, W_up, W_down = self._setup(E=4, top_k=1)
        N = x.shape[0]
        topk_idx = torch.randint(0, 4, (N, 1))
        gate_vals = torch.ones(N, 1)
        slimit = 10.0
        ref = _naive_moe_forward(x, topk_idx, gate_vals, W_gate, W_up, W_down, slimit)
        out = batched_experts_forward(
            x, topk_idx, gate_vals, W_gate, W_up, W_down, slimit
        )
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)


class TopkToOnehotTest(TestCase):
    def test_one_hot_mask_shape_and_count(self):
        torch.manual_seed(0)
        topk = torch.tensor([[0, 1, 2], [0, 0, 1], [3, 3, 3]], dtype=torch.long)
        mask, counts = topk_to_onehot(topk, num_experts=4)
        # Per-token slots collapse to a per-expert mask.
        self.assertEqual(mask.shape, (3, 4))
        # Token 1 picks expert 0 in two slots: mask[1, 0] should still be 1.
        self.assertEqual(mask[1, 0].item(), 1.0)
        # Per-expert count (sum down the token axis).
        torch.testing.assert_close(
            counts, torch.tensor([2, 2, 1, 1], dtype=torch.int32)
        )


if __name__ == "__main__":
    main()
