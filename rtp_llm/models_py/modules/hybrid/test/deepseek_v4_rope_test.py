"""Parity tests for the dual-base partial-RoPE cache.

Compares :class:`DeepSeekV4DualRope` against the un-cached single-base
``apply_partial_rope`` from ``hca_attention``. Each base is exercised
independently (``apply_q`` ↔ ``rope_theta_q``, ``apply_k`` ↔
``rope_theta_k``) and we verify that ``inverse=True`` undoes the forward
rotation exactly.

Reference: vLLM PR #40760 ``deepseek_scaling_rope.py`` and SGLang PR
#23600 ``deepseek_v4_rope.py``.
"""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.hybrid.deepseek_v4_rope import DeepSeekV4DualRope
from rtp_llm.models_py.modules.hybrid.hca_attention import (
    apply_partial_rope,
    precompute_rope_cache,
)


class DeepSeekV4DualRopeTest(TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.rope_dim = 16
        self.head_dim = 32
        self.B = 2
        self.H = 4
        self.T_q = 8
        self.T_k = 5
        self.theta_q = 10_000.0
        self.theta_k = 160_000.0
        self.rope = DeepSeekV4DualRope(
            rope_head_dim=self.rope_dim,
            max_pos_q=64,
            max_pos_k=32,
            rope_theta_q=self.theta_q,
            rope_theta_k=self.theta_k,
        )

    def test_apply_q_matches_partial_rope_with_q_base(self):
        x = torch.randn(self.B, self.H, self.T_q, self.head_dim)
        positions = torch.arange(self.T_q)
        cos, sin = precompute_rope_cache(self.T_q, self.rope_dim, base=self.theta_q)
        ref = apply_partial_rope(x, cos, sin, self.rope_dim)

        out = self.rope.apply_q(x, positions)
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)

    def test_apply_k_matches_partial_rope_with_k_base(self):
        x = torch.randn(self.B, self.H, self.T_k, self.head_dim)
        positions = torch.arange(self.T_k)
        cos, sin = precompute_rope_cache(self.T_k, self.rope_dim, base=self.theta_k)
        ref = apply_partial_rope(x, cos, sin, self.rope_dim)

        out = self.rope.apply_k(x, positions)
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)

    def test_q_and_k_use_different_bases(self):
        x = torch.randn(self.B, self.H, self.T_q, self.head_dim)
        positions = torch.arange(self.T_q)
        # Same input, same positions, different bases → different outputs.
        out_q = self.rope.apply_q(x, positions)
        out_k = self.rope.apply_k(x, positions)
        self.assertFalse(torch.allclose(out_q, out_k))

    def test_inverse_undoes_forward_q(self):
        x = torch.randn(self.B, self.H, self.T_q, self.head_dim)
        positions = torch.arange(self.T_q)
        rotated = self.rope.apply_q(x, positions)
        recovered = self.rope.apply_q(rotated, positions, inverse=True)
        torch.testing.assert_close(recovered, x, rtol=1e-5, atol=1e-5)

    def test_inverse_undoes_forward_k(self):
        x = torch.randn(self.B, self.H, self.T_k, self.head_dim)
        positions = torch.arange(self.T_k)
        rotated = self.rope.apply_k(x, positions)
        recovered = self.rope.apply_k(rotated, positions, inverse=True)
        torch.testing.assert_close(recovered, x, rtol=1e-5, atol=1e-5)

    def test_compressed_position_indexing(self):
        # K branch is meant to be indexed at compressed positions like i*m + m//2.
        # Exercise that custom indexing returns the right cached entry.
        m = 4
        positions = torch.tensor([i * m + m // 2 for i in range(self.T_k)])
        x = torch.randn(self.B, self.H, self.T_k, self.head_dim)
        out = self.rope.apply_k(x, positions)
        # Sanity: shape preserved and values finite.
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        # Compare against a direct cache build with these exact positions.
        cos, sin = precompute_rope_cache(
            int(positions.max()) + 1, self.rope_dim, base=self.theta_k
        )
        ref = apply_partial_rope(x, cos[positions], sin[positions], self.rope_dim)
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)


if __name__ == "__main__":
    main()
