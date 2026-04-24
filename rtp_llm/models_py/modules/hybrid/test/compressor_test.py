"""Tests for token-level compressors used by DeepSeek-V4 CSA + HCA paths.

Verifies the closed-form expectations:

  * shapes collapse from ``n`` to ``ceil(n / m)``
  * weights inside one block softmax to 1
  * tail padding contributes nothing (matches truncated reference)
  * for CSA, block 0's previous-window contributes zero
  * batch invariance — independent rows don't bleed into each other
"""

import math
from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.hybrid.compressor import (
    CsaCompressor,
    HcaCompressor,
    csa_compress,
    hca_compress,
)


def _set_seed(s: int = 0) -> None:
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
class HcaCompressTest(TestCase):
    def setUp(self):
        _set_seed(0)
        self.d, self.c, self.m_prime = 16, 8, 4
        self.W_kv = torch.randn(self.d, self.c) * 0.1
        self.W_z = torch.randn(self.d, self.c) * 0.1
        self.bias_pos = torch.randn(self.m_prime, self.c) * 0.1

    def _ref(self, H: torch.Tensor) -> torch.Tensor:
        """Naive Eq. 22-23 reference, no padding logic."""
        *prefix, n, _ = H.shape
        assert n % self.m_prime == 0
        n_blocks = n // self.m_prime
        C = H @ self.W_kv
        Z = H @ self.W_z
        out = torch.empty(*prefix, n_blocks, self.c, dtype=H.dtype)
        for i in range(n_blocks):
            sl = slice(i * self.m_prime, (i + 1) * self.m_prime)
            S = torch.softmax((Z[..., sl, :] + self.bias_pos).float(), dim=-2).to(H.dtype)
            out[..., i, :] = (S * C[..., sl, :]).sum(dim=-2)
        return out

    def test_shape(self):
        H = torch.randn(2, 3, 4 * self.m_prime, self.d)
        out = hca_compress(H, self.W_kv, self.W_z, self.bias_pos, self.m_prime)
        self.assertEqual(out.shape, (2, 3, 4, self.c))

    def test_matches_naive_no_padding(self):
        H = torch.randn(5, 4 * self.m_prime, self.d)
        out = hca_compress(H, self.W_kv, self.W_z, self.bias_pos, self.m_prime)
        ref = self._ref(H)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    def test_padding_extends_one_block(self):
        # n = 13, m' = 4 -> 4 blocks (last block has 3 real + 1 pad).
        n = 13
        H = torch.randn(n, self.d)
        out = hca_compress(H, self.W_kv, self.W_z, self.bias_pos, self.m_prime)
        self.assertEqual(out.shape, (math.ceil(n / self.m_prime), self.c))

    def test_padding_does_not_leak(self):
        """Last compressed entry must equal a hand-computed last-block softmax
        that ignores the padded tail row."""
        n_real = 3  # plus 1 pad row -> block of size 4
        H_block = torch.randn(n_real, self.d)
        out = hca_compress(
            H_block, self.W_kv, self.W_z, self.bias_pos, self.m_prime,
        )
        # Hand computation: pad row goes to -inf softmax weight, contributes 0.
        C = H_block @ self.W_kv
        Z = H_block @ self.W_z
        # Only the first n_real positions of bias_pos matter once pad is masked.
        S = torch.softmax((Z + self.bias_pos[:n_real]).float(), dim=-2).to(H_block.dtype)
        expected = (S * C).sum(dim=-2)
        torch.testing.assert_close(out[0], expected, rtol=1e-4, atol=1e-5)

    def test_softmax_weights_sum_to_one_per_block(self):
        """Indirect: feed all-zero H. Then C_comp_i = sum(S * 0) = 0,
        but softmax weights are well-defined; check by replacing W_kv with
        an identity-ish projection so the output equals mean-of-block."""
        m_p = 4
        d = c = 4
        H = torch.randn(2 * m_p, d)
        W_kv = torch.eye(d, c)            # C == H_pad
        W_z = torch.zeros(d, c)           # Z == 0 -> uniform softmax over m'
        bias_pos = torch.zeros(m_p, c)
        out = hca_compress(H, W_kv, W_z, bias_pos, m_p)
        # Uniform softmax means C_comp_i = mean of the block.
        expected = H.view(2, m_p, d).mean(dim=-2)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)

    def test_batch_independence(self):
        """Stacking two independent inputs gives the same per-row outputs."""
        H1 = torch.randn(self.m_prime * 3, self.d)
        H2 = torch.randn(self.m_prime * 3, self.d)
        H = torch.stack([H1, H2], dim=0)
        out_batched = hca_compress(H, self.W_kv, self.W_z, self.bias_pos, self.m_prime)
        out1 = hca_compress(H1, self.W_kv, self.W_z, self.bias_pos, self.m_prime)
        out2 = hca_compress(H2, self.W_kv, self.W_z, self.bias_pos, self.m_prime)
        torch.testing.assert_close(out_batched[0], out1)
        torch.testing.assert_close(out_batched[1], out2)

    def test_dtype_preserved_bf16(self):
        H = torch.randn(self.m_prime * 2, self.d, dtype=torch.bfloat16)
        W_kv = self.W_kv.to(torch.bfloat16)
        W_z = self.W_z.to(torch.bfloat16)
        bias = self.bias_pos.to(torch.bfloat16)
        out = hca_compress(H, W_kv, W_z, bias, self.m_prime)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_rejects_bad_args(self):
        H = torch.randn(self.m_prime, self.d)
        with self.assertRaises(ValueError):
            hca_compress(H, self.W_kv, self.W_z, self.bias_pos, m_prime=0)
        with self.assertRaises(ValueError):
            # bias_pos wrong length
            hca_compress(H, self.W_kv, self.W_z, torch.zeros(self.m_prime + 1, self.c), self.m_prime)
        with self.assertRaises(ValueError):
            # W_z wrong shape
            hca_compress(H, self.W_kv, torch.zeros(self.d + 1, self.c), self.bias_pos, self.m_prime)


# ---------------------------------------------------------------------------
class CsaCompressTest(TestCase):
    def setUp(self):
        _set_seed(1)
        self.d, self.c, self.m = 12, 6, 4
        self.W_a_kv = torch.randn(self.d, self.c) * 0.1
        self.W_b_kv = torch.randn(self.d, self.c) * 0.1
        self.W_a_z = torch.randn(self.d, self.c) * 0.1
        self.W_b_z = torch.randn(self.d, self.c) * 0.1
        self.bias_a = torch.randn(self.m, self.c) * 0.1
        self.bias_b = torch.randn(self.m, self.c) * 0.1

    def _ref(self, H: torch.Tensor) -> torch.Tensor:
        *prefix, n, _ = H.shape
        assert n % self.m == 0
        n_blocks = n // self.m
        Ca = H @ self.W_a_kv
        Cb = H @ self.W_b_kv
        Za = H @ self.W_a_z
        Zb = H @ self.W_b_z
        out = torch.empty(*prefix, n_blocks, self.c, dtype=H.dtype)
        for i in range(n_blocks):
            sl_a = slice(i * self.m, (i + 1) * self.m)
            za = Za[..., sl_a, :] + self.bias_a       # (..., m, c)
            ca = Ca[..., sl_a, :]
            if i == 0:
                # Build joint logits where the prev-block half is -inf.
                zb = torch.full_like(za, float("-inf"))
                cb = torch.zeros_like(ca)
            else:
                sl_b = slice((i - 1) * self.m, i * self.m)
                zb = Zb[..., sl_b, :] + self.bias_b
                cb = Cb[..., sl_b, :]
            z_concat = torch.cat([za, zb], dim=-2)
            c_concat = torch.cat([ca, cb], dim=-2)
            S = torch.softmax(z_concat.float(), dim=-2).to(H.dtype)
            out[..., i, :] = (S * c_concat).sum(dim=-2)
        return out

    def test_shape_no_padding(self):
        H = torch.randn(3, 5 * self.m, self.d)
        out = csa_compress(
            H, self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z,
            self.bias_a, self.bias_b, self.m,
        )
        self.assertEqual(out.shape, (3, 5, self.c))

    def test_matches_naive_no_padding(self):
        H = torch.randn(2, 4 * self.m, self.d)
        out = csa_compress(
            H, self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z,
            self.bias_a, self.bias_b, self.m,
        )
        ref = self._ref(H)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    def test_first_block_padded_to_hca_form(self):
        """Block 0 has no previous block: result must match a single-block
        softmax over only the current m positions (HCA-style)."""
        H = torch.randn(self.m, self.d)
        out = csa_compress(
            H, self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z,
            self.bias_a, self.bias_b, self.m,
        )
        # Hand-compute: -inf prev half is dropped entirely.
        Ca = H @ self.W_a_kv
        Za = H @ self.W_a_z
        S = torch.softmax((Za + self.bias_a).float(), dim=-2).to(H.dtype)
        expected = (S * Ca).sum(dim=-2)
        torch.testing.assert_close(out[0], expected, rtol=1e-4, atol=1e-5)

    def test_padding_extends_one_block(self):
        n = 9   # m=4 -> 3 blocks (last has 1 real + 3 pad).
        H = torch.randn(n, self.d)
        out = csa_compress(
            H, self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z,
            self.bias_a, self.bias_b, self.m,
        )
        self.assertEqual(out.shape, (math.ceil(n / self.m), self.c))
        self.assertTrue(torch.isfinite(out).all())

    def test_batch_independence(self):
        H1 = torch.randn(self.m * 3, self.d)
        H2 = torch.randn(self.m * 3, self.d)
        H = torch.stack([H1, H2], dim=0)
        out_batched = csa_compress(
            H, self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z,
            self.bias_a, self.bias_b, self.m,
        )
        out1 = csa_compress(
            H1, self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z,
            self.bias_a, self.bias_b, self.m,
        )
        out2 = csa_compress(
            H2, self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z,
            self.bias_a, self.bias_b, self.m,
        )
        torch.testing.assert_close(out_batched[0], out1)
        torch.testing.assert_close(out_batched[1], out2)

    def test_softmax_normalisation_via_uniform(self):
        """Identity W_kv + zero W_z + zero bias on block 0 -> output equals
        mean of the m positions in block 0 (since prev block contributes 0)."""
        m, d = self.m, self.d
        c = d
        W_a_kv = torch.eye(d, c)
        W_b_kv = torch.zeros(d, c)
        W_a_z = torch.zeros(d, c)
        W_b_z = torch.zeros(d, c)
        bias_a = torch.zeros(m, c)
        bias_b = torch.zeros(m, c)
        H = torch.randn(m, d)
        out = csa_compress(H, W_a_kv, W_b_kv, W_a_z, W_b_z, bias_a, bias_b, m)
        torch.testing.assert_close(out[0], H.mean(dim=-2), rtol=1e-4, atol=1e-5)

    def test_dtype_preserved_bf16(self):
        H = torch.randn(self.m * 2, self.d, dtype=torch.bfloat16)
        ws = [w.to(torch.bfloat16) for w in (
            self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z
        )]
        bs = [b.to(torch.bfloat16) for b in (self.bias_a, self.bias_b)]
        out = csa_compress(H, *ws, *bs, self.m)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_rejects_bad_args(self):
        H = torch.randn(self.m, self.d)
        with self.assertRaises(ValueError):
            csa_compress(H, self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z,
                         self.bias_a, self.bias_b, m=0)
        with self.assertRaises(ValueError):
            # bias_a wrong length
            csa_compress(H, self.W_a_kv, self.W_b_kv, self.W_a_z, self.W_b_z,
                         torch.zeros(self.m + 1, self.c), self.bias_b, self.m)
        with self.assertRaises(ValueError):
            # W shape mismatch
            csa_compress(H, self.W_a_kv, torch.zeros(self.d + 1, self.c),
                         self.W_a_z, self.W_b_z,
                         self.bias_a, self.bias_b, self.m)


# ---------------------------------------------------------------------------
class HcaCompressorModuleTest(TestCase):
    def test_forward_matches_functional(self):
        _set_seed(2)
        d, c, m_p = 16, 8, 4
        layer = HcaCompressor(hidden_size=d, head_dim=c, m_prime=m_p, dtype=torch.float32)
        H = torch.randn(2, 3 * m_p, d)
        out = layer(H)
        ref = hca_compress(H, layer.W_kv, layer.W_z, layer.bias_pos, m_p)
        torch.testing.assert_close(out, ref)

    def test_state_dict_roundtrip(self):
        layer = HcaCompressor(hidden_size=8, head_dim=4, m_prime=4)
        sd = layer.state_dict()
        layer2 = HcaCompressor(hidden_size=8, head_dim=4, m_prime=4)
        layer2.load_state_dict(sd)
        H = torch.randn(8, 8)
        torch.testing.assert_close(layer(H), layer2(H))

    def test_gradient_flows(self):
        layer = HcaCompressor(hidden_size=4, head_dim=4, m_prime=2)
        H = torch.randn(4, 4, requires_grad=True)
        out = layer(H).sum()
        out.backward()
        self.assertIsNotNone(H.grad)
        self.assertIsNotNone(layer.W_kv.grad)
        self.assertIsNotNone(layer.W_z.grad)
        self.assertIsNotNone(layer.bias_pos.grad)


class CsaCompressorModuleTest(TestCase):
    def test_forward_matches_functional(self):
        _set_seed(3)
        d, c, m = 12, 6, 4
        layer = CsaCompressor(hidden_size=d, head_dim=c, m=m, dtype=torch.float32)
        H = torch.randn(2, 3 * m, d)
        out = layer(H)
        ref = csa_compress(
            H, layer.W_a_kv, layer.W_b_kv, layer.W_a_z, layer.W_b_z,
            layer.bias_a, layer.bias_b, m,
        )
        torch.testing.assert_close(out, ref)

    def test_gradient_flows(self):
        layer = CsaCompressor(hidden_size=4, head_dim=4, m=2)
        H = torch.randn(4, 4, requires_grad=True)
        layer(H).sum().backward()
        self.assertIsNotNone(H.grad)
        for p in layer.parameters():
            self.assertIsNotNone(p.grad)


if __name__ == "__main__":
    main()
