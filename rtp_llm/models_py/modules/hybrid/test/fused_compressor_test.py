"""Parity tests for the joint-matmul fused compressor helpers.

Compares :func:`fused_hca_compress` and :func:`fused_csa_compress`
against the reference :func:`hca_compress` / :func:`csa_compress`. The
fused variants concatenate the projection weights along the channel
axis to do a single matmul instead of two (HCA) / four (CSA) — math
must be identical bit-for-bit modulo fp32 reduction order.

Reference: vLLM PR #40760
``vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py``
and SGLang PR #23600 ``compressor.py``.
"""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.hybrid.compressor import csa_compress, hca_compress
from rtp_llm.models_py.modules.hybrid.fused_compressor import (
    fused_csa_compress,
    fused_hca_compress,
)


class FusedHcaCompressTest(TestCase):
    def _make_weights(self, d=16, c=8, m_prime=4, seed=0):
        torch.manual_seed(seed)
        W_kv = torch.randn(d, c) * 0.1
        W_z = torch.randn(d, c) * 0.1
        bias_pos = torch.randn(m_prime, c) * 0.1
        return W_kv, W_z, bias_pos

    def test_matches_reference_no_padding(self):
        d, c, m_prime = 16, 8, 4
        W_kv, W_z, bias_pos = self._make_weights(d, c, m_prime)
        H = torch.randn(2, 12, d)  # n=12 is a multiple of 4
        ref = hca_compress(H, W_kv, W_z, bias_pos, m_prime)
        out = fused_hca_compress(H, W_kv, W_z, bias_pos, m_prime)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_matches_reference_with_padding(self):
        d, c, m_prime = 16, 8, 4
        W_kv, W_z, bias_pos = self._make_weights(d, c, m_prime)
        H = torch.randn(2, 13, d)  # n=13 → pad 3 → 4 blocks
        ref = hca_compress(H, W_kv, W_z, bias_pos, m_prime)
        out = fused_hca_compress(H, W_kv, W_z, bias_pos, m_prime)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_with_leading_batch_dims(self):
        d, c, m_prime = 16, 8, 2
        W_kv, W_z, bias_pos = self._make_weights(d, c, m_prime)
        H = torch.randn(2, 3, 6, d)
        ref = hca_compress(H, W_kv, W_z, bias_pos, m_prime)
        out = fused_hca_compress(H, W_kv, W_z, bias_pos, m_prime)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_m_prime_one_passthrough(self):
        d, c = 16, 8
        W_kv, W_z, bias_pos = self._make_weights(d, c, m_prime=1)
        H = torch.randn(1, 5, d)
        ref = hca_compress(H, W_kv, W_z, bias_pos, m_prime=1)
        out = fused_hca_compress(H, W_kv, W_z, bias_pos, m_prime=1)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_m_prime_zero_raises(self):
        W_kv, W_z, bias_pos = self._make_weights(m_prime=4)
        H = torch.randn(1, 4, 16)
        with self.assertRaises(ValueError):
            fused_hca_compress(H, W_kv, W_z, bias_pos, m_prime=0)


class FusedCsaCompressTest(TestCase):
    def _make_weights(self, d=16, c=8, m=4, seed=1):
        torch.manual_seed(seed)
        Wakv = torch.randn(d, c) * 0.1
        Wbkv = torch.randn(d, c) * 0.1
        Waz = torch.randn(d, c) * 0.1
        Wbz = torch.randn(d, c) * 0.1
        bias_a = torch.randn(m, c) * 0.1
        bias_b = torch.randn(m, c) * 0.1
        return Wakv, Wbkv, Waz, Wbz, bias_a, bias_b

    def test_matches_reference_no_padding(self):
        d, c, m = 16, 8, 4
        Wakv, Wbkv, Waz, Wbz, ba, bb = self._make_weights(d, c, m)
        H = torch.randn(2, 16, d)
        ref = csa_compress(H, Wakv, Wbkv, Waz, Wbz, ba, bb, m)
        out = fused_csa_compress(H, Wakv, Wbkv, Waz, Wbz, ba, bb, m)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_matches_reference_with_padding(self):
        d, c, m = 16, 8, 4
        Wakv, Wbkv, Waz, Wbz, ba, bb = self._make_weights(d, c, m)
        H = torch.randn(2, 14, d)  # pad 2 → 4 blocks
        ref = csa_compress(H, Wakv, Wbkv, Waz, Wbz, ba, bb, m)
        out = fused_csa_compress(H, Wakv, Wbkv, Waz, Wbz, ba, bb, m)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_first_block_collapses_to_hca_style(self):
        """For block 0 the prev-block contributions are -inf-masked; the
        fused output of block 0 should equal a stand-alone HCA softmax
        over just the current block."""
        d, c, m = 8, 4, 4
        Wakv, Wbkv, Waz, Wbz, ba, bb = self._make_weights(d, c, m)
        H = torch.randn(1, 4, d)  # exactly one block
        ref_a = hca_compress(H, Wakv, Waz, ba, m)
        out = fused_csa_compress(H, Wakv, Wbkv, Waz, Wbz, ba, bb, m)
        torch.testing.assert_close(out, ref_a, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    main()
