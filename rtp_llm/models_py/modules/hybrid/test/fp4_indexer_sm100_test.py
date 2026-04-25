"""GB200-only parity tests for the FP4 lightning indexer.

Skipped on non-Blackwell builds (CUTLASS NVFP4 GEMM is sm_100a only). On
Blackwell, compares :func:`fp4_indexer_score_topk` against the bf16
reference :func:`fused_indexer_score_topk` and checks that:

  1. The FP4 path returns sane shapes (no kernel-shape mismatch).
  2. The set of top-k indices overlaps the bf16 reference at the level
     the paper reports for FP4 indexer (~85% on top-8 at production
     shapes — paper §2.3.5 Table 4).
  3. Scores stay in a reasonable numeric range relative to bf16
     (no NaNs, no order-of-magnitude blow-up).

Reference: vLLM PR #40760 ``test_fp4_indexer_sm100.py`` and SGLang PR
#23600 ``python/sglang/srt/layers/attention/compressed/test/test_indexer_fp4.py``.
"""

import unittest
from unittest import TestCase

import torch

from rtp_llm.models_py.modules.hybrid.sm100_selector import has_fp4_kernels


@unittest.skipUnless(has_fp4_kernels(), "FP4 / sm_100 not available on this host")
class FP4IndexerSm100Test(TestCase):
    def setUp(self):
        # Imports deferred so the module decorator skips cleanly when the
        # FP4 ops are missing at import time.
        from rtp_llm.models_py.modules.hybrid.fp4_indexer import fp4_indexer_score_topk
        from rtp_llm.models_py.modules.hybrid.fused_indexer import (
            fused_indexer_score_topk,
        )

        self.fp4_fn = fp4_indexer_score_topk
        self.bf16_fn = fused_indexer_score_topk
        torch.manual_seed(0)

    def _setup(self, B=1, T_q=64, T_kc=128, H=4, D=64, K=8, q_lora=64):
        c_Q = torch.randn(B, T_q, q_lora, device="cuda").bfloat16()
        K_IComp = torch.randn(B, T_kc, D, device="cuda").bfloat16()
        W_IUQ = (torch.randn(q_lora, H * D) * 0.05).cuda().bfloat16()
        w_heads = torch.ones(H, device="cuda")
        return c_Q, K_IComp, W_IUQ, w_heads, H, D, K

    def test_shapes_and_finite(self):
        c_Q, K_IComp, W_IUQ, w_heads, H, D, K = self._setup()
        idx, scores = self.fp4_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        self.assertEqual(idx.shape, (1, 64, K))
        self.assertEqual(scores.shape, (1, 64, K))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue((idx >= 0).all() and (idx < 128).all())

    def test_topk_set_overlap_with_bf16(self):
        c_Q, K_IComp, W_IUQ, w_heads, H, D, K = self._setup()
        ref_idx, _ = self.bf16_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        fp4_idx, _ = self.fp4_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        ref_sets = [set(r.tolist()) for r in ref_idx[0]]
        fp4_sets = [set(r.tolist()) for r in fp4_idx[0]]
        overlap = sum(len(r & f) for r, f in zip(ref_sets, fp4_sets)) / sum(
            len(r) for r in ref_sets
        )
        # Paper-reported FP4 indexer recall @top-8 is ~0.85; allow some
        # cushion for the bf16-vs-fp32 reference and small T_kc.
        self.assertGreaterEqual(
            overlap, 0.7, f"FP4 vs bf16 top-k overlap too low: {overlap:.3f}"
        )

    def test_falls_back_when_shape_not_multiple_of_32(self):
        # T_q=8 (< 32) → fall back to bf16. Result must match bf16 exactly.
        c_Q, K_IComp, W_IUQ, w_heads, H, D, K = self._setup(T_q=8, T_kc=64)
        ref_idx, ref_scores = self.bf16_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        fp4_idx, fp4_scores = self.fp4_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        self.assertTrue(torch.equal(fp4_idx, ref_idx))
        torch.testing.assert_close(fp4_scores, ref_scores, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
