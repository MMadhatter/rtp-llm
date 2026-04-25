"""Parity + benchmark tests for the DeepGEMM FP8 lightning indexer.

Skipped when ``deep_gemm`` is unavailable. Compares
:func:`deepgemm_indexer_score_topk` against the bf16 reference
:func:`fused_indexer_score_topk`. Set ``RUN_DEEPSEEK_V4_BENCH=1`` to
also run the perf bench at production V4-Flash shape (H=64, D=128,
T_kc=4096).

Reference: in-tree V3.2 indexer
:mod:`rtp_llm.models_py.modules.base.cuda.indexer_op` already uses the
same DeepGEMM kernels; the parity bar there is a sanity check rather
than bit-exact.
"""

from __future__ import annotations

import os
import unittest
from unittest import TestCase

import torch


def _has_deepgemm():
    try:
        import deep_gemm  # noqa: F401

        return True
    except Exception:
        return False


HAS_DEEPGEMM = _has_deepgemm()
RUN_BENCH = os.environ.get("RUN_DEEPSEEK_V4_BENCH", "0") == "1"


@unittest.skipUnless(HAS_DEEPGEMM, "deep_gemm not available")
class DeepGemmIndexerParityTest(TestCase):
    def setUp(self):
        from rtp_llm.models_py.modules.hybrid.deepgemm_indexer import (
            deepgemm_indexer_score_topk,
        )
        from rtp_llm.models_py.modules.hybrid.fused_indexer import (
            fused_indexer_score_topk,
        )

        self.dg_fn = deepgemm_indexer_score_topk
        self.bf16_fn = fused_indexer_score_topk
        torch.manual_seed(0)

    def _setup(self, B=1, T_q=32, T_kc=128, H=32, D=128, K=8, q_lora=128):
        # Default shape satisfies all three DeepGEMM constraints:
        #   D=128 % 128, T_kc=128 % 128, M=B*T_q=32 % (128/H)=4 == 0.
        c_Q = (torch.randn(B, T_q, q_lora) * 0.1).cuda().bfloat16()
        K_IComp = (torch.randn(B, T_kc, D) * 0.1).cuda().bfloat16()
        W_IUQ = (torch.randn(q_lora, H * D) * 0.05).cuda().bfloat16()
        w_heads = torch.ones(H, device="cuda")
        return c_Q, K_IComp, W_IUQ, w_heads, H, D, K

    def test_shape_and_finite(self):
        c_Q, K_IComp, W_IUQ, w_heads, H, D, K = self._setup()
        idx, scores = self.dg_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        self.assertEqual(idx.shape, (1, 32, K))
        self.assertEqual(scores.shape, (1, 32, K))
        self.assertTrue(torch.isfinite(scores).all())

    def test_topk_set_overlap_with_bf16(self):
        c_Q, K_IComp, W_IUQ, w_heads, H, D, K = self._setup()
        ref_idx, _ = self.bf16_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        dg_idx, _ = self.dg_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        ref_sets = [set(r.tolist()) for r in ref_idx[0]]
        dg_sets = [set(r.tolist()) for r in dg_idx[0]]
        overlap = sum(len(r & f) for r, f in zip(ref_sets, dg_sets)) / sum(
            len(r) for r in ref_sets
        )
        # FP8 indexer recall is much tighter than FP4 (paper §2.3.5
        # Table 4 reports >0.95 on top-8). Allow some cushion for our
        # small T_kc.
        self.assertGreaterEqual(
            overlap, 0.7, f"FP8 vs bf16 top-k overlap = {overlap:.3f}"
        )

    def test_falls_back_when_kv_not_aligned(self):
        # T_kc=64 (< 128 alignment) → fall back to bf16. Result must match.
        c_Q, K_IComp, W_IUQ, w_heads, H, D, K = self._setup(T_kc=64)
        ref_idx, ref_scores = self.bf16_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        dg_idx, dg_scores = self.dg_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        self.assertTrue(torch.equal(dg_idx, ref_idx))
        torch.testing.assert_close(dg_scores, ref_scores, rtol=1e-5, atol=1e-5)

    def test_falls_back_when_M_not_aligned_to_block_q(self):
        # H=4 → block_q=32; M=B*T_q=8 isn't a multiple → falls back.
        c_Q, K_IComp, W_IUQ, w_heads, H, D, K = self._setup(T_q=8, H=4)
        ref_idx, ref_scores = self.bf16_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        dg_idx, dg_scores = self.dg_fn(c_Q, K_IComp, W_IUQ, w_heads, H, D, K)
        self.assertTrue(torch.equal(dg_idx, ref_idx))
        torch.testing.assert_close(dg_scores, ref_scores, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(HAS_DEEPGEMM and RUN_BENCH, "deep_gemm + RUN_DEEPSEEK_V4_BENCH=1")
class DeepGemmIndexerBenchmark(TestCase):
    """Set ``RUN_DEEPSEEK_V4_BENCH=1`` to run."""

    def setUp(self):
        from rtp_llm.models_py.modules.hybrid.deepgemm_indexer import (
            deepgemm_indexer_score_topk,
        )
        from rtp_llm.models_py.modules.hybrid.fused_indexer import (
            fused_indexer_score_topk,
        )

        self.dg_fn = deepgemm_indexer_score_topk
        self.bf16_fn = fused_indexer_score_topk
        torch.manual_seed(0)

    def _bench(self, fn, args, warmup=5, iters=20):
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def test_bench_v4_flash_indexer_shape(self):
        # V4-Flash indexer: H=64, D=128, K=2048.
        # Use T_q=128, T_kc=4096 for a tractable bench.
        B, T_q, T_kc, H, D, K = 1, 128, 4096, 64, 128, 2048
        q_lora = 1024
        c_Q = (torch.randn(B, T_q, q_lora) * 0.1).cuda().bfloat16()
        K_IComp = (torch.randn(B, T_kc, D) * 0.1).cuda().bfloat16()
        W_IUQ = (torch.randn(q_lora, H * D) * 0.02).cuda().bfloat16()
        w_heads = torch.ones(H, device="cuda")

        ms_ref = self._bench(self.bf16_fn, (c_Q, K_IComp, W_IUQ, w_heads, H, D, K))
        ms_dg = self._bench(self.dg_fn, (c_Q, K_IComp, W_IUQ, w_heads, H, D, K))

        # Logit FLOPs per call: per (m, n) we do H · D MAC pairs.
        flops = 2 * B * T_q * T_kc * H * D
        tflops_ref = flops / (ms_ref * 1e-3) / 1e12
        tflops_dg = flops / (ms_dg * 1e-3) / 1e12
        speedup = ms_ref / ms_dg
        print(
            f"\n[bench V4-Indexer] B={B} T_q={T_q} T_kc={T_kc} H={H} D={D}"
            f"\n  reference (bf16 fused):    {ms_ref:7.3f} ms  ({tflops_ref:6.1f} TFLOPS)"
            f"\n  deepgemm  (fp8 mqa_logits):{ms_dg:7.3f} ms  ({tflops_dg:6.1f} TFLOPS)"
            f"\n  speedup:                   {speedup:6.2f}x"
        )
        self.assertGreater(
            speedup,
            1.0,
            f"deepgemm slower than ref ({speedup:.2f}x) — investigate",
        )


if __name__ == "__main__":
    unittest.main()
