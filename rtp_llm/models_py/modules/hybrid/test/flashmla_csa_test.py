"""Parity + benchmark tests for the FlashMLA-backed CSA wrapper.

CPU-runnable parity tests cover the Python reference path
(:func:`_reference_sparse_csa`) — invariants like sink behaviour,
``-1`` invalid index handling, and ``topk_length`` truncation.

GPU-only tests (skipped without ``flash_mla`` + Blackwell sm_100 + the
right h_q / topk shapes) cross-check the FlashMLA kernel against the
reference at <1% relative L2.

Set ``RUN_DEEPSEEK_V4_BENCH=1`` to run the V4-Flash micro-benchmark.

Reference: SGLang PR #23600 ``test_flashmla_csa.py`` and FlashMLA's own
``tests/test_sparse_prefill.py``.
"""

from __future__ import annotations

import os
import unittest
from unittest import TestCase

import torch


def _has_flashmla():
    try:
        import flash_mla  # noqa: F401

        return True
    except Exception:
        return False


def _is_blackwell():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


HAS_FLASHMLA = _has_flashmla()
IS_BLACKWELL = _is_blackwell()
RUN_BENCH = os.environ.get("RUN_DEEPSEEK_V4_BENCH", "0") == "1"


class ReferenceSparseCsaTest(TestCase):
    """CPU-runnable invariants of the Python reference path."""

    def setUp(self):
        from rtp_llm.models_py.modules.hybrid.flashmla_csa import (
            _V4_D_QK,
            _V4_D_V,
            _reference_sparse_csa,
        )

        self.D_QK = _V4_D_QK
        self.D_V = _V4_D_V
        self.ref = _reference_sparse_csa
        torch.manual_seed(0)

    def _make(self, s_q=4, h_q=8, s_kv=16, topk=4):
        q = torch.randn(s_q, h_q, self.D_QK)
        kv = torch.randn(s_kv, 1, self.D_QK)
        idx = (
            torch.stack([torch.randperm(s_kv)[:topk].int() for _ in range(s_q)])
            .unsqueeze(1)
            .contiguous()
        )
        sink = torch.zeros(h_q, dtype=torch.float32)
        return q, kv, idx, sink

    def test_shape_and_finite(self):
        q, kv, idx, sink = self._make()
        out = self.ref(q, kv, idx, 1.0 / (self.D_QK**0.5), sink, None)
        self.assertEqual(out.shape, (q.shape[0], q.shape[1], self.D_V))
        self.assertTrue(torch.isfinite(out).all())

    def test_minus_one_index_is_masked(self):
        q, kv, idx, sink = self._make()
        # Make every query miss its first slot.
        idx[:, 0, 0] = -1
        out = self.ref(q, kv, idx, 1.0 / (self.D_QK**0.5), sink, None)
        self.assertTrue(torch.isfinite(out).all())

    def test_inf_sink_zeroes_output(self):
        """sink = +∞ → exp(sink) dominates Z, output → 0."""
        q, kv, idx, _ = self._make()
        h_q = q.shape[1]
        sink = torch.full((h_q,), 1e6, dtype=torch.float32)
        out = self.ref(q, kv, idx, 1.0 / (self.D_QK**0.5), sink, None)
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-3, atol=1e-3)

    def test_topk_length_truncation(self):
        q, kv, idx, sink = self._make(s_q=4, topk=8)
        # First query attends to only 1 slot; should still be finite.
        tlens = torch.tensor([1, 4, 8, 8], dtype=torch.int32)
        out = self.ref(q, kv, idx, 1.0 / (self.D_QK**0.5), sink, tlens)
        self.assertTrue(torch.isfinite(out).all())

    def test_no_sink_collapses_to_pure_softmax(self):
        """attn_sink=None should give the same answer as attn_sink=-inf."""
        q, kv, idx, _ = self._make()
        h_q = q.shape[1]
        out_no_sink = self.ref(q, kv, idx, 1.0 / (self.D_QK**0.5), None, None)
        sink_neginf = torch.full((h_q,), float("-inf"), dtype=torch.float32)
        out_neginf = self.ref(q, kv, idx, 1.0 / (self.D_QK**0.5), sink_neginf, None)
        torch.testing.assert_close(out_no_sink, out_neginf, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(
    HAS_FLASHMLA and IS_BLACKWELL,
    "flash_mla + Blackwell (sm_100) required for FlashMLA sparse prefill",
)
class FlashMlaCsaSm100Test(TestCase):
    """Parity vs the bf16 reference on the actual GPU kernel.

    Constraints (from FlashMLA 1.0.0 sm_100 sparse-prefill kernel):
      * h_q ∈ {64, 128} only — V4-Flash uses 64, V4-Pro uses 128
      * topk must be a multiple of 64 (B_TOPK tile size)
    """

    def setUp(self):
        from rtp_llm.models_py.modules.hybrid.flashmla_csa import (
            _V4_D_QK,
            _V4_D_V,
            _reference_sparse_csa,
            flashmla_csa_or_reference,
            flashmla_csa_sparse_fwd,
        )

        self.D_QK = _V4_D_QK
        self.D_V = _V4_D_V
        self.ref = _reference_sparse_csa
        self.fwd = flashmla_csa_sparse_fwd
        self.selector = flashmla_csa_or_reference
        torch.manual_seed(0)

    def _make_gpu(self, s_q=8, h_q=64, s_kv=512, topk=128):
        q = torch.randn(s_q, h_q, self.D_QK).cuda().bfloat16()
        kv = torch.randn(s_kv, 1, self.D_QK).cuda().bfloat16()
        idx = (
            torch.stack(
                [torch.randperm(s_kv, device="cuda")[:topk].int() for _ in range(s_q)]
            )
            .unsqueeze(1)
            .contiguous()
        )
        sink = torch.zeros(h_q, dtype=torch.float32, device="cuda")
        return q, kv, idx, sink

    def test_kernel_matches_reference_h64_topk128(self):
        q, kv, idx, sink = self._make_gpu(h_q=64, topk=128)
        sm_scale = 1.0 / (self.D_QK**0.5)
        out = self.fwd(q, kv, idx, sm_scale, attn_sink=sink)
        ref = self.ref(q, kv, idx, sm_scale, sink, None)
        rel = (out.float() - ref.float()).norm() / ref.float().norm()
        self.assertLess(
            float(rel),
            1e-2,
            f"FlashMLA vs ref relative L2 = {rel:.3e}",
        )

    def test_kernel_matches_reference_h64_topk1024(self):
        q, kv, idx, sink = self._make_gpu(h_q=64, s_kv=2048, topk=1024)
        sm_scale = 1.0 / (self.D_QK**0.5)
        out = self.fwd(q, kv, idx, sm_scale, attn_sink=sink)
        ref = self.ref(q, kv, idx, sm_scale, sink, None)
        rel = (out.float() - ref.float()).norm() / ref.float().norm()
        self.assertLess(float(rel), 1e-2)

    def test_kernel_handles_sink_inf(self):
        """sink = +inf → output should be ~0 (matches reference)."""
        q, kv, idx, _ = self._make_gpu()
        sink = torch.full((q.shape[1],), 1e6, dtype=torch.float32, device="cuda")
        out = self.fwd(q, kv, idx, 1.0 / (self.D_QK**0.5), attn_sink=sink)
        torch.testing.assert_close(
            out.float(),
            torch.zeros_like(out.float()),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_kernel_handles_topk_length(self):
        s_q, h_q, s_kv, topk = 8, 64, 512, 128
        q, kv, idx, sink = self._make_gpu(s_q=s_q, h_q=h_q, s_kv=s_kv, topk=topk)
        # All queries truncated to 64 effective KV (still % 64 == 0).
        tlens = torch.full((s_q,), 64, dtype=torch.int32, device="cuda")
        out = self.fwd(
            q,
            kv,
            idx,
            1.0 / (self.D_QK**0.5),
            attn_sink=sink,
            topk_length=tlens,
        )
        self.assertTrue(torch.isfinite(out).all())

    def test_selector_falls_back_for_h_q_8(self):
        # h_q=8 isn't supported by the kernel → selector must route to
        # reference. Output should match reference exactly.
        q, kv, idx, sink = self._make_gpu(h_q=8, topk=64)
        ref = self.ref(q, kv, idx, 1.0 / (self.D_QK**0.5), sink, None)
        out = self.selector(q, kv, idx, 1.0 / (self.D_QK**0.5), attn_sink=sink)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_selector_falls_back_for_topk_not_aligned(self):
        # topk=96 isn't a multiple of 64 → selector falls back.
        q, kv, idx, sink = self._make_gpu(h_q=64, topk=96)
        ref = self.ref(q, kv, idx, 1.0 / (self.D_QK**0.5), sink, None)
        out = self.selector(q, kv, idx, 1.0 / (self.D_QK**0.5), attn_sink=sink)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(
    HAS_FLASHMLA and IS_BLACKWELL and RUN_BENCH,
    "flash_mla + sm_100 + RUN_DEEPSEEK_V4_BENCH=1",
)
class FlashMlaCsaBenchmark(TestCase):
    """Set ``RUN_DEEPSEEK_V4_BENCH=1`` to run."""

    def setUp(self):
        from rtp_llm.models_py.modules.hybrid.flashmla_csa import (
            _V4_D_QK,
            _reference_sparse_csa,
            flashmla_csa_sparse_fwd,
        )

        self.D_QK = _V4_D_QK
        self.ref = _reference_sparse_csa
        self.fwd = flashmla_csa_sparse_fwd
        torch.manual_seed(0)

    def _bench(self, fn, args, warmup=5, iters=20):
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn(*args)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    def test_bench_v4_flash_csa_shape(self):
        # V4-Flash CSA prefill: h_q=64, topk=2048 (per indexer config),
        # moderate s_q to fit in mem.
        s_q, h_q, s_kv, topk = 256, 64, 8192, 2048
        q = torch.randn(s_q, h_q, self.D_QK).cuda().bfloat16()
        kv = torch.randn(s_kv, 1, self.D_QK).cuda().bfloat16()
        idx = (
            torch.stack(
                [torch.randperm(s_kv, device="cuda")[:topk].int() for _ in range(s_q)]
            )
            .unsqueeze(1)
            .contiguous()
        )
        sink = torch.zeros(h_q, dtype=torch.float32, device="cuda")
        sm_scale = 1.0 / (self.D_QK**0.5)

        ms_ref = self._bench(self.ref, (q, kv, idx, sm_scale, sink, None))
        ms_fmla = self._bench(self.fwd, (q, kv, idx, sm_scale))  # no sink kw arg

        # Approx FLOPs for the per-query softmax + value blend:
        #   logits   = 2 * s_q * h_q * topk * D_QK
        #   blend    = 2 * s_q * h_q * topk * D_V
        flops = 2 * s_q * h_q * topk * (self.D_QK + 512)
        tflops_ref = flops / (ms_ref * 1e-3) / 1e12
        tflops_fmla = flops / (ms_fmla * 1e-3) / 1e12
        speedup = ms_ref / ms_fmla
        print(
            f"\n[bench V4-CSA] s_q={s_q} h_q={h_q} s_kv={s_kv} topk={topk}"
            f"\n  reference (bf16 inline):   {ms_ref:7.3f} ms  ({tflops_ref:6.1f} TFLOPS)"
            f"\n  flashmla  (sparse_fwd):    {ms_fmla:7.3f} ms  ({tflops_fmla:6.1f} TFLOPS)"
            f"\n  speedup:                   {speedup:6.2f}x"
        )
        self.assertGreater(
            speedup,
            1.0,
            f"FlashMLA slower than ref ({speedup:.2f}x) — investigate",
        )


if __name__ == "__main__":
    unittest.main()
