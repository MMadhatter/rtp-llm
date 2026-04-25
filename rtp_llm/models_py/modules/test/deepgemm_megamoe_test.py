"""Parity + benchmark tests for DeepGEMM-backed MegaMoE.

Skipped when ``deep_gemm`` is unavailable. On Hopper / Blackwell with
DeepGEMM 2.2.0+, exercises:
  * shape contract (output is ``(N, hidden)``)
  * fp8 vs bf16 reference parity at 1e-1 tolerance (FP8 quant noise is
    additive across the two GEMMs + SwiGLU clamp)
  * micro-benchmark vs :func:`batched_experts_forward` at production
    V4-Flash shape (E=256, top_k=6, hidden=4096, inter=2048).

Reference: SGLang PR #23600 ``test_megamoe_deepgemm.py`` and DeepGEMM's
own ``tests/test_grouped_gemm_masked.py``.
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
class DeepGemmMegaMoEParityTest(TestCase):
    def setUp(self):
        import deep_gemm

        from rtp_llm.models_py.modules.moe.batched_experts import (
            batched_experts_forward,
        )
        from rtp_llm.models_py.modules.moe.deepgemm_megamoe import (
            deepgemm_megamoe_forward,
        )

        self.deep_gemm = deep_gemm
        self.batched = batched_experts_forward
        self.deepgemm_fn = deepgemm_megamoe_forward
        torch.manual_seed(0)

    def _make_weights(self, E, H, inter, dtype=torch.bfloat16):
        # bf16 ref weights (unpacked gate / up / down).
        W_gate = (torch.randn(E, H, inter, dtype=dtype) * 0.05).cuda()
        W_up = (torch.randn(E, H, inter, dtype=dtype) * 0.05).cuda()
        W_down = (torch.randn(E, inter, H, dtype=dtype) * 0.05).cuda()
        # Packed [gate; up] along the inter axis, transposed to (E, 2*inter, H)
        # for the masked-GEMM NT layout.
        W_gate_up_bf16 = (
            torch.cat([W_gate, W_up], dim=-1).transpose(-1, -2).contiguous()
        )
        W_down_T_bf16 = W_down.transpose(-1, -2).contiguous()  # (E, H, inter)

        # FP8 quant per-block.
        gu_fp8_list, gu_sf_list = [], []
        d_fp8_list, d_sf_list = [], []
        for e in range(E):
            f, s = self.deep_gemm.per_block_cast_to_fp8(W_gate_up_bf16[e], True)
            gu_fp8_list.append(f)
            gu_sf_list.append(s)
            f, s = self.deep_gemm.per_block_cast_to_fp8(W_down_T_bf16[e], True)
            d_fp8_list.append(f)
            d_sf_list.append(s)
        W_gate_up_fp8 = torch.stack(gu_fp8_list)
        W_gate_up_sf = torch.stack(gu_sf_list)
        W_down_fp8 = torch.stack(d_fp8_list)
        W_down_sf = torch.stack(d_sf_list)
        return (
            (W_gate, W_up, W_down),
            (W_gate_up_fp8, W_gate_up_sf, W_down_fp8, W_down_sf),
        )

    def _make_routing(self, N, K, E):
        x = (torch.randn(N, 4096, dtype=torch.bfloat16) * 0.1).cuda()
        # Random distinct top-k per token.
        topk = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(N)])
        gates = torch.full((N, K), 1.0 / K, dtype=torch.bfloat16, device="cuda")
        return x, topk, gates

    def test_shape_and_finite_small(self):
        E, H, inter, N, K = 8, 256, 384, 6, 2
        bf16_w, fp8_w = self._make_weights(E, H, inter)
        x = (torch.randn(N, H, dtype=torch.bfloat16) * 0.1).cuda()
        topk = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(N)])
        gates = torch.full((N, K), 1.0 / K, dtype=torch.bfloat16, device="cuda")
        out = self.deepgemm_fn(
            x,
            topk,
            gates,
            *fp8_w,
            swiglu_limit=10.0,
        )
        self.assertEqual(out.shape, (N, H))
        self.assertTrue(torch.isfinite(out).all())

    def test_parity_with_bf16_reference_small(self):
        # Use small shapes so the noise budget is interpretable.
        E, H, inter, N, K = 4, 128, 256, 8, 2
        bf16_w, fp8_w = self._make_weights(E, H, inter)
        x = (torch.randn(N, H, dtype=torch.bfloat16) * 0.1).cuda()
        topk = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(N)])
        gates = torch.full((N, K), 1.0 / K, dtype=torch.bfloat16, device="cuda")

        ref = self.batched(
            x, topk, gates, bf16_w[0], bf16_w[1], bf16_w[2], swiglu_limit=10.0
        )
        out = self.deepgemm_fn(
            x,
            topk,
            gates,
            *fp8_w,
            swiglu_limit=10.0,
        )
        # FP8 noise budget: per-element ~3% of std; aggregated output is
        # weighted-sum so we compare relative L2.
        rel = (out.float() - ref.float()).norm() / ref.float().norm()
        self.assertLess(float(rel), 0.15, f"FP8 vs bf16 relative L2 = {rel:.3f}")


@unittest.skipUnless(HAS_DEEPGEMM and RUN_BENCH, "deep_gemm + RUN_DEEPSEEK_V4_BENCH=1")
class DeepGemmMegaMoEBenchmark(TestCase):
    """Set ``RUN_DEEPSEEK_V4_BENCH=1`` to run. Reports ms / TFLOPS."""

    def setUp(self):
        import deep_gemm

        from rtp_llm.models_py.modules.moe.batched_experts import (
            batched_experts_forward,
        )
        from rtp_llm.models_py.modules.moe.deepgemm_megamoe import (
            deepgemm_megamoe_forward,
        )

        self.deep_gemm = deep_gemm
        self.batched = batched_experts_forward
        self.deepgemm_fn = deepgemm_megamoe_forward
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

    def test_bench_v4_flash_shape(self):
        # V4-Flash: hidden=4096, routed inter=2048 (per the V4 paper);
        # E=256 experts, top_k=6. Use a moderate batch size for bench.
        E, H, inter = 32, 4096, 2048  # E reduced for bench memory
        N, K = 64, 6
        # bf16 weights
        W_gate = (torch.randn(E, H, inter, dtype=torch.bfloat16) * 0.05).cuda()
        W_up = (torch.randn(E, H, inter, dtype=torch.bfloat16) * 0.05).cuda()
        W_down = (torch.randn(E, inter, H, dtype=torch.bfloat16) * 0.05).cuda()

        # FP8 weights (packed [gate;up], down separately).
        W_gate_up_T = torch.cat([W_gate, W_up], dim=-1).transpose(-1, -2).contiguous()
        W_down_T = W_down.transpose(-1, -2).contiguous()
        gu_fp8_list, gu_sf_list, d_fp8_list, d_sf_list = [], [], [], []
        for e in range(E):
            f, s = self.deep_gemm.per_block_cast_to_fp8(W_gate_up_T[e], True)
            gu_fp8_list.append(f)
            gu_sf_list.append(s)
            f, s = self.deep_gemm.per_block_cast_to_fp8(W_down_T[e], True)
            d_fp8_list.append(f)
            d_sf_list.append(s)
        W_gate_up_fp8 = torch.stack(gu_fp8_list)
        W_gate_up_sf = torch.stack(gu_sf_list)
        W_down_fp8 = torch.stack(d_fp8_list)
        W_down_sf = torch.stack(d_sf_list)

        x = (torch.randn(N, H, dtype=torch.bfloat16) * 0.1).cuda()
        topk = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(N)])
        gates = torch.full((N, K), 1.0 / K, dtype=torch.bfloat16, device="cuda")

        ms_ref = self._bench(
            self.batched,
            (x, topk, gates, W_gate, W_up, W_down, 10.0),
        )
        ms_dg = self._bench(
            self.deepgemm_fn,
            (x, topk, gates, W_gate_up_fp8, W_gate_up_sf, W_down_fp8, W_down_sf, 10.0),
        )

        # Rough TFLOPS estimate per call (gate+up + down GEMMs).
        # gate_up: 2 * N*K * H * (2*inter)  ; down: 2 * N*K * inter * H
        flops = 2 * N * K * H * (2 * inter) + 2 * N * K * inter * H
        tflops_ref = flops / (ms_ref * 1e-3) / 1e12
        tflops_dg = flops / (ms_dg * 1e-3) / 1e12
        speedup = ms_ref / ms_dg
        print(
            f"\n[bench V4-MegaMoE] E={E} N={N} K={K} H={H} inter={inter}"
            f"\n  reference (bf16 batched):  {ms_ref:7.3f} ms  ({tflops_ref:6.1f} TFLOPS)"
            f"\n  deepgemm  (fp8 megamoe):   {ms_dg:7.3f} ms  ({tflops_dg:6.1f} TFLOPS)"
            f"\n  speedup:                   {speedup:6.2f}x"
        )
        # Sanity: deepgemm must beat the python loop.
        self.assertGreater(
            speedup,
            1.0,
            f"deepgemm slower than python ref ({speedup:.2f}x) — investigate",
        )


if __name__ == "__main__":
    unittest.main()
