"""Parity tests for the online-softmax MQA + sink attention path.

Compares :func:`online_sink_mqa` against the materialised-concat reference
:func:`mqa_attention_with_sink`. The fused variant folds the sink into
the softmax denominator without ever concatenating it onto the logits
tensor; the two paths must produce identical outputs up to fp32 rounding.

Reference: vLLM PR #40760 ``vllm/v1/attention/backends/mla/sparse_swa.py``
(``_apply_sink_softmax``) and SGLang PR #23600
``python/sglang/srt/layers/attention/compressed/paged_prefill.py``.
"""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.hybrid.hca_attention import mqa_attention_with_sink
from rtp_llm.models_py.modules.hybrid.online_sink_attention import online_sink_mqa


class OnlineSinkMqaParityTest(TestCase):
    def _setup(
        self,
        B=2,
        T_q=5,
        T_kc=8,
        n_win=4,
        H=4,
        D=16,
        seed=0,
        dtype=torch.float32,
    ):
        torch.manual_seed(seed)
        Q = torch.randn(B, T_q, H, D, dtype=dtype)
        K_c = torch.randn(B, T_kc, D, dtype=dtype)
        V_c = torch.randn(B, T_kc, D, dtype=dtype)
        K_w = torch.randn(B, T_q, n_win, D, dtype=dtype)
        V_w = torch.randn(B, T_q, n_win, D, dtype=dtype)
        sink = torch.randn(H, dtype=dtype)
        return Q, K_c, V_c, K_w, V_w, sink

    def test_matches_reference_compressed_only(self):
        Q, K_c, V_c, _, _, sink = self._setup()
        ref = mqa_attention_with_sink(Q, K_c, V_c, None, None, sink)
        out = online_sink_mqa(Q, K_c, V_c, None, None, sink)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_matches_reference_with_window(self):
        Q, K_c, V_c, K_w, V_w, sink = self._setup()
        ref = mqa_attention_with_sink(Q, K_c, V_c, K_w, V_w, sink)
        out = online_sink_mqa(Q, K_c, V_c, K_w, V_w, sink)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_matches_reference_with_causal_mask(self):
        Q, K_c, V_c, _, _, sink = self._setup(T_q=5, T_kc=5)
        # Upper-triangular mask = future positions blocked.
        causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        causal = causal.unsqueeze(0).expand(Q.shape[0], -1, -1).contiguous()
        ref = mqa_attention_with_sink(
            Q, K_c, V_c, None, None, sink, causal_compressed_mask=causal
        )
        out = online_sink_mqa(
            Q, K_c, V_c, None, None, sink, causal_compressed_mask=causal
        )
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_matches_reference_with_swa_valid_mask(self):
        Q, K_c, V_c, K_w, V_w, sink = self._setup(T_q=5, n_win=4)
        # Mask the last position of every window.
        swa_mask = torch.zeros(Q.shape[0], 5, 4, dtype=torch.bool)
        swa_mask[..., -1] = True
        ref = mqa_attention_with_sink(
            Q, K_c, V_c, K_w, V_w, sink, swa_valid_mask=swa_mask
        )
        out = online_sink_mqa(Q, K_c, V_c, K_w, V_w, sink, swa_valid_mask=swa_mask)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_huge_sink_dominates_softmax(self):
        """sink_logit → +∞ should drive ALL key weights to ~0."""
        Q, K_c, V_c, _, _, _ = self._setup()
        sink = torch.full((Q.shape[2],), 1e6)
        out = online_sink_mqa(Q, K_c, V_c, None, None, sink)
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-5, atol=1e-3)

    def test_bf16_within_tolerance(self):
        Q, K_c, V_c, K_w, V_w, sink = self._setup(dtype=torch.bfloat16)
        ref = mqa_attention_with_sink(Q, K_c, V_c, K_w, V_w, sink)
        out = online_sink_mqa(Q, K_c, V_c, K_w, V_w, sink)
        # bf16 mantissa drops ~3 decimal digits; both paths use fp32 softmax.
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    main()
