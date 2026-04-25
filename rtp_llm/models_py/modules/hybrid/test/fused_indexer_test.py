"""Parity tests for the fused lightning-indexer Q forward.

Compares :func:`fused_indexer_score_topk` against the four-pass
reference baked into :class:`CsaLightningIndexer.forward`.

Reference: vLLM PR #40760
``vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py`` and SGLang
PR #23600 ``python/sglang/srt/layers/attention/compressed/indexer.py``.
"""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.hybrid.csa_attention import CsaLightningIndexer
from rtp_llm.models_py.modules.hybrid.fused_indexer import fused_indexer_score_topk


class FusedIndexerParityTest(TestCase):
    def _setup(
        self,
        B=2,
        T_q=7,
        T_kc=11,
        q_lora_rank=24,
        H=4,
        D=8,
        top_k=4,
        seed=0,
        dtype=torch.float32,
    ):
        torch.manual_seed(seed)
        c_Q = torch.randn(B, T_q, q_lora_rank, dtype=dtype)
        K_IComp = torch.randn(B, T_kc, D, dtype=dtype)
        idx = CsaLightningIndexer(
            q_lora_rank=q_lora_rank,
            num_indexer_heads=H,
            indexer_head_dim=D,
            top_k=top_k,
            dtype=dtype,
        )
        # Randomise heads so the reduction across heads isn't trivial.
        with torch.no_grad():
            torch.nn.init.normal_(idx.W_IUQ, std=0.02)
            torch.nn.init.normal_(idx.w_heads, std=0.5)
        return idx, c_Q, K_IComp

    def test_topk_indices_match_reference(self):
        idx, c_Q, K_IComp = self._setup()
        ref_idx, ref_scores = idx(c_Q, K_IComp)

        fused_idx, fused_scores = fused_indexer_score_topk(
            c_Q,
            K_IComp,
            idx.W_IUQ,
            idx.w_heads,
            num_indexer_heads=idx.num_indexer_heads,
            indexer_head_dim=idx.indexer_head_dim,
            top_k=idx.top_k,
        )

        # Indices should agree exactly when scores are well-separated.
        self.assertTrue(torch.equal(fused_idx, ref_idx))
        torch.testing.assert_close(
            fused_scores, ref_scores.float(), rtol=1e-5, atol=1e-5
        )

    def test_top_k_clamped_to_T_kc(self):
        idx, c_Q, K_IComp = self._setup(T_kc=3, top_k=10)
        fused_idx, fused_scores = fused_indexer_score_topk(
            c_Q,
            K_IComp,
            idx.W_IUQ,
            idx.w_heads,
            num_indexer_heads=idx.num_indexer_heads,
            indexer_head_dim=idx.indexer_head_dim,
            top_k=10,
        )
        # k_eff should be clamped to T_kc=3.
        self.assertEqual(fused_idx.shape[-1], 3)
        self.assertEqual(fused_scores.shape[-1], 3)

    def test_zero_input_returns_zero_scores(self):
        idx, c_Q, K_IComp = self._setup()
        c_Q_zero = torch.zeros_like(c_Q)
        # All scores should be 0 (ReLU(0) = 0); top-k indices arbitrary but valid.
        _, fused_scores = fused_indexer_score_topk(
            c_Q_zero,
            K_IComp,
            idx.W_IUQ,
            idx.w_heads,
            num_indexer_heads=idx.num_indexer_heads,
            indexer_head_dim=idx.indexer_head_dim,
            top_k=idx.top_k,
        )
        torch.testing.assert_close(
            fused_scores, torch.zeros_like(fused_scores), rtol=0, atol=0
        )

    def test_negative_head_weight_flips_sign(self):
        """w_h < 0 should *subtract* relu logits, exercising the per-head
        weighted accumulation rather than a fixed positive sum."""
        idx, c_Q, K_IComp = self._setup(H=2)
        with torch.no_grad():
            idx.w_heads.data = torch.tensor([1.0, -1.0])

        ref_idx, ref_scores = idx(c_Q, K_IComp)
        fused_idx, fused_scores = fused_indexer_score_topk(
            c_Q,
            K_IComp,
            idx.W_IUQ,
            idx.w_heads,
            num_indexer_heads=2,
            indexer_head_dim=idx.indexer_head_dim,
            top_k=idx.top_k,
        )
        self.assertTrue(torch.equal(fused_idx, ref_idx))
        torch.testing.assert_close(
            fused_scores, ref_scores.float(), rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    main()
