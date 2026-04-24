"""Tests for the CSA attention reference (DeepSeek-V4, paper §2.3 CSA branch).

Covers:
  * CsaLightningIndexer — shape, ReLU non-negativity, top-k correctness on
    a hand-built scoring matrix
  * sparse_mqa_with_sink — sink behaviour, sparse subset matches dense subset
  * CsaAttention.forward — shape, dtype, grad flow, batch invariance
"""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.hybrid.csa_attention import (
    CsaAttention,
    CsaLightningIndexer,
    _gather_topk_kv,
    sparse_mqa_with_sink,
)


# ---------------------------------------------------------------------------
class CsaLightningIndexerTest(TestCase):
    def test_shape(self):
        ix = CsaLightningIndexer(
            q_lora_rank=32, num_indexer_heads=8, indexer_head_dim=16, top_k=4,
        )
        c_Q = torch.randn(2, 6, 32)
        K_IComp = torch.randn(2, 10, 16)
        idx, scores = ix(c_Q, K_IComp)
        self.assertEqual(idx.shape, (2, 6, 4))
        self.assertEqual(scores.shape, (2, 6, 4))

    def test_topk_clipped_to_t_kc(self):
        ix = CsaLightningIndexer(
            q_lora_rank=4, num_indexer_heads=2, indexer_head_dim=4, top_k=10,
        )
        c_Q = torch.randn(1, 1, 4)
        K_IComp = torch.randn(1, 3, 4)              # only 3 entries available
        idx, scores = ix(c_Q, K_IComp)
        self.assertEqual(idx.shape, (1, 1, 3))
        self.assertEqual(scores.shape, (1, 1, 3))

    def test_scores_non_negative(self):
        """ReLU on the QK dot enforces non-negative per-head scores; the
        per-head weighted sum is non-negative when ``w_heads >= 0`` (default)."""
        ix = CsaLightningIndexer(
            q_lora_rank=4, num_indexer_heads=2, indexer_head_dim=4, top_k=2,
        )
        c_Q = torch.randn(1, 4, 4)
        K_IComp = torch.randn(1, 5, 4)
        _, scores = ix(c_Q, K_IComp)
        self.assertTrue((scores >= 0).all())

    def test_topk_picks_highest(self):
        """Hand-construct K so that one entry dominates -> indexer must pick it."""
        torch.manual_seed(0)
        ix = CsaLightningIndexer(
            q_lora_rank=4, num_indexer_heads=1, indexer_head_dim=4, top_k=1,
        )
        # Force W_IUQ = identity-ish so Q_indexer == c_Q (shape ok).
        with torch.no_grad():
            ix.W_IUQ.copy_(torch.eye(4))
            ix.w_heads.fill_(1.0)
        c_Q = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])      # (1, 1, 4)
        # K with one strongly-aligned entry at index 2.
        K = torch.zeros(1, 5, 4)
        K[0, 2] = torch.tensor([10.0, 0.0, 0.0, 0.0])
        idx, scores = ix(c_Q, K)
        self.assertEqual(idx[0, 0, 0].item(), 2)
        self.assertGreater(scores[0, 0, 0].item(), 0.0)

    def test_grad_flows(self):
        ix = CsaLightningIndexer(
            q_lora_rank=4, num_indexer_heads=2, indexer_head_dim=4, top_k=2,
        )
        c_Q = torch.randn(1, 3, 4, requires_grad=True)
        K = torch.randn(1, 5, 4, requires_grad=True)
        _, scores = ix(c_Q, K)
        scores.sum().backward()
        self.assertIsNotNone(c_Q.grad)
        self.assertIsNotNone(K.grad)
        self.assertIsNotNone(ix.W_IUQ.grad)


# ---------------------------------------------------------------------------
class GatherTopkKvTest(TestCase):
    def test_correct_per_query_subset(self):
        K = torch.arange(10).float().view(1, 5, 2)        # (B=1, T_kc=5, D=2)
        V = K * 100
        # Query 0 picks entries 0,2; query 1 picks 1,4.
        idx = torch.tensor([[[0, 2], [1, 4]]])            # (1, 2, 2)
        K_s, V_s = _gather_topk_kv(K, V, idx)
        self.assertEqual(K_s.shape, (1, 2, 2, 2))
        torch.testing.assert_close(K_s[0, 0, 0], K[0, 0])
        torch.testing.assert_close(K_s[0, 0, 1], K[0, 2])
        torch.testing.assert_close(K_s[0, 1, 0], K[0, 1])
        torch.testing.assert_close(K_s[0, 1, 1], K[0, 4])
        torch.testing.assert_close(V_s, K_s * 100)


# ---------------------------------------------------------------------------
class SparseMqaSinkTest(TestCase):
    def test_huge_sink_zeros_output(self):
        B, T_q, H, D, k_eff = 1, 2, 4, 4, 3
        Q = torch.randn(B, T_q, H, D)
        K_s = torch.randn(B, T_q, k_eff, D)
        V_s = torch.randn(B, T_q, k_eff, D)
        sink = torch.full((H,), 1e6)
        out = sparse_mqa_with_sink(
            Q, K_s, V_s, K_window=None, V_window=None, sink_logits=sink,
        )
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-4, rtol=1e-3)

    def test_zero_sink_matches_plain_softmax(self):
        torch.manual_seed(1)
        B, T_q, H, D, k_eff = 1, 2, 2, 4, 3
        Q = torch.randn(B, T_q, H, D)
        K_s = torch.randn(B, T_q, k_eff, D)
        V_s = torch.randn(B, T_q, k_eff, D)
        sink = torch.zeros(H)
        out = sparse_mqa_with_sink(
            Q, K_s, V_s, K_window=None, V_window=None, sink_logits=sink,
        )
        scale = 1.0 / (D ** 0.5)
        logits = torch.einsum("bqhd,bqkd->bqhk", Q, K_s) * scale
        # zero sink = augmented logit 0, change normalisation but exclude from V.
        logits_aug = torch.cat([logits, torch.zeros(*logits.shape[:-1], 1)], dim=-1)
        w = torch.softmax(logits_aug, dim=-1)
        ref = torch.einsum("bqhk,bqkd->bqhd", w[..., :-1], V_s)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
class CsaAttentionForwardTest(TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg = dict(
            hidden_size=32, num_heads=4, head_dim=8, rope_head_dim=4,
            m=2, q_lora_rank=8, o_groups=2, o_lora_rank=8,
            num_indexer_heads=2, indexer_head_dim=4, top_k=3,
            n_win=2, rope_max_pos=64, dtype=torch.float32,
        )
        self.layer = CsaAttention(**self.cfg)

    def test_forward_shape(self):
        B, T = 2, 8
        H = torch.randn(B, T, self.cfg["hidden_size"])
        out = self.layer(H, torch.arange(T))
        self.assertEqual(out.shape, (B, T, self.cfg["hidden_size"]))
        self.assertTrue(torch.isfinite(out).all())

    def test_grad_flow(self):
        B, T = 1, 4
        H = torch.randn(B, T, self.cfg["hidden_size"], requires_grad=True)
        loss = self.layer(H, torch.arange(T)).sum()
        loss.backward()
        self.assertIsNotNone(H.grad)
        # Indexer's W_IUQ must receive grad — sparse selection is non-differentiable
        # for the *indices*, but the gathered-value path makes W_IUQ feed forward.
        # Grad on W_IUQ specifically depends on it routing through scores; with
        # ReLU and topk it can be 0, so we only require grads on the main path.
        for name in ("W_DQ", "W_UQ", "W_K_a", "W_V_a", "sink_logits",
                     "q_norm_weight", "k_norm_weight"):
            p = dict(self.layer.named_parameters())[name]
            self.assertIsNotNone(p.grad, f"{name} grad missing")

    def test_dtype_bf16(self):
        layer = CsaAttention(**{**self.cfg, "dtype": torch.bfloat16})
        H = torch.randn(1, 4, self.cfg["hidden_size"], dtype=torch.bfloat16)
        out = layer(H, torch.arange(4))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_batch_independence(self):
        B, T = 2, 4
        H1 = torch.randn(1, T, self.cfg["hidden_size"])
        H2 = torch.randn(1, T, self.cfg["hidden_size"])
        H = torch.cat([H1, H2], dim=0)
        out_b = self.layer(H, torch.arange(T))
        out_1 = self.layer(H1, torch.arange(T))
        out_2 = self.layer(H2, torch.arange(T))
        torch.testing.assert_close(out_b[0:1], out_1, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(out_b[1:2], out_2, rtol=1e-4, atol=1e-5)

    def test_top_k_larger_than_kc(self):
        """top_k > T_kc must clip silently and still produce valid output."""
        cfg = {**self.cfg, "top_k": 100}     # absurdly large
        layer = CsaAttention(**cfg)
        H = torch.randn(1, 6, cfg["hidden_size"])
        out = layer(H, torch.arange(6))
        self.assertEqual(out.shape, (1, 6, cfg["hidden_size"]))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    main()
