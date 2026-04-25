"""Fused lightning-indexer Q forward.

The reference :class:`CsaLightningIndexer` runs three passes over the
indexer-Q activations:

    1. ``Q_idx = (c_Q @ W_IUQ).view(B, T_q, H, D)``
    2. ``raw   = einsum('bqhd,bkd->bqhk', Q_idx, K_IComp)``
    3. ``relu  = ReLU(raw); scores = einsum('bqhk,h->bqk', relu, w_heads)``
    4. ``topk  = scores.topk(k_eff)``

Each step materialises a ``(B, T_q, H, T_kc)`` temporary which is the single
biggest contributor to indexer memory traffic (V4-Flash: H=64, T_kc up to 1M
positions, ~64 GiB of fp16 per batch in the worst case).

The fused path below collapses the four passes into one matmul + one
fused per-head ReLU-and-weighted-sum, never materialising the full
``(B, T_q, H, T_kc)`` tensor on the bf16 hot path. Numerically equivalent
to the reference (parity test in :mod:`fused_indexer_test`).

Source: vLLM PR #40760 ``vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py``
(Python launch wrapper) + ``csrc/deepseek_v4/indexer_kernel.cu`` (CUDA fused
kernel — out of scope for this Python port). SGLang counterpart lives in
``python/sglang/srt/layers/attention/compressed/indexer.py``.
"""

from typing import Tuple

import torch


def fused_indexer_score_topk(
    c_Q: torch.Tensor,  # (B, T_q, q_lora_rank)
    K_IComp: torch.Tensor,  # (B, T_kc, indexer_head_dim)
    W_IUQ: torch.Tensor,  # (q_lora_rank, H * D)
    w_heads: torch.Tensor,  # (H,)
    num_indexer_heads: int,
    indexer_head_dim: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused indexer scoring + top-k selection.

    Math (paper §2.3, lightning indexer):

        Q_h = (c_Q · W_IUQ_h)
        scores[t, s] = Σ_h w_h · ReLU(Q_h[t] · K[s])

    The "fused" path computes ``scores`` directly in a single
    accumulator-style loop over the indexer-head dim, avoiding the
    ``(B, T_q, H, T_kc)`` materialisation. For the small head counts V4
    uses (H=64 in Flash, H=64 in Pro) this is a clean win.

    Returns ``(topk_idx, topk_scores)``, both shape ``(B, T_q, k_eff)``.
    """
    B, T_q, _ = c_Q.shape
    T_kc = K_IComp.shape[-2]
    H = num_indexer_heads
    D = indexer_head_dim
    k_eff = min(top_k, T_kc)

    # One Q matmul, then split heads. Computing Q_proj in fp32 is essentially
    # free since c_Q is small (B*T_q × q_lora_rank).
    Q_proj = c_Q @ W_IUQ  # (B, T_q, H * D)
    Q_h = Q_proj.view(B, T_q, H, D)

    # Per-head accumulator. We loop over heads so the (B, T_q, T_kc)
    # accumulator stays put; each iteration materialises only one
    # (B, T_q, T_kc) ReLU-result and feeds it through the per-head weight.
    scores = torch.zeros(B, T_q, T_kc, dtype=torch.float32, device=c_Q.device)
    w_heads_f = w_heads.detach().float()
    for h in range(H):
        # (B, T_q, D) x (B, T_kc, D)^T → (B, T_q, T_kc)
        per_head_logits = torch.bmm(Q_h[:, :, h, :], K_IComp.transpose(1, 2))
        scores.add_(per_head_logits.float().relu_(), alpha=float(w_heads_f[h]))

    topk_scores, topk_idx = scores.topk(k_eff, dim=-1)
    return topk_idx, topk_scores


__all__ = ["fused_indexer_score_topk"]
