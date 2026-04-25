"""Online-softmax MQA + sink + SWA-bypass attention.

Reference :func:`hca_attention.mqa_attention_with_sink` materialises the
``(B, H, T_q, T_kc + n_win + 1)`` logits tensor (sink concatenated as the
last column) before softmaxing. For long contexts this is the largest
intermediate of the entire attention block (V4-Flash decode at 1M ctx ≈
``H=64 × 250k entries × 4 bytes ≈ 64 GiB`` in fp32 — only fits because we
chunk).

The online formulation below avoids the concat by folding the sink into
the softmax denominator directly (``Z = Σexp(logits) + exp(sink)``) and
running the value-blend in two streams (compressed-K and SWA-window),
re-using the same denominator. Math is identical to the reference; the
saving is the extra concat + the duplicate ``softmax`` materialisation.

Source: vLLM PR #40760 ``vllm/v1/attention/backends/mla/sparse_swa.py``
(``_apply_sink_softmax`` function) + SGLang PR #23600
``python/sglang/srt/layers/attention/compressed/paged_prefill.py``. The
production CUDA kernel is ``csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu``;
this Python helper is the small-batch / bring-up reference.
"""

from typing import Optional

import torch


def online_sink_mqa(
    Q: torch.Tensor,  # (B, T_q, H, D)
    K_compressed: torch.Tensor,  # (B, T_kc, D)            single MQA KV head
    V_compressed: torch.Tensor,  # (B, T_kc, D)
    K_window: Optional[torch.Tensor],  # (B, T_q, n_win, D) per-query SWA tail
    V_window: Optional[torch.Tensor],  # (B, T_q, n_win, D)
    sink_logits: torch.Tensor,  # (H,)                    per-head sink logit
    *,
    causal_compressed_mask: Optional[torch.Tensor] = None,  # (B, T_q, T_kc)
    swa_valid_mask: Optional[torch.Tensor] = None,  # (B, T_q, n_win)
    scale: Optional[float] = None,
) -> torch.Tensor:
    """MQA softmax attention with an in-line sink term, no concat.

    Returns ``(B, T_q, H, D)`` attention output (still in rotated frame —
    caller applies the inverse RoPE).

    The trick: with ``Z = Σ_k exp(logit_k) + exp(sink)``, the per-key
    weight is ``exp(logit_k) / Z`` and the sink contributes nothing to
    the value-side reduction (it's a "no-op key"). We therefore never need
    to gather a value for the sink; the only effect is on the denominator,
    which we compute in fp32 once and divide both compressed + window
    contributions by it.
    """
    B, T_q, H, D = Q.shape
    if scale is None:
        scale = 1.0 / (D**0.5)

    # ---- Compressed-KV branch ---------------------------------------------
    Q_h = Q.transpose(1, 2)  # (B, H, T_q, D)
    K_c = K_compressed.unsqueeze(1)  # (B, 1, T_kc, D)
    V_c = V_compressed.unsqueeze(1)  # (B, 1, T_kc, D)
    logits_c = torch.einsum("bhqd,bnkd->bhqk", Q_h, K_c) * scale  # (B,H,T_q,T_kc)
    if causal_compressed_mask is not None:
        logits_c = logits_c.masked_fill(
            causal_compressed_mask.unsqueeze(1), float("-inf")
        )

    # ---- SWA window branch ------------------------------------------------
    if K_window is not None:
        # Q (B, T_q, H, D) x K_w (B, T_q, n_win, D) → (B, T_q, H, n_win)
        logits_w = torch.einsum("bqhd,bqkd->bqhk", Q, K_window) * scale
        if swa_valid_mask is not None:
            logits_w = logits_w.masked_fill(swa_valid_mask.unsqueeze(2), float("-inf"))
        # Reshape to (B, H, T_q, n_win) for ease of joint normalisation.
        logits_w = logits_w.permute(0, 2, 1, 3).contiguous()
    else:
        logits_w = None

    # ---- Joint max for numerically stable softmax (incl. sink) ------------
    sink = sink_logits.view(1, H, 1, 1).expand(B, H, T_q, 1).float()
    # Per (B, H, T_q) max across compressed + window + sink.
    max_c = logits_c.amax(dim=-1, keepdim=True)
    if logits_w is not None:
        max_w = logits_w.amax(dim=-1, keepdim=True)
        global_max = torch.maximum(torch.maximum(max_c, max_w), sink)
    else:
        global_max = torch.maximum(max_c, sink)

    # Compute weights * value contribution per branch in fp32 then sum.
    # Denominator includes the sink term implicitly.
    e_c = (logits_c.float() - global_max).exp()
    Z = e_c.sum(dim=-1, keepdim=True)
    if logits_w is not None:
        e_w = (logits_w.float() - global_max).exp()
        Z = Z + e_w.sum(dim=-1, keepdim=True)
    e_sink = (sink - global_max).exp()
    Z = Z + e_sink

    w_c = (e_c / Z).to(Q.dtype)  # (B, H, T_q, T_kc)
    out = torch.einsum("bhqk,bnkd->bhqd", w_c, V_c)  # (B, H, T_q, D)
    if logits_w is not None:
        w_w = (e_w / Z).to(Q.dtype)
        # back to (B, H, T_q, n_win) → matmul with V_w (B, T_q, n_win, D)
        out_w = torch.einsum("bhqk,bqkd->bhqd", w_w, V_window)
        out = out + out_w
    return out.transpose(1, 2).contiguous()  # (B, T_q, H, D)


__all__ = ["online_sink_mqa"]
