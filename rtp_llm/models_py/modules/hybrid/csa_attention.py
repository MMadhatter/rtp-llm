"""Compressed Sparse Attention (CSA) reference, DeepSeek-V4 (paper §2.3).

Same structural skeleton as :mod:`hca_attention` but with two differences:

  1. **Compressor is overlapping (two-branch)** — uses
     :func:`compressor.csa_compress` with ``m = 4`` (V4-Flash).

  2. **Lightning indexer + sparse MQA**: instead of attending over *all*
     compressed entries, each query first scores all entries with a
     low-rank multi-head ReLU dot product and keeps only the
     ``top_k`` highest. The core MQA then operates on that sparse subset.

Like :mod:`hca_attention`, this is the reference path; the production
path will swap the indexer + sparse attention for FlashMLA's CSA kernel.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.hybrid.compressor import csa_compress
from rtp_llm.models_py.modules.hybrid.hca_attention import (
    GroupedOutputProjection,
    apply_partial_rope,
    mqa_attention_with_sink,
    precompute_rope_cache,
)


# ---------------------------------------------------------------------------
class CsaLightningIndexer(nn.Module):
    """Cheap multi-head ReLU dot-product scorer over compressed CSA entries.

    Structure (paper §2.3, lightning indexer):

      c_Q_t          : (..., q_lora_rank)  — already LoRA-projected query
      Q_indexer      : c_Q_t · W_IUQ  ->  (..., num_indexer_heads * indexer_head_dim)
      K_IComp        : pre-computed CSA-compressed K stream (..., T_kc, indexer_head_dim)
      I_{t,s}        : Σ_h w_h · ReLU(q_h · K_s)            -> (..., T_q, T_kc)
      topk           : indices and gathered values for the top ``top_k`` entries

    The indexer is independent of the main K/V branch — it has its own
    compression path and its own per-head weights ``w_h``.
    """

    def __init__(
        self,
        q_lora_rank: int,
        num_indexer_heads: int,
        indexer_head_dim: int,
        top_k: int,
        *,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        self.num_indexer_heads = num_indexer_heads
        self.indexer_head_dim = indexer_head_dim
        self.top_k = top_k

        f = {"device": device, "dtype": dtype}
        self.W_IUQ = nn.Parameter(torch.empty(
            q_lora_rank, num_indexer_heads * indexer_head_dim, **f
        ))
        # One scalar weight per indexer head.
        self.w_heads = nn.Parameter(torch.empty(num_indexer_heads, **f))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.W_IUQ, std=0.02)
        nn.init.ones_(self.w_heads)

    def forward(
        self,
        c_Q: torch.Tensor,                # (B, T_q, q_lora_rank)
        K_IComp: torch.Tensor,            # (B, T_kc, indexer_head_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(topk_idx, topk_scores)`` of shape ``(B, T_q, k_eff)``.

        ``k_eff = min(top_k, T_kc)``. The caller uses ``topk_idx`` to gather
        from the main K/V compressed streams before running sparse MQA.
        """
        B, T_q, _ = c_Q.shape
        T_kc = K_IComp.shape[-2]
        H = self.num_indexer_heads
        D = self.indexer_head_dim
        k_eff = min(self.top_k, T_kc)

        # Q_indexer: (B, T_q, H * D) -> (B, T_q, H, D)
        Q_idx = (c_Q @ self.W_IUQ).view(B, T_q, H, D)
        # K is single-(MQA-style) stream broadcast across indexer heads:
        # logits (B, T_q, H, T_kc).
        # einsum: q (B, T_q, H, D) x k (B, T_kc, D) -> (B, T_q, H, T_kc)
        raw = torch.einsum("bqhd,bkd->bqhk", Q_idx, K_IComp)
        # ReLU then per-head weighted sum.
        relu = torch.relu(raw)
        scores = torch.einsum("bqhk,h->bqk", relu, self.w_heads)   # (B, T_q, T_kc)

        topk_scores, topk_idx = scores.topk(k_eff, dim=-1)
        return topk_idx, topk_scores


# ---------------------------------------------------------------------------
def _gather_topk_kv(
    K_compressed: torch.Tensor,   # (B, T_kc, D)
    V_compressed: torch.Tensor,   # (B, T_kc, D)
    topk_idx: torch.Tensor,       # (B, T_q, k_eff)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather a per-query subset of the compressed K/V streams.

    Returns ``K_sparse, V_sparse`` of shape ``(B, T_q, k_eff, D)``.
    """
    B, T_kc, D = K_compressed.shape
    T_q, k_eff = topk_idx.shape[1], topk_idx.shape[2]
    # Expand to (B, T_q, k_eff, D) and gather along T_kc.
    idx = topk_idx.unsqueeze(-1).expand(B, T_q, k_eff, D)
    K_b = K_compressed.unsqueeze(1).expand(B, T_q, T_kc, D)
    V_b = V_compressed.unsqueeze(1).expand(B, T_q, T_kc, D)
    K_sparse = torch.gather(K_b, 2, idx)
    V_sparse = torch.gather(V_b, 2, idx)
    return K_sparse, V_sparse


def sparse_mqa_with_sink(
    Q: torch.Tensor,                      # (B, T_q, H, D)
    K_sparse: torch.Tensor,               # (B, T_q, k_eff, D)
    V_sparse: torch.Tensor,               # (B, T_q, k_eff, D)
    K_window: Optional[torch.Tensor],     # (B, T_q, n_win, D)
    V_window: Optional[torch.Tensor],     # (B, T_q, n_win, D)
    sink_logits: torch.Tensor,            # (H,)
    *,
    swa_valid_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """MQA + sink + optional SWA bypass over a per-query sparse KV subset.

    Returns ``(B, T_q, H, D)``.
    """
    B, T_q, H, D = Q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # Sparse-KV logits: (B, T_q, H, k_eff)
    logits_s = torch.einsum("bqhd,bqkd->bqhk", Q, K_sparse) * scale

    if K_window is not None:
        logits_w = torch.einsum("bqhd,bqkd->bqhk", Q, K_window) * scale
        if swa_valid_mask is not None:
            logits_w = logits_w.masked_fill(swa_valid_mask.unsqueeze(2), float("-inf"))
        logits = torch.cat([logits_s, logits_w], dim=-1)
    else:
        logits = logits_s

    # Sink logit per head, broadcast.
    sink = sink_logits.view(1, 1, H, 1).expand(B, T_q, H, 1)
    logits_with_sink = torch.cat([logits, sink], dim=-1)
    weights = torch.softmax(logits_with_sink.float(), dim=-1).to(Q.dtype)

    k_eff = K_sparse.shape[2]
    n_w = K_window.shape[2] if K_window is not None else 0
    w_s = weights[..., :k_eff]
    w_w = weights[..., k_eff : k_eff + n_w]

    out_s = torch.einsum("bqhk,bqkd->bqhd", w_s, V_sparse)
    if K_window is not None:
        out_w = torch.einsum("bqhk,bqkd->bqhd", w_w, V_window)
        out_s = out_s + out_w
    return out_s


# ---------------------------------------------------------------------------
class CsaAttention(nn.Module):
    """Reference CSA attention block (paper §2.3, CSA branch)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        rope_head_dim: int,
        m: int,
        q_lora_rank: int,
        o_groups: int,
        o_lora_rank: int,
        num_indexer_heads: int,
        indexer_head_dim: int,
        top_k: int,
        *,
        n_win: int = 128,
        rope_base: float = 160_000.0,
        rope_max_pos: int = 4096,
        rms_eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.m = m
        self.q_lora_rank = q_lora_rank
        self.n_win = n_win
        self.rope_base = rope_base
        self.scale = 1.0 / (head_dim ** 0.5)

        f = {"device": device, "dtype": dtype}

        # Q LoRA path: shared trunk c_Q_t feeds both UQ (main heads) and IUQ (indexer).
        self.W_DQ = nn.Parameter(torch.empty(hidden_size, q_lora_rank, **f))
        self.W_UQ = nn.Parameter(torch.empty(q_lora_rank, num_heads * head_dim, **f))

        # CSA two-branch projections for K and V (independent).
        self.W_K_a = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_K_b = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_KZ_a = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_KZ_b = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.bias_K_a = nn.Parameter(torch.empty(m, head_dim, **f))
        self.bias_K_b = nn.Parameter(torch.empty(m, head_dim, **f))

        self.W_V_a = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_V_b = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_VZ_a = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_VZ_b = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.bias_V_a = nn.Parameter(torch.empty(m, head_dim, **f))
        self.bias_V_b = nn.Parameter(torch.empty(m, head_dim, **f))

        # Indexer's K stream — its own CSA compression with indexer_head_dim.
        self.W_IK_a = nn.Parameter(torch.empty(hidden_size, indexer_head_dim, **f))
        self.W_IK_b = nn.Parameter(torch.empty(hidden_size, indexer_head_dim, **f))
        self.W_IKZ_a = nn.Parameter(torch.empty(hidden_size, indexer_head_dim, **f))
        self.W_IKZ_b = nn.Parameter(torch.empty(hidden_size, indexer_head_dim, **f))
        self.bias_IK_a = nn.Parameter(torch.empty(m, indexer_head_dim, **f))
        self.bias_IK_b = nn.Parameter(torch.empty(m, indexer_head_dim, **f))

        # Q/K RMSNorm weights.
        self.q_norm_weight = nn.Parameter(torch.ones(head_dim, **f))
        self.k_norm_weight = nn.Parameter(torch.ones(head_dim, **f))

        # Sink logit per head.
        self.sink_logits = nn.Parameter(torch.zeros(num_heads, **f))

        cos, sin = precompute_rope_cache(
            rope_max_pos, rope_head_dim, rope_base, device=device, dtype=dtype
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.indexer = CsaLightningIndexer(
            q_lora_rank=q_lora_rank,
            num_indexer_heads=num_indexer_heads,
            indexer_head_dim=indexer_head_dim,
            top_k=top_k,
            dtype=dtype, device=device,
        )

        self.o_proj = GroupedOutputProjection(
            num_heads=num_heads, head_dim=head_dim, hidden_size=hidden_size,
            o_groups=o_groups, o_lora_rank=o_lora_rank,
            dtype=dtype, device=device,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (self.W_DQ, self.W_UQ,
                  self.W_K_a, self.W_K_b, self.W_KZ_a, self.W_KZ_b,
                  self.W_V_a, self.W_V_b, self.W_VZ_a, self.W_VZ_b,
                  self.W_IK_a, self.W_IK_b, self.W_IKZ_a, self.W_IKZ_b):
            nn.init.normal_(p, std=0.02)
        for b in (self.bias_K_a, self.bias_K_b, self.bias_V_a, self.bias_V_b,
                  self.bias_IK_a, self.bias_IK_b):
            nn.init.zeros_(b)
        nn.init.zeros_(self.sink_logits)
        nn.init.ones_(self.q_norm_weight)
        nn.init.ones_(self.k_norm_weight)

    @staticmethod
    def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + eps).to(x.dtype) * weight

    def forward(
        self,
        H: torch.Tensor,                       # (B, T, hidden_size)
        positions: torch.Tensor,               # (T,)
        compressed_positions: Optional[torch.Tensor] = None,
        swa_valid_mask: Optional[torch.Tensor] = None,
        rms_eps: float = 1e-6,
    ) -> torch.Tensor:
        B, T, _ = H.shape

        # --- Q LoRA ----------------------------------------------------------
        c_Q = H @ self.W_DQ                                 # (B, T, q_lora_rank)
        Q = (c_Q @ self.W_UQ).view(B, T, self.num_heads, self.head_dim)

        # --- CSA-compressed K, V, indexer-K ---------------------------------
        K_comp = csa_compress(
            H, self.W_K_a, self.W_K_b, self.W_KZ_a, self.W_KZ_b,
            self.bias_K_a, self.bias_K_b, self.m,
        )
        V_comp = csa_compress(
            H, self.W_V_a, self.W_V_b, self.W_VZ_a, self.W_VZ_b,
            self.bias_V_a, self.bias_V_b, self.m,
        )
        K_IComp = csa_compress(
            H, self.W_IK_a, self.W_IK_b, self.W_IKZ_a, self.W_IKZ_b,
            self.bias_IK_a, self.bias_IK_b, self.m,
        )
        T_kc = K_comp.shape[-2]

        # --- RMSNorm + partial RoPE -----------------------------------------
        Q = self._rmsnorm(Q, self.q_norm_weight, rms_eps)
        K_comp = self._rmsnorm(K_comp, self.k_norm_weight, rms_eps)

        if compressed_positions is None:
            compressed_positions = (
                torch.arange(T_kc, device=H.device, dtype=torch.long) * self.m
                + self.m // 2
            )
        cos_q = self.rope_cos[positions]
        sin_q = self.rope_sin[positions]
        cos_k = self.rope_cos[compressed_positions]
        sin_k = self.rope_sin[compressed_positions]
        Q = apply_partial_rope(Q, cos_q, sin_q, self.rope_head_dim)
        K_comp = apply_partial_rope(K_comp, cos_k, sin_k, self.rope_head_dim)

        # --- Lightning indexer top-k selection ------------------------------
        topk_idx, _ = self.indexer(c_Q, K_IComp)            # (B, T, k_eff)
        K_sparse, V_sparse = _gather_topk_kv(K_comp, V_comp, topk_idx)

        # --- SWA window (raw H projected to a single MQA head) --------------
        if self.n_win > 0:
            K_raw = H @ self.W_K_a                          # use 'a' projection arbitrarily
            V_raw = H @ self.W_V_a
            K_window, V_window, mask = self._build_swa_views(K_raw, V_raw, B, T)
            if swa_valid_mask is None:
                swa_valid_mask = mask
        else:
            K_window = V_window = None

        # --- Sparse MQA core ------------------------------------------------
        attn_out = sparse_mqa_with_sink(
            Q, K_sparse, V_sparse, K_window, V_window, self.sink_logits,
            swa_valid_mask=swa_valid_mask, scale=self.scale,
        )                                                    # (B, T, H, D)

        # --- Inverse partial RoPE on output ---------------------------------
        attn_out = apply_partial_rope(
            attn_out, cos_q, sin_q, self.rope_head_dim, inverse=True,
        )

        # --- Grouped output projection --------------------------------------
        return self.o_proj(attn_out)

    def _build_swa_views(
        self, K_raw: torch.Tensor, V_raw: torch.Tensor, B: int, T: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_win = self.n_win
        t_idx = torch.arange(T, device=K_raw.device).unsqueeze(-1)
        offsets = torch.arange(n_win, device=K_raw.device).unsqueeze(0)
        gather_idx = t_idx - (n_win - 1) + offsets
        invalid = ~(gather_idx >= 0)
        gather_idx = gather_idx.clamp_min(0)
        K_window = K_raw[:, gather_idx, :]                   # (B, T, n_win, D)
        V_window = V_raw[:, gather_idx, :]
        invalid = invalid.unsqueeze(0).expand(B, -1, -1)
        return K_window, V_window, invalid
