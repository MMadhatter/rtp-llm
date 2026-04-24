"""Heavily Compressed Attention (HCA) reference, DeepSeek-V4 (paper §2.3).

Reference implementation, PyTorch-only. Wires together:

  * LoRA-decomposed Q projection: ``q = (h @ W_DQ) @ W_UQ``
  * Token-level KV compression via :func:`compressor.hca_compress`
    (``m'`` raw tokens → 1 compressed entry)
  * Q/K RMSNorm, then *partial* RoPE on the last ``rope_head_dim`` dims
    (V4 uses ``base = 160_000``)
  * MQA core attention (single KV head shared by all Q heads), with:
    * **Attention sink** — one extra learnable logit per head added to the
      softmax denominator
    * **Sliding-window bypass** — each query also attends to the last
      ``n_win`` *uncompressed* KV tokens (not just compressed entries)
  * **Inverse partial RoPE** on the attention output — undo the rotation
    so the residual stream stays in the un-rotated frame
  * **Grouped Output Projection** — ``n_h`` heads partitioned into
    ``o_groups`` groups; each group projects ``g_heads * head_dim``
    → ``o_lora_rank`` → ``hidden_size / o_groups`` and the per-group
    outputs concat back into ``hidden_size``

This file is the *reference*; the production path is the FlashMLA kernel
extension noted in the dev plan. Tests use this module directly.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.hybrid.compressor import hca_compress


# ---------------------------------------------------------------------------
# Partial RoPE helpers — pure-PyTorch, batched over leading dims.
# ---------------------------------------------------------------------------
def precompute_rope_cache(
    max_pos: int,
    rope_dim: int,
    base: float = 160_000.0,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` of shape ``(max_pos, rope_dim // 2)``.

    Uses the GPT-NeoX/Llama style: ``theta_i = 1 / base^(2i / rope_dim)``.
    """
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    half = rope_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)            # (max_pos, half)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # NeoX rotation: split last dim in halves, swap.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the last ``rope_dim`` dims of ``x`` by per-position ``(cos, sin)``.

    Args:
        x: ``(..., L, head_dim)`` tensor; last ``rope_dim`` dims get rotated.
        cos, sin: ``(L, rope_dim // 2)`` from :func:`precompute_rope_cache`.
        rope_dim: number of trailing dims to rotate.
        inverse: if True, rotate by ``-pos`` (equivalent to flipping the sign
            of ``sin``). Used to undo the query-side rotation on the attention
            output so the residual stream stays in the un-rotated frame.
    """
    head_dim = x.shape[-1]
    if rope_dim > head_dim:
        raise ValueError(f"rope_dim {rope_dim} > head_dim {head_dim}")
    if rope_dim == 0:
        return x

    nope = head_dim - rope_dim
    if nope:
        x_nope, x_rope = x.split([nope, rope_dim], dim=-1)
    else:
        x_nope, x_rope = None, x

    # cos/sin currently (L, rope_dim/2); duplicate to (L, rope_dim) by [c c | s s]
    # so a (rope_dim,) elementwise rotation matches `_rotate_half` form.
    cos_full = torch.cat([cos, cos], dim=-1)      # (L, rope_dim)
    sin_full = torch.cat([sin, sin], dim=-1)
    if inverse:
        sin_full = -sin_full

    # Broadcast (L, rope_dim) over leading shape (..., L, rope_dim).
    while cos_full.dim() < x_rope.dim():
        cos_full = cos_full.unsqueeze(0)
        sin_full = sin_full.unsqueeze(0)

    rotated = (x_rope * cos_full) + (_rotate_half(x_rope) * sin_full)
    if x_nope is None:
        return rotated
    return torch.cat([x_nope, rotated], dim=-1)


# ---------------------------------------------------------------------------
# MQA + sink + SWA-bypass core attention.
# ---------------------------------------------------------------------------
def mqa_attention_with_sink(
    Q: torch.Tensor,                      # (B, T_q, H, D)
    K_compressed: torch.Tensor,           # (B, T_kc, D)        — single KV head
    V_compressed: torch.Tensor,           # (B, T_kc, D)
    K_window: Optional[torch.Tensor],     # (B, T_q, n_win, D)  — per-query SWA tail
    V_window: Optional[torch.Tensor],     # (B, T_q, n_win, D)
    sink_logits: torch.Tensor,            # (H,)                — per-head sink in softmax denom
    *,
    causal_compressed_mask: Optional[torch.Tensor] = None,  # (B, T_q, T_kc) bool, True = mask out
    swa_valid_mask: Optional[torch.Tensor] = None,          # (B, T_q, n_win) bool, True = mask out
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Single-step reference for V4's MQA core.

    Returns ``(B, T_q, H, D)`` attention output (still in rotated frame for the
    rope sub-block — caller must apply :func:`apply_partial_rope(..., inverse=True)`).
    """
    B, T_q, H, D = Q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # --- Compressed-KV branch -----------------------------------------------
    # K shape (B, T_kc, D) -> (B, 1, T_kc, D), Q (B, T_q, H, D) -> (B, H, T_q, D)
    Q_h = Q.transpose(1, 2)                             # (B, H, T_q, D)
    K_c = K_compressed.unsqueeze(1)                     # (B, 1, T_kc, D)
    V_c = V_compressed.unsqueeze(1)                     # (B, 1, T_kc, D)
    # Logits over compressed keys: (B, H, T_q, T_kc)
    logits_c = torch.einsum("bhqd,bnkd->bhqk", Q_h, K_c) * scale
    if causal_compressed_mask is not None:
        # mask: (B, T_q, T_kc) -> (B, 1, T_q, T_kc)
        logits_c = logits_c.masked_fill(causal_compressed_mask.unsqueeze(1), float("-inf"))

    # --- SWA-window branch (per-query own slice of last n_win raw KV) -------
    if K_window is not None:
        # Q (B, T_q, H, D), K_w (B, T_q, n_win, D) -> logits_w (B, T_q, H, n_win)
        logits_w = torch.einsum("bqhd,bqkd->bqhk", Q, K_window) * scale
        if swa_valid_mask is not None:
            # (B, T_q, n_win) -> broadcast to (B, T_q, 1, n_win)
            logits_w = logits_w.masked_fill(swa_valid_mask.unsqueeze(2), float("-inf"))
        # Reorder to (B, H, T_q, n_win) so we can concat along the keys dim.
        logits_w = logits_w.permute(0, 2, 1, 3)         # (B, H, T_q, n_win)
        logits = torch.cat([logits_c, logits_w], dim=-1)
    else:
        logits = logits_c

    # --- Sink: append a per-head logit, then softmax along keys -------------
    # sink_logits (H,) -> (1, H, 1, 1) and concat onto last dim.
    sink = sink_logits.view(1, H, 1, 1).expand(B, H, T_q, 1)
    logits_with_sink = torch.cat([logits, sink], dim=-1)
    weights = torch.softmax(logits_with_sink.float(), dim=-1).to(Q.dtype)

    # Drop the sink column from the weights when computing the output.
    n_compressed = K_compressed.shape[1]
    n_window = K_window.shape[2] if K_window is not None else 0
    w_c = weights[..., :n_compressed]                   # (B, H, T_q, T_kc)
    w_w = weights[..., n_compressed : n_compressed + n_window]  # (B, H, T_q, n_win)
    # weights[..., -1] is the sink (discarded as a value contribution).

    # Compressed-KV value: (B, H, T_q, D)
    out_c = torch.einsum("bhqk,bnkd->bhqd", w_c, V_c)
    if K_window is not None:
        # (B, H, T_q, n_win) x (B, T_q, n_win, D) -> (B, H, T_q, D)
        # Move H to dim 1 for v_window then einsum back.
        out_w = torch.einsum("bhqk,bqkd->bhqd", w_w, V_window)
        out = out_c + out_w
    else:
        out = out_c
    return out.transpose(1, 2).contiguous()             # (B, T_q, H, D)


# ---------------------------------------------------------------------------
class GroupedOutputProjection(nn.Module):
    """Two-step grouped projection used by V4's output head.

    ``n_h`` attention heads are partitioned into ``o_groups`` groups of
    ``g_heads = n_h / o_groups`` heads. Each group projects:

        (g_heads * head_dim)  --W_a-->  (o_lora_rank,)  --W_b-->  (hidden / o_groups,)

    All groups' outputs concatenate to ``hidden_size``. Compared to a single
    dense ``(n_h * head_dim) -> hidden_size`` projection this saves
    ``g_heads * o_groups * head_dim * (1 - rank / (head_dim * g_heads))``
    parameters per layer.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        o_groups: int,
        o_lora_rank: int,
        *,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        if num_heads % o_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by o_groups ({o_groups})"
            )
        if hidden_size % o_groups != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by o_groups ({o_groups})"
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.g_heads = num_heads // o_groups
        self.out_per_group = hidden_size // o_groups

        f = {"device": device, "dtype": dtype}
        # Per-group weights stacked along an extra leading dim so the whole
        # projection is one batched matmul.
        self.W_a = nn.Parameter(torch.empty(
            o_groups, self.g_heads * head_dim, o_lora_rank, **f
        ))
        self.W_b = nn.Parameter(torch.empty(
            o_groups, o_lora_rank, self.out_per_group, **f
        ))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.W_a, std=0.02)
        nn.init.normal_(self.W_b, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(..., n_h, head_dim)`` -> ``(..., hidden_size)``."""
        *prefix, n_h, d_h = x.shape
        if n_h != self.num_heads or d_h != self.head_dim:
            raise ValueError(
                f"Input shape {x.shape} doesn't match n_h={self.num_heads}, "
                f"head_dim={self.head_dim}"
            )
        # (..., g, g_heads * head_dim)
        x_g = x.reshape(*prefix, self.o_groups, self.g_heads * d_h)
        # (..., g, lora_rank)
        z = torch.einsum("...gi,gij->...gj", x_g, self.W_a)
        # (..., g, out_per_group)
        y = torch.einsum("...gj,gjk->...gk", z, self.W_b)
        return y.reshape(*prefix, self.hidden_size)


# ---------------------------------------------------------------------------
class HcaAttention(nn.Module):
    """Reference HCA attention block (paper §2.3, HCA branch)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        rope_head_dim: int,
        m_prime: int,
        q_lora_rank: int,
        o_groups: int,
        o_lora_rank: int,
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
        self.m_prime = m_prime
        self.q_lora_rank = q_lora_rank
        self.n_win = n_win
        self.rope_base = rope_base
        self.scale = 1.0 / (head_dim ** 0.5)

        f = {"device": device, "dtype": dtype}

        # Q LoRA path: (h) -> (q_lora_rank) -> (n_h * head_dim).
        self.W_DQ = nn.Parameter(torch.empty(hidden_size, q_lora_rank, **f))
        self.W_UQ = nn.Parameter(torch.empty(q_lora_rank, num_heads * head_dim, **f))
        # Single MQA KV head: shared across all Q heads.
        self.W_KV = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_Z = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.bias_pos = nn.Parameter(torch.empty(m_prime, head_dim, **f))
        # V is computed with its own projection on raw H, then compressed too.
        self.W_V = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.W_VZ = nn.Parameter(torch.empty(hidden_size, head_dim, **f))
        self.bias_v_pos = nn.Parameter(torch.empty(m_prime, head_dim, **f))

        # Q / K RMSNorm weights — one per head dim. Q is per-head, K shared.
        self.q_norm_weight = nn.Parameter(torch.ones(head_dim, **f))
        self.k_norm_weight = nn.Parameter(torch.ones(head_dim, **f))

        # Attention sink: one extra learnable logit per head.
        self.sink_logits = nn.Parameter(torch.zeros(num_heads, **f))

        # RoPE cos/sin cache: registered as buffer so it moves with .to(device).
        cos, sin = precompute_rope_cache(
            rope_max_pos, rope_head_dim, rope_base, device=device, dtype=dtype
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.o_proj = GroupedOutputProjection(
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            o_groups=o_groups,
            o_lora_rank=o_lora_rank,
            dtype=dtype,
            device=device,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (self.W_DQ, self.W_UQ, self.W_KV, self.W_Z,
                  self.W_V, self.W_VZ):
            nn.init.normal_(p, std=0.02)
        nn.init.zeros_(self.bias_pos)
        nn.init.zeros_(self.bias_v_pos)
        nn.init.zeros_(self.sink_logits)
        nn.init.ones_(self.q_norm_weight)
        nn.init.ones_(self.k_norm_weight)

    @staticmethod
    def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """RMSNorm over the last dim (head_dim)."""
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + eps).to(x.dtype)
        return x_norm * weight

    def forward(
        self,
        H: torch.Tensor,                       # (B, T, hidden_size)
        positions: torch.Tensor,               # (T,) — absolute token positions for queries
        compressed_positions: Optional[torch.Tensor] = None,  # (T_kc,) for compressed KV
        causal_compressed_mask: Optional[torch.Tensor] = None,
        swa_valid_mask: Optional[torch.Tensor] = None,
        rms_eps: float = 1e-6,
    ) -> torch.Tensor:
        """Run one HCA block on a (batch, seq, hidden) input.

        ``positions`` indexes the rope cache for queries. ``compressed_positions``
        indexes it for the *compressed* KV stream — by default we use the
        midpoint of each ``m'``-token block (``floor(i*m' + m'/2)``). Override
        when the call-site has a more accurate convention (e.g. block-end).
        """
        B, T, _ = H.shape

        # --- Q projection (LoRA) --------------------------------------------
        c_Q = H @ self.W_DQ                                # (B, T, q_lora_rank)
        q = c_Q @ self.W_UQ                                # (B, T, n_h * head_dim)
        Q = q.view(B, T, self.num_heads, self.head_dim)

        # --- KV compression -------------------------------------------------
        # Single-head MQA: (B, T, head_dim) -> (B, T_kc, head_dim).
        K_comp = hca_compress(H, self.W_KV, self.W_Z, self.bias_pos, self.m_prime)
        V_comp = hca_compress(H, self.W_V, self.W_VZ, self.bias_v_pos, self.m_prime)
        T_kc = K_comp.shape[-2]

        # --- RMSNorm Q (per-head) and K (single-head) -----------------------
        Q = self._rmsnorm(Q, self.q_norm_weight, rms_eps)
        K_comp = self._rmsnorm(K_comp, self.k_norm_weight, rms_eps)

        # --- Partial RoPE on Q and K_comp -----------------------------------
        if compressed_positions is None:
            # Default: midpoint of each block.
            compressed_positions = (
                torch.arange(T_kc, device=H.device, dtype=torch.long) * self.m_prime
                + self.m_prime // 2
            )
        cos_q = self.rope_cos[positions]
        sin_q = self.rope_sin[positions]
        cos_k = self.rope_cos[compressed_positions]
        sin_k = self.rope_sin[compressed_positions]

        Q = apply_partial_rope(Q, cos_q, sin_q, self.rope_head_dim)
        # K_comp shape (B, T_kc, head_dim): treat as single (...,L,D)
        K_comp = apply_partial_rope(K_comp, cos_k, sin_k, self.rope_head_dim)

        # --- SWA bypass: per-query view onto the last n_win raw H tokens ----
        # We use H itself as the un-compressed K/V (it would be a per-layer
        # K/V projection in the production path; keeping it as raw H here
        # is faithful enough for the reference's structural test).
        if self.n_win > 0 and T > 0:
            K_window, V_window, swa_mask = self._build_swa_views(H, B, T)
            if swa_valid_mask is None:
                swa_valid_mask = swa_mask
        else:
            K_window = V_window = None

        # --- Core MQA + sink ------------------------------------------------
        attn_out = mqa_attention_with_sink(
            Q, K_comp, V_comp,
            K_window, V_window,
            self.sink_logits,
            causal_compressed_mask=causal_compressed_mask,
            swa_valid_mask=swa_valid_mask,
            scale=self.scale,
        )  # (B, T, H, D)

        # --- Inverse partial RoPE on output (V4 trick) ---------------------
        attn_out = apply_partial_rope(
            attn_out, cos_q, sin_q, self.rope_head_dim, inverse=True,
        )

        # --- Grouped output projection -------------------------------------
        return self.o_proj(attn_out)

    def _build_swa_views(
        self, H: torch.Tensor, B: int, T: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialise per-query windows ``K_window, V_window`` of shape
        ``(B, T, n_win, head_dim)`` plus a ``(B, T, n_win)`` validity mask.

        Out-of-bounds positions get zero KV and a True mask entry (will be
        masked to ``-inf`` in the softmax). Returns:

          K_window, V_window  — projections of the raw H tokens (W_KV / W_V)
          mask                — True == invalid / pad
        """
        # Project raw H to a single MQA KV head for the window read.
        K_raw = H @ self.W_KV                    # (B, T, head_dim)
        V_raw = H @ self.W_V

        n_win = self.n_win
        # Build (T, n_win) index map: window for query t is positions
        # max(0, t - n_win + 1) .. t. Out-of-range positions clamp to 0
        # and are masked.
        t_idx = torch.arange(T, device=H.device).unsqueeze(-1)         # (T, 1)
        offsets = torch.arange(n_win, device=H.device).unsqueeze(0)    # (1, n_win)
        gather_idx = t_idx - (n_win - 1) + offsets                     # (T, n_win)
        valid = gather_idx >= 0
        invalid_mask = ~valid                                          # (T, n_win)
        gather_idx = gather_idx.clamp_min(0)

        # Index along T dim of K_raw / V_raw: result (B, T, n_win, D).
        # K_raw[:, gather_idx, :] -> (B, T, n_win, D).
        K_window = K_raw[:, gather_idx, :]
        V_window = V_raw[:, gather_idx, :]
        # Broadcast mask to (B, T, n_win)
        invalid_mask = invalid_mask.unsqueeze(0).expand(B, -1, -1)
        return K_window, V_window, invalid_mask
