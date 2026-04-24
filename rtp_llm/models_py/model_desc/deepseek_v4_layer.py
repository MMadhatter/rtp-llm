"""DeepSeek-V4 decoder layer — reference assembly (PR-E, HCA-only path).

Composes the M2 / M5 building blocks into a runnable per-layer module:

  residual stream (..., n_hc, d)
        │
        ▼  mHC pre-mix:  (A, B, C) ← f(residual);  layer_in = A · residual
  layer_in (..., d)
        │
        ▼  HCA / CSA attention block (with grouped O proj baked in)
  attn_out (..., d)
        │
        ▼  mHC post-mix:  residual ← B · residual + C · attn_out
  residual (..., n_hc, d)
        │
        ▼  mHC pre-mix again
  ffn_in (..., d)
        │
        ▼  V4 MoE (hash route OR sqrt_softplus topk + clamped SwiGLU experts)
  ffn_out (..., d)
        │
        ▼  mHC post-mix
  residual (..., n_hc, d)

This module is **reference quality**, not the production hot path. The C++
fused kernel + per-layer ckpt loader land in separate PRs (see
develop_ds_v4.md §6 PR-E). Tests exercise it on small dims to verify shape
/dtype/grad-flow contracts.

Production integration TODO:
  * Real per-layer weight loader keyed by HF V4 ckpt names (not done yet)
  * Swap PyTorch reference HCA for the FlashMLA V4 backend
  * Hook into ``GptModelBase.kv_cache.get_layer_cache(i)`` instead of the
    layer running stateless
"""

from typing import Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.hybrid.hca_attention import HcaAttention
from rtp_llm.models_py.modules.mhc import MhcLayer
from rtp_llm.models_py.modules.moe.clamped_swiglu import clamped_swiglu_split
from rtp_llm.models_py.modules.moe.hash_router import hash_route_topk
from rtp_llm.models_py.modules.moe.v4_gating import noaux_tc_topk_v4


# ---------------------------------------------------------------------------
class DeepSeekV4MoE(nn.Module):
    """Reference V4 MoE: ``num_routed_experts`` experts + 1 shared expert.

    Routing strategy depends on ``layer_idx``:

      * ``layer_idx < num_hash_layers`` (V4 default 3) — deterministic hash
        routing on ``token_ids``. The router is content-agnostic.
      * else — learned ``sqrt(softplus)`` topk routing on a linear gate.

    Each expert is a clamped-SwiGLU MLP:
      ``y = silu(clamp(gate, ≤10)) * clamp(linear, [-10, 10])  →  W_o``

    Notes:
      * The shared expert always fires regardless of routing; it is *not*
        gated. Its output is added to the routed-expert sum.
      * ``moe_intermediate_size`` is the per-expert MLP inter dim; the shared
        expert reuses this width times ``n_shared_experts``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_routed_experts: int,
        moe_intermediate_size: int,
        top_k: int,
        layer_idx: int,
        num_hash_layers: int,
        *,
        n_shared_experts: int = 1,
        routed_scaling_factor: float = 1.5,
        norm_topk_prob: bool = True,
        swiglu_limit: float = 10.0,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.top_k = top_k
        self.layer_idx = layer_idx
        self.num_hash_layers = num_hash_layers
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.swiglu_limit = swiglu_limit
        self.use_hash_routing = layer_idx < num_hash_layers

        f = {"device": device, "dtype": dtype}
        # Routed experts — stack along an extra leading dim so a single batched
        # matmul covers all of them. (Reference; production uses MegaMoE.)
        self.W_gate = nn.Parameter(torch.empty(
            num_routed_experts, hidden_size, moe_intermediate_size, **f
        ))
        self.W_up = nn.Parameter(torch.empty(
            num_routed_experts, hidden_size, moe_intermediate_size, **f
        ))
        self.W_down = nn.Parameter(torch.empty(
            num_routed_experts, moe_intermediate_size, hidden_size, **f
        ))

        # Shared expert (always-on)
        shared_inter = moe_intermediate_size * n_shared_experts
        self.W_shared_gate = nn.Parameter(torch.empty(hidden_size, shared_inter, **f))
        self.W_shared_up = nn.Parameter(torch.empty(hidden_size, shared_inter, **f))
        self.W_shared_down = nn.Parameter(torch.empty(shared_inter, hidden_size, **f))

        if not self.use_hash_routing:
            self.gate = nn.Parameter(torch.empty(hidden_size, num_routed_experts, **f))
            # e_score_correction_bias — added before topk selection only.
            self.e_score_bias = nn.Parameter(torch.empty(num_routed_experts, **f))
        else:
            self.register_parameter("gate", None)
            self.register_parameter("e_score_bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (self.W_gate, self.W_up, self.W_down,
                  self.W_shared_gate, self.W_shared_up, self.W_shared_down):
            nn.init.normal_(p, std=0.02)
        if self.gate is not None:
            nn.init.normal_(self.gate, std=0.02)
            nn.init.zeros_(self.e_score_bias)

    def _expert_forward(self, x: torch.Tensor, expert_id: int) -> torch.Tensor:
        """Run one routed expert. ``x``: ``(N, hidden)`` -> ``(N, hidden)``."""
        gate = x @ self.W_gate[expert_id]
        linear = x @ self.W_up[expert_id]
        h = clamped_swiglu_split(gate, linear, self.swiglu_limit)
        return h @ self.W_down[expert_id]

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = x @ self.W_shared_gate
        linear = x @ self.W_shared_up
        h = clamped_swiglu_split(gate, linear, self.swiglu_limit)
        return h @ self.W_shared_down

    def forward(
        self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``x``: ``(B, T, hidden)``. Returns ``(B, T, hidden)``.

        ``token_ids``: ``(B, T)`` int tensor. Required when this layer uses
        hash routing; ignored otherwise.
        """
        B, T, H = x.shape
        flat = x.reshape(B * T, H)

        # ---- Routing -------------------------------------------------------
        if self.use_hash_routing:
            if token_ids is None:
                raise ValueError(
                    f"layer {self.layer_idx} uses hash routing but token_ids "
                    f"is None; pass token_ids through the layer call."
                )
            ids_flat = token_ids.reshape(-1)
            topk_idx, gate_vals = hash_route_topk(
                ids_flat, self.num_routed_experts, self.top_k,
                self.routed_scaling_factor,
            )
            gate_vals = gate_vals.to(x.dtype)
        else:
            logits = flat @ self.gate                           # (N, E)
            topk_idx, gate_vals = noaux_tc_topk_v4(
                logits, self.e_score_bias, self.top_k,
                self.routed_scaling_factor,
                norm_topk_prob=self.norm_topk_prob,
            )

        # ---- Routed experts (reference: scatter loop over experts) ---------
        out = torch.zeros_like(flat)
        for e in range(self.num_routed_experts):
            # mask: (N, top_k) of which slots picked expert e
            mask = (topk_idx == e)
            if not mask.any():
                continue
            tok_idx, slot_idx = mask.nonzero(as_tuple=True)
            x_e = flat[tok_idx]                                  # (M, H)
            y_e = self._expert_forward(x_e, e)                   # (M, H)
            scale = gate_vals[tok_idx, slot_idx].unsqueeze(-1)   # (M, 1)
            out.index_add_(0, tok_idx, y_e * scale)

        # ---- Shared expert (always on) ------------------------------------
        out = out + self._shared_forward(flat)
        return out.reshape(B, T, H)


# ---------------------------------------------------------------------------
class DeepSeekV4DecoderLayer(nn.Module):
    """One V4 decoder layer: mHC ⨂ HCA-attention ⨂ V4-MoE.

    The residual stream lives at width ``hc_mult * hidden_size`` (paper §2.5).
    This layer expects the residual stream as input and returns the updated
    residual stream — the model module is responsible for the initial
    expansion (``mhc.expand_residual``) and final reduction.
    """

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
        hc_mult: int,
        num_routed_experts: int,
        moe_intermediate_size: int,
        moe_top_k: int,
        layer_idx: int,
        num_hash_layers: int,
        *,
        n_win: int = 128,
        rope_base: float = 160_000.0,
        rope_max_pos: int = 4096,
        n_shared_experts: int = 1,
        routed_scaling_factor: float = 1.5,
        swiglu_limit: float = 10.0,
        sinkhorn_iters: int = 20,
        rms_eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult

        # mHC has two slots: one wrapping attention, one wrapping MoE. The
        # paper uses *separate* parameter sets for the two — each sub-block
        # generates its own A/B/C from the *current* residual stream.
        self.mhc_attn = MhcLayer(
            hidden_size=hidden_size, hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters, eps=rms_eps,
            dtype=dtype, device=device,
        )
        self.mhc_ffn = MhcLayer(
            hidden_size=hidden_size, hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters, eps=rms_eps,
            dtype=dtype, device=device,
        )

        self.attention = HcaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            m_prime=m_prime,
            q_lora_rank=q_lora_rank,
            o_groups=o_groups,
            o_lora_rank=o_lora_rank,
            n_win=n_win,
            rope_base=rope_base,
            rope_max_pos=rope_max_pos,
            rms_eps=rms_eps,
            dtype=dtype, device=device,
        )

        self.moe = DeepSeekV4MoE(
            hidden_size=hidden_size,
            num_routed_experts=num_routed_experts,
            moe_intermediate_size=moe_intermediate_size,
            top_k=moe_top_k,
            layer_idx=layer_idx,
            num_hash_layers=num_hash_layers,
            n_shared_experts=n_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            swiglu_limit=swiglu_limit,
            dtype=dtype, device=device,
        )

    def forward(
        self,
        residual: torch.Tensor,             # (B, T, n_hc, d)
        positions: torch.Tensor,            # (T,)
        token_ids: Optional[torch.Tensor],  # (B, T) — needed iff hash routing
    ) -> torch.Tensor:
        # ---- Attention sub-block -----------------------------------------
        attn_in, attn_params = self.mhc_attn.pre_mix(residual)         # (B, T, d)
        attn_out = self.attention(attn_in, positions)                  # (B, T, d)
        residual = self.mhc_attn.post_mix(residual, attn_out, attn_params)

        # ---- FFN sub-block (MoE) -----------------------------------------
        ffn_in, ffn_params = self.mhc_ffn.pre_mix(residual)
        ffn_out = self.moe(ffn_in, token_ids)                           # (B, T, d)
        residual = self.mhc_ffn.post_mix(residual, ffn_out, ffn_params)
        return residual
