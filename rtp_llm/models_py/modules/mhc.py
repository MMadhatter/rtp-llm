"""Manifold-Constrained Hyper-Connections (mHC) — DeepSeek-V4.

Pure PyTorch reference implementation of the mHC residual block defined in the
DeepSeek-V4 paper (§ 2.2, Eq. 1–8). This is what PR-B (M1) ships; the fused
TileLang / DeepGEMM kernel is M1.5.

Shape contract (matches vLLM PR #40760's ``mhc_pre`` / ``mhc_post``):

    residual : (..., n_hc, d)        the expanded residual stream
    layer_in : (..., d)              what we feed to the inner block (attention/MoE)
    A        : (..., n_hc)           pre-mix weights, in [0, 1]
    B        : (..., n_hc, n_hc)     residual mapping, doubly stochastic
    C        : (..., n_hc, 1)        post-mix weights, in [0, post_scale]
    layer_out: (..., d)              the inner block's output, F(layer_in)
    residual': (..., n_hc, d)        B @ residual + C * layer_out

A, B and C are *dynamic* — generated per token from RMSNorm(vec(residual)) — and
each carries a per-token learnable static bias (``S_*``) plus a scalar gating
factor (``alpha_*``). The output mapping ``C`` is doubly bounded ([0, 2] by
default, ``2·sigmoid``); the residual mapping ``B`` is projected onto the
Birkhoff polytope (doubly stochastic matrices) via Sinkhorn-Knopp.

Usage
-----
::

    mhc = MhcLayer(hidden_size=4096, hc_mult=4, sinkhorn_iters=20)

    # Inside the model loop:
    layer_in, params = mhc.pre_mix(residual)        # residual: (..., 4, 4096)
    layer_out = block(layer_in)                     # block input/output: (..., 4096)
    residual = mhc.post_mix(residual, layer_out, params)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

# (A, B, C) cached between pre_mix and post_mix.
MhcParams = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def sinkhorn_knopp(
    logits: torch.Tensor,
    iters: int = 20,
    eps: float = 0.0,
) -> torch.Tensor:
    """Project a (..., n, n) matrix onto the Birkhoff polytope.

    Implements DeepSeek-V4 paper Eq. (8) with one numerical-safety tweak: the
    initial ``M^(0) = exp(B_raw)`` is replaced by a row-wise softmax (subtracts
    the row max before exp). The Sinkhorn fixed point is the same up to scaling,
    but this avoids overflow when the dynamic raw matrix has large entries
    early in training. vLLM's tilelang kernel uses the same trick.

    The iteration order matches the paper: ``M^(t) = T_r(T_c(M^(t-1)))`` —
    column-normalize first, then row-normalize. After ``iters`` passes, the
    last operation is row-normalization, so rows sum to exactly 1 and columns
    sum to ≈ 1 (within Sinkhorn convergence — typically < 1e-4 after 20 iters
    for 4×4 matrices).

    Args:
        logits: ``(..., n, n)`` raw matrix; any sign.
        iters: number of full ``T_r ∘ T_c`` passes (default 20 per the paper).
        eps: small additive constant before each division to avoid degeneracy.

    Returns:
        ``(..., n, n)`` doubly stochastic matrix in the input dtype.
    """
    in_dtype = logits.dtype
    p = torch.softmax(logits.float(), dim=-1)
    if eps:
        p = p + eps
    for _ in range(iters):
        # T_c: column normalization
        p = p / (p.sum(dim=-2, keepdim=True) + eps)
        # T_r: row normalization
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
    return p.to(in_dtype)


def expand_residual(x: torch.Tensor, n_hc: int) -> torch.Tensor:
    """Lift a ``(..., d)`` hidden state to a ``(..., n_hc, d)`` residual stream.

    Channel 0 carries the original state; the remaining channels start at zero
    so that an immediate ``B = I, C = 1`` mHC step preserves the activation.
    DeepSeek-V4's pre-trained checkpoint may encode a different (learned)
    expansion; this is a stable default for unit tests / engine bring-up.
    """
    *prefix, d = x.shape
    out = x.new_zeros(*prefix, n_hc, d)
    out[..., 0, :] = x
    return out


def reduce_residual(x: torch.Tensor) -> torch.Tensor:
    """Collapse a ``(..., n_hc, d)`` residual stream back to ``(..., d)``.

    Default: sum across the ``n_hc`` channels — the conjugate of the channel-0
    expansion when ``B = I``. The production V4 final norm + lm_head will use
    whatever projection the checkpoint was trained with.
    """
    return x.sum(dim=-2)


class MhcLayer(nn.Module):
    """Per-block mHC parameters and forward (pre_mix + post_mix).

    Parameters (per the paper, layer index ``l`` elided):

    ===========  =========================  ==========================
    Tensor       Shape                      Role
    ===========  =========================  ==========================
    ``W_pre``    ``(n_hc·d, n_hc)``         dynamic generator for A
    ``W_res``    ``(n_hc·d, n_hc²)``        dynamic generator for B
    ``W_post``   ``(n_hc·d, n_hc)``         dynamic generator for C
    ``S_pre``    ``(n_hc,)``                static bias for A
    ``S_res``    ``(n_hc, n_hc)``           static bias for B
    ``S_post``   ``(n_hc,)``                static bias for C
    ``alpha_*``  ``(1,)``                   scalar dynamic gate factor
    ``norm_w``   ``(n_hc·d,)``              RMSNorm gain
    ===========  =========================  ==========================

    Args:
        hidden_size: model hidden dim ``d`` (4096 for V4-Flash, 7168 for V4-Pro).
        hc_mult: residual stream width factor ``n_hc`` (4 for V4).
        sinkhorn_iters: Sinkhorn-Knopp passes (20 in the paper, t_max).
        eps: RMSNorm epsilon.
        sinkhorn_eps: additive ε inside Sinkhorn divisions; 0 mimics the paper.
        post_scale: max value of C; the paper uses ``C = 2·sigmoid(...)`` so 2.0.
        alpha_init: initial value for the three gating factors. Paper says
            "initialized to small values"; 0.0 makes mHC a static (input-
            independent) operator at step 0, easing training takeoff.
        dtype / device: passed straight to ``torch.empty`` for parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
        sinkhorn_eps: float = 0.0,
        post_scale: float = 2.0,
        alpha_init: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if hc_mult < 1:
            raise ValueError(f"hc_mult must be >= 1, got {hc_mult}")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")

        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.sinkhorn_eps = sinkhorn_eps
        self.post_scale = post_scale
        self.alpha_init = alpha_init

        n = hc_mult
        nd = n * hidden_size
        f = {"device": device, "dtype": dtype}

        # Dynamic generators
        self.W_pre = nn.Parameter(torch.empty(nd, n, **f))
        self.W_res = nn.Parameter(torch.empty(nd, n * n, **f))
        self.W_post = nn.Parameter(torch.empty(nd, n, **f))

        # Static biases (broadcast across all tokens)
        self.S_pre = nn.Parameter(torch.empty(n, **f))
        self.S_res = nn.Parameter(torch.empty(n, n, **f))
        self.S_post = nn.Parameter(torch.empty(n, **f))

        # Dynamic gating factors (paper: "initialized to small values")
        self.alpha_pre = nn.Parameter(torch.empty(1, **f))
        self.alpha_res = nn.Parameter(torch.empty(1, **f))
        self.alpha_post = nn.Parameter(torch.empty(1, **f))

        # RMSNorm gain over flattened residual stream (length n_hc·d)
        self.norm_weight = nn.Parameter(torch.empty(nd, **f))

        self.reset_parameters()

    # ------------------------------------------------------------------ init
    def reset_parameters(self) -> None:
        nn.init.normal_(self.W_pre, std=0.02)
        nn.init.normal_(self.W_res, std=0.02)
        nn.init.normal_(self.W_post, std=0.02)
        nn.init.zeros_(self.S_pre)
        nn.init.zeros_(self.S_res)
        nn.init.zeros_(self.S_post)
        with torch.no_grad():
            self.alpha_pre.fill_(self.alpha_init)
            self.alpha_res.fill_(self.alpha_init)
            self.alpha_post.fill_(self.alpha_init)
        nn.init.ones_(self.norm_weight)

    # ----------------------------------------------------------- internals
    def _rmsnorm_flat(self, x_flat: torch.Tensor) -> torch.Tensor:
        """RMSNorm with the flattened-residual gain, computed in fp32."""
        in_dtype = x_flat.dtype
        x32 = x_flat.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (x32 * self.norm_weight.float()).to(in_dtype)

    # ------------------------------------------------------- public methods
    def compute_dynamic_params(
        self, residual: torch.Tensor
    ) -> MhcParams:
        """Generate ``(A, B, C)`` per token from the residual stream.

        Args:
            residual: ``(..., n_hc, d)``. Any leading shape; usually
                ``(num_tokens, n_hc, d)`` or ``(batch, seq, n_hc, d)``.

        Returns:
            ``A: (..., n_hc)``, ``B: (..., n_hc, n_hc)``, ``C: (..., n_hc, 1)``
            in fp32 (caller may cast back to the residual dtype).
        """
        if residual.dim() < 2:
            raise ValueError(
                f"residual must have at least 2 dims (..., n_hc, d); "
                f"got shape {tuple(residual.shape)}"
            )
        if residual.shape[-2] != self.hc_mult:
            raise ValueError(
                f"residual.shape[-2] = {residual.shape[-2]} != hc_mult "
                f"({self.hc_mult})"
            )
        if residual.shape[-1] != self.hidden_size:
            raise ValueError(
                f"residual.shape[-1] = {residual.shape[-1]} != hidden_size "
                f"({self.hidden_size})"
            )

        n = self.hc_mult
        # Promote to fp32 for the dynamic-param chain — small matrices, cheap.
        x_flat = residual.flatten(-2).float()                         # (..., n·d)
        x_hat = self._rmsnorm_flat(x_flat)                            # (..., n·d)

        pre_dyn = x_hat @ self.W_pre.float()                          # (..., n)
        res_dyn = x_hat @ self.W_res.float()                          # (..., n²)
        post_dyn = x_hat @ self.W_post.float()                        # (..., n)

        a_raw = self.alpha_pre.float() * pre_dyn + self.S_pre.float()
        c_raw = self.alpha_post.float() * post_dyn + self.S_post.float()
        # Mat(·): R^{1×n²} → R^{n×n}, then add the (n,n) static bias.
        b_raw = (
            self.alpha_res.float() * res_dyn.unflatten(-1, (n, n))
            + self.S_res.float()
        )

        A = torch.sigmoid(a_raw)                                      # (..., n)
        C = self.post_scale * torch.sigmoid(c_raw)                    # (..., n)
        B = sinkhorn_knopp(b_raw, iters=self.sinkhorn_iters,
                           eps=self.sinkhorn_eps)                     # (..., n, n)
        return A, B, C.unsqueeze(-1)                                  # C: (..., n, 1)

    def pre_mix(
        self, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, MhcParams]:
        """Compute layer input ``A·X`` and cache ``(A, B, C)`` for ``post_mix``.

        Returns:
            ``layer_in``: ``(..., d)`` in the residual's dtype.
            ``params``: ``(A, B, C)`` tuple to feed back into ``post_mix``.
        """
        params = self.compute_dynamic_params(residual)
        A, _, _ = params
        # A: (..., n), residual: (..., n, d)  -> layer_in: (..., d)
        layer_in = torch.einsum("...h,...hd->...d", A, residual.float())
        return layer_in.to(residual.dtype), params

    def post_mix(
        self,
        residual: torch.Tensor,
        layer_out: torch.Tensor,
        params: MhcParams,
    ) -> torch.Tensor:
        """Apply ``X_{l+1} = B·X + C·layer_out``.

        Args:
            residual: original ``(..., n_hc, d)`` from before the inner block.
            layer_out: ``(..., d)`` produced by the inner block on ``layer_in``.
            params: the ``(A, B, C)`` tuple returned by :meth:`pre_mix`.

        Returns:
            Updated residual stream ``(..., n_hc, d)`` in ``residual.dtype``.
        """
        _, B, C = params
        if layer_out.shape[:-1] != residual.shape[:-2]:
            raise ValueError(
                f"layer_out leading shape {tuple(layer_out.shape[:-1])} != "
                f"residual leading shape {tuple(residual.shape[:-2])}"
            )
        if layer_out.shape[-1] != self.hidden_size:
            raise ValueError(
                f"layer_out.shape[-1] = {layer_out.shape[-1]} != hidden_size "
                f"({self.hidden_size})"
            )

        # B: (..., n, n), residual: (..., n, d)  -> (..., n, d)
        bx = torch.einsum("...hg,...gd->...hd", B, residual.float())
        # C: (..., n, 1), layer_out: (..., d) -> broadcast to (..., n, d)
        c_f = C * layer_out.float().unsqueeze(-2)
        return (bx + c_f).to(residual.dtype)

    def forward(
        self,
        residual: torch.Tensor,
        layer_fn,
    ) -> torch.Tensor:
        """Convenience: ``pre_mix → layer_fn → post_mix`` in one call.

        Args:
            residual: ``(..., n_hc, d)`` residual stream entering the block.
            layer_fn: callable ``(..., d) -> (..., d)`` (attention + MoE block).

        Returns:
            New residual stream ``(..., n_hc, d)``.
        """
        layer_in, params = self.pre_mix(residual)
        layer_out = layer_fn(layer_in)
        return self.post_mix(residual, layer_out, params)
