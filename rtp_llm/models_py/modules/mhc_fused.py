"""Fused mHC step — single-pass pre_mix → block_fn → post_mix.

The reference :class:`MhcLayer` exposes pre_mix / post_mix as separate
methods so the model loop can sandwich an arbitrary inner block between
them. That ergonomics costs three things on the hot path:

  1. ``A``, ``B``, ``C`` are *materialised* per-token tensors that survive
     across the block call (kept in the ``MhcParams`` tuple).
  2. RMSNorm of the flattened residual is computed once, then re-materialised
     for the dynamic generators ``W_pre / W_res / W_post``.
  3. The post-mix takes the same residual + new ``layer_out`` through one
     more pass to produce the next residual stream.

The fused helper below collapses the three matmul + the residual update
into a single function. The block must be a closure that takes
``layer_in: (..., d)`` and returns ``layer_out: (..., d)`` — same
contract as ``MhcLayer.forward``'s ``layer_fn`` argument. Numerical
behaviour is identical to ``pre_mix → fn → post_mix`` (parity test in
:mod:`mhc_fused_test`).

Source: vLLM PR #40760 ``vllm/model_executor/layers/mhc.py::MhcLayer.forward``
(Python launch wrapper that batches RMSNorm + 3-way matmul + Sinkhorn into
one CUDA call) + SGLang PR #23600 ``python/sglang/srt/layers/mhc.py``. The
production fused TileLang kernel lands in M6; this is the reference path.
"""

from typing import Callable

import torch

from rtp_llm.models_py.modules.mhc import MhcLayer, sinkhorn_knopp


def fused_mhc_step(
    mhc: MhcLayer,
    residual: torch.Tensor,  # (..., n_hc, d)
    block_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Run one mHC sandwich on ``residual`` in a single function.

    Equivalent to::

        layer_in, params = mhc.pre_mix(residual)
        layer_out = block_fn(layer_in)
        residual = mhc.post_mix(residual, layer_out, params)

    but never materialises the ``MhcParams`` tuple on the Python side, so
    autograd has one fewer set of intermediates to retain.

    The math (paper §2.2, Eq. 1–8):

        x_hat = RMSNorm(vec(residual))
        a_raw = α_pre  · (x_hat · W_pre)  + S_pre
        b_raw = α_res  · (x_hat · W_res)  + S_res         # reshape (..., n, n)
        c_raw = α_post · (x_hat · W_post) + S_post
        A = σ(a_raw)
        B = sinkhorn(softmax(b_raw))
        C = post_scale · σ(c_raw)
        layer_in    = einsum('...h,...hd->...d', A, residual)
        layer_out   = block_fn(layer_in)
        residual'   = einsum('...hg,...gd->...hd', B, residual)
                    + C[..., None] * layer_out[..., None, :]
    """
    n = mhc.hc_mult
    d = mhc.hidden_size
    in_dtype = residual.dtype
    leading = residual.shape[:-2]
    nd = n * d

    # ---- RMSNorm in fp32 -------------------------------------------------
    x_flat = residual.float().reshape(*leading, nd)
    var = x_flat.pow(2).mean(-1, keepdim=True)
    x_hat = x_flat * torch.rsqrt(var + mhc.eps) * mhc.norm_weight.float()

    # ---- Dynamic generators: 3-way matmul on the same x_hat --------------
    # Weights kept in fp32 so the dynamic generator stays in fp32 throughout
    # — matches the reference path which only quantises back at the end.
    W_pre = mhc.W_pre.float()
    W_res = mhc.W_res.float()
    W_post = mhc.W_post.float()
    a_raw = mhc.alpha_pre.float() * (x_hat @ W_pre) + mhc.S_pre.float()  # (..., n)
    res_dyn = (x_hat @ W_res).reshape(*leading, n, n)  # (..., n, n)
    b_raw = mhc.alpha_res.float() * res_dyn + mhc.S_res.float()
    c_raw = mhc.alpha_post.float() * (x_hat @ W_post) + mhc.S_post.float()

    # ---- Constraints ------------------------------------------------------
    A = torch.sigmoid(a_raw)
    C = mhc.post_scale * torch.sigmoid(c_raw)  # (..., n)
    B = sinkhorn_knopp(b_raw, iters=mhc.sinkhorn_iters, eps=mhc.sinkhorn_eps)

    # ---- Inner block ------------------------------------------------------
    # layer_in = Σ_h A_h * residual_h
    layer_in = torch.einsum("...h,...hd->...d", A.to(in_dtype), residual)
    layer_out = block_fn(layer_in).to(torch.float32)

    # ---- Residual update --------------------------------------------------
    bx = torch.einsum("...hg,...gd->...hd", B, residual.float())
    cf = C.unsqueeze(-1) * layer_out.unsqueeze(-2)
    return (bx + cf).to(in_dtype)


__all__ = ["fused_mhc_step"]
