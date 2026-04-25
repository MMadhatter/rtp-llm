"""Parity tests for the single-pass fused mHC step.

Compares :func:`fused_mhc_step` against the un-fused
``pre_mix → block_fn → post_mix`` chain on :class:`MhcLayer`. They must
produce identical outputs (the fused path only changes the order of
intermediate materialisation, not the math).

Reference: vLLM PR #40760 ``vllm/model_executor/layers/mhc.py`` and
SGLang PR #23600 ``python/sglang/srt/layers/mhc.py``.
"""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.mhc import MhcLayer
from rtp_llm.models_py.modules.mhc_fused import fused_mhc_step


def _identity(x):
    return x


def _linear(W):
    def fn(x):
        return x @ W

    return fn


class FusedMhcStepParityTest(TestCase):
    def _make_layer(self, hidden_size=8, hc_mult=4, seed=0):
        torch.manual_seed(seed)
        layer = MhcLayer(
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            sinkhorn_iters=10,
            alpha_init=0.5,  # break the trivial alpha=0 case
        )
        with torch.no_grad():
            torch.nn.init.normal_(layer.W_pre, std=0.02)
            torch.nn.init.normal_(layer.W_res, std=0.02)
            torch.nn.init.normal_(layer.W_post, std=0.02)
            torch.nn.init.normal_(layer.S_pre, std=0.05)
            torch.nn.init.normal_(layer.S_res, std=0.05)
            torch.nn.init.normal_(layer.S_post, std=0.05)
            torch.nn.init.normal_(layer.norm_weight, mean=1.0, std=0.05)
        return layer

    def test_matches_reference_with_identity_block(self):
        layer = self._make_layer()
        residual = torch.randn(3, layer.hc_mult, layer.hidden_size)
        ref = layer.forward(residual, _identity)
        fused = fused_mhc_step(layer, residual, _identity)
        torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-5)

    def test_matches_reference_with_linear_block(self):
        layer = self._make_layer()
        torch.manual_seed(123)
        W = torch.randn(layer.hidden_size, layer.hidden_size) * 0.05
        residual = torch.randn(2, layer.hc_mult, layer.hidden_size)
        ref = layer.forward(residual, _linear(W))
        fused = fused_mhc_step(layer, residual, _linear(W))
        torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-5)

    def test_with_leading_batch_dims(self):
        layer = self._make_layer(hidden_size=6, hc_mult=2)
        residual = torch.randn(2, 4, layer.hc_mult, layer.hidden_size)
        ref = layer.forward(residual, _identity)
        fused = fused_mhc_step(layer, residual, _identity)
        torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-5)

    def test_block_fn_called_with_correct_input(self):
        """The fused path must feed block_fn the same A·X tensor that
        pre_mix would emit."""
        layer = self._make_layer()
        residual = torch.randn(3, layer.hc_mult, layer.hidden_size)

        captured_ref = {}

        def _capture_ref(x):
            captured_ref["x"] = x.detach().clone()
            return x

        captured_fused = {}

        def _capture_fused(x):
            captured_fused["x"] = x.detach().clone()
            return x

        layer.forward(residual, _capture_ref)
        fused_mhc_step(layer, residual, _capture_fused)

        torch.testing.assert_close(
            captured_fused["x"], captured_ref["x"], rtol=1e-5, atol=1e-5
        )

    def test_bf16_within_tolerance(self):
        layer = self._make_layer()
        residual = torch.randn(2, layer.hc_mult, layer.hidden_size).bfloat16()
        ref = layer.forward(residual, _identity)
        fused = fused_mhc_step(layer, residual, _identity)
        torch.testing.assert_close(fused.float(), ref.float(), rtol=1e-2, atol=1e-2)

    def test_alpha_zero_makes_static(self):
        """alpha=0 → A,B,C come purely from S_*; output should not depend on
        residual content beyond the linear application."""
        layer = self._make_layer()
        with torch.no_grad():
            layer.alpha_pre.zero_()
            layer.alpha_res.zero_()
            layer.alpha_post.zero_()
        torch.manual_seed(99)
        r1 = torch.randn(2, layer.hc_mult, layer.hidden_size)
        r2 = torch.randn(2, layer.hc_mult, layer.hidden_size)
        # Each call independently should still match the ref.
        for r in (r1, r2):
            ref = layer.forward(r, _identity)
            fused = fused_mhc_step(layer, r, _identity)
            torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    main()
