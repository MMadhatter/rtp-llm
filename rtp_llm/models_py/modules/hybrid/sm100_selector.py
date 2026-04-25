"""Single source of truth for "is this a GB200/Blackwell build?" gating.

Keeps the model-side imports cheap on non-Blackwell targets — the CUTLASS
NVFP4 MM kernel and the FlashInfer cute-DSL grouped GEMM both live behind
``rtp_llm.ops.compute_ops``/``flashinfer`` imports that fail at import
time on Hopper. Centralising the gate avoids duplicating the
"if Blackwell then FP4 else bf16" branch in every call site.

Three booleans, all cached:

  * :func:`has_blackwell_gpu` — capability ≥ (10, 0). Necessary but not
    sufficient: the bf16-only Blackwell SKUs (e.g. simulators) still fail
    the FP4 import.
  * :func:`has_fp4_kernels` — the underlying ``cutlass_scaled_fp4_mm`` and
    ``scaled_fp4_quant`` ops are importable.
  * :func:`has_flashinfer_cutedsl` — the ``flashinfer.cute_dsl`` package
    is importable (needed for the MegaMoE expert GEMM).

The model-side selector should treat ``True`` from each as "use the FP4
path here"; absence is silently rerouted to the bf16/fp8 fused fallbacks
already in this directory.

Source: vLLM PR #40760 ``vllm/utils/__init__.py::current_platform`` and
the ``has_blackwell()`` / ``has_nvfp4_grouped()`` helpers in
``vllm/_custom_ops.py``. SGLang PR #23600 ``python/sglang/srt/utils.py``
exposes the same gate as ``has_sm100_fp4()``.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


@functools.cache
def has_blackwell_gpu() -> bool:
    """Cuda compute capability major == 10 (Blackwell, sm_100)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major == 10
    except Exception as e:  # pragma: no cover - defensive on broken envs
        logger.debug("has_blackwell_gpu probe failed: %s", e)
        return False


@functools.cache
def has_fp4_kernels() -> bool:
    """The FP4 quant + MM ops are importable from ``rtp_llm.ops.compute_ops``.

    Importing fails on Hopper builds because the CUTLASS NVFP4 GEMM is
    only registered for ``sm_100a``.
    """
    if not has_blackwell_gpu():
        return False
    try:
        from rtp_llm.ops.compute_ops import (  # noqa: F401
            cutlass_scaled_fp4_mm,
            scaled_fp4_quant,
        )

        return True
    except Exception as e:
        logger.debug("FP4 kernel import probe failed: %s", e)
        return False


@functools.cache
def has_flashinfer_cutedsl() -> bool:
    """The FlashInfer cute-DSL grouped GEMM is importable.

    Required for the NVFP4 MegaMoE expert path; absent on most Hopper
    images and on builds that pin an older FlashInfer.
    """
    if not has_blackwell_gpu():
        return False
    try:
        from flashinfer.cute_dsl.blockscaled_gemm import (  # noqa: F401
            grouped_gemm_nt_masked,
        )

        return True
    except Exception as e:
        logger.debug("FlashInfer cute-DSL probe failed: %s", e)
        return False


def use_fp4_indexer() -> bool:
    """Top-level gate the indexer module checks once per layer build."""
    return has_fp4_kernels()


def use_megamoe() -> bool:
    """Top-level gate the MoE module checks once per layer build."""
    return has_flashinfer_cutedsl()


__all__ = [
    "has_blackwell_gpu",
    "has_fp4_kernels",
    "has_flashinfer_cutedsl",
    "use_fp4_indexer",
    "use_megamoe",
]
