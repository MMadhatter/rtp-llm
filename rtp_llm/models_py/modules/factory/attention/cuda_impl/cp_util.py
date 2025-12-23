"""
Wrapper for FlashInfer Ragged Prefill Attention
Integrates with RTP-LLM's FMHA infrastructure
"""

import logging
from typing import Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAPrefillImplBase,
)
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs

from .flashinfer_prefill_ragged import (
    FlashInferRaggedPrefillOp,
    warmup_flashinfer_kernels,
)

logger = logging.getLogger(__name__)


class FlashInferRaggedPrefillWrapper(FMHAPrefillImplBase):
    """
    Wrapper for FlashInfer ragged prefill attention.

    This class integrates FlashInferRaggedPrefillOp with RTP-LLM's
    attention framework, providing RoPE handling and cache management.
    """

    def __init__(
        self,
        config: GptInitModelParameters,
        attn_inputs: PyAttentionInputs,
        backend: str = "auto",
        enable_warmup: bool = True,
    ):
        """
        Initialize FlashInfer ragged prefill wrapper.

        Args:
            config: Model configuration
            attn_inputs: Attention input metadata
            backend: FlashInfer backend ("auto", "fa2", or "fa3")
            enable_warmup: Whether to warmup kernels on first use
        """
        # Calculate dimensions
        num_heads = config.head_num // config.tp_size
        num_kv_heads = getattr(config, "num_kv_heads", num_heads) // config.tp_size
        head_dim = config.head_dim

        # Create FlashInfer operator
        fmha_impl = FlashInferRaggedPrefillOp(
            config=config,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            backend=backend,
            causal=True,
        )

        # Initialize base class
        # Note: rope_kvcache_impl should be passed if RoPE is used
        super().__init__(
            fmha_impl=fmha_impl,
            rope_kvcache_impl=None,  # Will be set later if needed
            attn_inputs=attn_inputs,
        )

        self.config = config
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.enable_warmup = enable_warmup
        self._warmed_up = False

        # Prepare FMHA parameters
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)

        logger.info(
            f"Initialized FlashInferRaggedPrefillWrapper: "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, batch_size={self.fmha_params['batch_size']}"
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        """Return the FMHA type."""
        return FMHAType.FLASH_INFER

    def _maybe_warmup(self):
        """Warmup FlashInfer kernels on first use."""
        if self.enable_warmup and not self._warmed_up:
            device = torch.cuda.current_device()
            warmup_flashinfer_kernels(device)
            self._warmed_up = True

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for prefill attention.

        Args:
            q: Query tensor [total_tokens, num_heads, head_dim] or [total_tokens, num_heads * head_dim]
            k: Key tensor [total_tokens, num_kv_heads, head_dim] or [total_tokens, num_kv_heads * head_dim]
            v: Value tensor [total_tokens, num_kv_heads, head_dim] or [total_tokens, num_kv_heads * head_dim]
            kv_cache: Optional KV cache
            layer_id: Layer ID (for multi-layer models)

        Returns:
            Attention output [total_tokens, num_heads, head_dim]
        """
        # Warmup on first use
        self._maybe_warmup()

        # Apply RoPE if rope_kvcache_impl is set
        if self.rope_kvcache_impl is not None and self.rope_params is not None:
            # RoPE is applied to q and k
            # Assuming rope_kvcache_impl has a forward method
            self.rope_kvcache_impl.forward(q, k, kv_cache, self.rope_params)

        # Handle cache store if needed
        if (
            self.attn_inputs.is_prefill
            and self.attn_inputs.cache_store_inputs
            and self.write_cache_store_impl is not None
        ):
            self.write_cache_store_impl(kv_cache)

        # Ensure correct shape for FlashInfer
        # FlashInfer expects [total_tokens, num_heads, head_dim]
        if q.dim() == 2:
            q = q.view(-1, self.num_heads, self.head_dim)
        if k.dim() == 2:
            k = k.view(-1, self.num_kv_heads, self.head_dim)
        if v.dim() == 2:
            v = v.view(-1, self.num_kv_heads, self.head_dim)

        # Call FlashInfer implementation
        attn_output = self.fmha_impl.forward(
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            fmha_params=self.fmha_params,
        )

        return attn_output


def create_flashinfer_ragged_prefill_wrapper(
    config: GptInitModelParameters,
    attn_inputs: PyAttentionInputs,
    backend: str = "auto",
) -> FlashInferRaggedPrefillWrapper:
    """
    Factory function to create FlashInfer ragged prefill wrapper.

    Args:
        config: Model configuration
        attn_inputs: Attention input metadata
        backend: FlashInfer backend

    Returns:
        Configured wrapper
    """
    return FlashInferRaggedPrefillWrapper(
        config=config,
        attn_inputs=attn_inputs,
        backend=backend,
    )
