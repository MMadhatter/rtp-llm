"""
Unified Sparse MLA implementation for both prefill and decode stages.
Uses flash_mla_sparse_fwd kernel with triton-based index conversion.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

# Check CUDA version for flash_mla compatibility
_FLASH_MLA_AVAILABLE = False
try:
    if torch.version.cuda:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        if (major, minor) >= (12, 9):
            from flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata

            _FLASH_MLA_AVAILABLE = True
except (ImportError, AttributeError, ValueError) as e:
    import logging

    logging.warning(f"flash_mla not available: {e}. Requires CUDA >= 12.9")

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, barrier
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_kv_indices,
    generate_q_indices,
)
from rtp_llm.ops import AttentionConfigs, CPProcessorType, FMHAConfig, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    CPSlotMapper,
    KVCache,
    PyAttentionInputs,
    rtp_llm_ops,
    compute_ops,
)

from .flashmla_sparse_impl import (
    SparseMlaFp8DecodeParams,
    SparseMlaFp8Op,
    SparseMlaImpl,
)


class SparseMlaFp8CPOp(SparseMlaFp8Op):
    """
    Context Parallel prefill for Sparse MLA (FP8).

    All-gather KV, restore to logical order, write via the same kv_cache_write_op as
    non-CP (line 508 in flashmla_sparse_impl), then run attention in two parts (q0, q1)
    using self.block_table, self._fp8_kernel_metadata, self._convert_topk_indices_to_global.
    """

    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        page_size: int,
        softmax_extra_scale: float,
        top_k: int,
        attn_inputs: Optional[PyAttentionInputs] = None,
        parallelism_config: Optional[ParallelismConfig] = None,
    ):
        super().__init__(
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=softmax_extra_scale,
            top_k=top_k,
        )

        self.attn_inputs = attn_inputs
        self.cp_info = attn_inputs.context_parallel_info
        assert (
            self.cp_info is not None
        ), "Context parallel info is required for SparseMlaFp8CPOp"

        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size
        self.processor_type = parallelism_config.prefill_cp_config.processor_type
        self.kv_cache_sharded = parallelism_config.prefill_cp_config.kv_cache_sharded
        self.device = torch.cuda.current_device()

        self.cp_slot_mapper = CPSlotMapper(
            cp_rank=self.prefill_cp_rank,
            cp_size=self.prefill_cp_size,
            block_size=page_size,
        )

        self.kv_restore_unpad_indices = None

        self.q0_idx = None
        self.q1_idx = None
        self.q0_idx_global = None
        self.q1_idx_global = None
        self.kv0_idx = None
        self.kv1_idx = None
        self.kv_cache_write_op = None
        self.write_cache_store_impl = None

    def plan(
        self, mla_params: rtp_llm_ops.FlashInferMlaAttnParams, block_table: torch.Tensor
    ) -> None:
        self.block_table = block_table
        self.mla_params = mla_params

        cp_info = self.cp_info
        padding_mask = cp_info.prefill_qkv_padding_mask
        kv_restore_indices = cp_info.prefill_qkv_restore_indice
        self.kv_restore_unpad_indices = kv_restore_indices[padding_mask == 1]

        chunk_lengths = cp_info.prefill_cp_chunk_lengths
        if hasattr(chunk_lengths, "cpu"):
            chunk_lengths_list = chunk_lengths.cpu().tolist()
        else:
            chunk_lengths_list = list(chunk_lengths)
        local_tokens = sum(chunk_lengths_list)

        restore = kv_restore_indices.to(device=self.device, dtype=torch.long)
        inv_restore = torch.empty(restore.size(0), device=self.device, dtype=torch.long)
        inv_restore[restore] = torch.arange(
            restore.size(0), device=self.device, dtype=torch.long
        )
        padding_mask_d = padding_mask.to(device=self.device)

        if self.processor_type == CPProcessorType.ROUND_ROBIN:
            self._plan_round_robin(local_tokens, restore, inv_restore, padding_mask_d)
        else:
            self._plan_zigzag(chunk_lengths, local_tokens, kv_restore_indices, inv_restore, padding_mask_d)

        # Build global cu_kv_seqlens from actual (pre-CP-split) input lengths
        actual_input_lengths = cp_info.prefill_actual_input_lengths_cpu
        prefix_lengths = self.attn_inputs.prefix_lengths
        kv_lengths = actual_input_lengths.int() + prefix_lengths.int()
        cu_kv_seqlens_cpu = torch.zeros(kv_lengths.shape[0] + 1, dtype=torch.int32)
        cu_kv_seqlens_cpu[1:] = torch.cumsum(kv_lengths, dim=0)
        self.cu_kv_seqlens_global = cu_kv_seqlens_cpu.to(self.device)

        # get_mla_metadata for flash_mla kernel scheduling
        n_q = self.total_global_ids.size(0)
        if n_q > 0:
            tile_sched, num_splits = get_mla_metadata(  # type: ignore
                cache_seqlens=None,
                num_q_tokens_per_head_k=n_q * self.num_heads,
                topk=self.top_k,
                num_heads_q=self.num_heads,
                num_heads_k=1,
                is_fp8_kvcache=True,
            )
            self._fp8_kernel_metadata_q0 = SparseMlaFp8DecodeParams(
                tile_sched, num_splits
            )
        else:
            self._fp8_kernel_metadata_q0 = None

    def _plan_zigzag(self, chunk_lengths, local_tokens, kv_restore_indices, inv_restore, padding_mask_d):
        """Zigzag index computation: split each request's chunk into q0/q1 halves."""
        q0_idx, q1_idx = generate_q_indices(chunk_lengths)
        kv0_idx, kv1_idx = generate_kv_indices(
            chunk_lengths,
            self.prefill_cp_rank,
            self.prefill_cp_size,
        )

        self.kv0_idx = kv_restore_indices[kv0_idx]
        self.kv1_idx = kv_restore_indices[kv1_idx]
        self.q0_idx = torch.tensor(q0_idx, device=self.device, dtype=torch.long)
        self.q1_idx = torch.tensor(q1_idx, device=self.device, dtype=torch.long)

        source_flat_0 = self.prefill_cp_rank * local_tokens + self.q0_idx
        source_flat_1 = self.prefill_cp_rank * local_tokens + self.q1_idx
        self.q0_idx_global = inv_restore[source_flat_0]
        self.q1_idx_global = inv_restore[source_flat_1]

        # Keep only valid (non-padded) positions
        valid_mask_q0 = padding_mask_d[self.q0_idx_global] == 1
        valid_mask_q1 = padding_mask_d[self.q1_idx_global] == 1
        self.q0_idx_global = self.q0_idx_global[valid_mask_q0]
        self.q1_idx_global = self.q1_idx_global[valid_mask_q1]
        self.q0_idx = self.q0_idx[valid_mask_q0]
        self.q1_idx = self.q1_idx[valid_mask_q1]

        # Convert from padded to unpadded coordinate space
        pad_to_unpad = torch.cumsum(padding_mask_d, dim=0).long() - 1
        self.q0_idx_global = pad_to_unpad[self.q0_idx_global]
        self.q1_idx_global = pad_to_unpad[self.q1_idx_global]

        self.total_global_ids = torch.cat(
            [self.q0_idx_global, self.q1_idx_global], dim=0
        )
        self.total_local_ids = torch.cat([self.q0_idx, self.q1_idx], dim=0)

    def _plan_round_robin(self, local_tokens, restore, inv_restore, padding_mask_d):
        """Round-robin index computation: token i → rank (i % cp_size), no q0/q1 split."""
        # This rank's flat source indices in the all-gathered layout
        source_flat = torch.arange(
            self.prefill_cp_rank * local_tokens,
            (self.prefill_cp_rank + 1) * local_tokens,
            device=self.device,
            dtype=torch.long,
        )
        global_ids_padded = inv_restore[source_flat]

        # Filter out padding tokens
        valid_mask = padding_mask_d[global_ids_padded] == 1
        global_ids_padded_valid = global_ids_padded[valid_mask]
        local_ids_valid = torch.arange(local_tokens, device=self.device, dtype=torch.long)[valid_mask]

        # Convert from padded to unpadded coordinate space
        pad_to_unpad = torch.cumsum(padding_mask_d, dim=0).long() - 1
        global_ids_unpadded = pad_to_unpad[global_ids_padded_valid]

        self.total_local_ids = local_ids_valid
        self.total_global_ids = global_ids_unpadded

        # For indexer compatibility: round-robin has no q0/q1 split
        self.q0_idx = self.total_local_ids
        self.q1_idx = torch.empty(0, device=self.device, dtype=torch.long)
        self.q0_idx_global = self.total_global_ids
        self.q1_idx_global = torch.empty(0, device=self.device, dtype=torch.long)

    def _convert_topk_indices_to_global(
        self, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """CP: topk 行与 total_local_ids 对齐，req_id 需用 total_global_ids 取 batch_indice_d，保证第 i 行对应 global token 的 request id。"""
        if topk_indices.dim() == 2:
            num_tokens, topk = topk_indices.shape
            h_kv = 1
            topk_indices_2d = topk_indices
        else:
            num_tokens, h_kv, topk = topk_indices.shape
            topk_indices_2d = topk_indices[:, 0, :]
        assert topk == self.top_k
        assert self.block_table is not None
        assert self.mla_params is not None
        # req_id[i] = request id for global token total_global_ids[i]
        req_id = self.mla_params.batch_indice_d[self.total_global_ids]
        from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
            triton_convert_req_index_to_global_index,
        )

        global_indices_2d = triton_convert_req_index_to_global_index(
            req_id=req_id,
            block_table=self.block_table,
            token_indices=topk_indices_2d,
            BLOCK_SIZE=self.token_per_block,
            NUM_TOPK_TOKENS=topk,
            BLOCK_N=min(128, topk),
            HAS_PREFILL_WORKSPACE=False,
        )
        global_indices_3d = global_indices_2d.unsqueeze(1).expand(
            num_tokens, h_kv, topk
        )
        return global_indices_3d

    def _convert_topk_indices_to_workspace(
        self, topk_indices: torch.Tensor, workspace_block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Round-robin CP: convert request-local topk indices to global page
        offsets in the temporary FP8 workspace using workspace_block_table.

        This reuses the same triton kernel as _convert_topk_indices_to_global,
        but with the workspace's block_table instead of the real cache's.
        """
        if topk_indices.dim() == 2:
            num_tokens, topk = topk_indices.shape
            h_kv = 1
            topk_indices_2d = topk_indices
        else:
            num_tokens, h_kv, topk = topk_indices.shape
            topk_indices_2d = topk_indices[:, 0, :]
        assert topk == self.top_k
        assert self.mla_params is not None

        req_id = self.mla_params.batch_indice_d[self.total_global_ids]

        from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
            triton_convert_req_index_to_global_index,
        )

        global_indices_2d = triton_convert_req_index_to_global_index(
            req_id=req_id,
            block_table=workspace_block_table,
            token_indices=topk_indices_2d,
            BLOCK_SIZE=self.token_per_block,
            NUM_TOPK_TOKENS=topk,
            BLOCK_N=min(128, topk),
            HAS_PREFILL_WORKSPACE=False,
        )
        global_indices_3d = global_indices_2d.unsqueeze(1).expand(
            num_tokens, h_kv, topk
        )
        return global_indices_3d

    def _build_workspace_fp8(
        self,
        restored_ckv: torch.Tensor,
        restored_k_pe: torch.Tensor,
        kv_cache,
    ):
        """Build a temporary FP8 paged workspace from all-gathered BF16 KV.

        Uses concat_and_cache_mla (the same kernel that writes the real cache)
        to quantize BF16 → FP8 into a temporary contiguous paged tensor.

        NOTE: Currently only supports prefix_length=0. With prefix > 0, we would
        need to all_gather prefix KV from sharded cache and prepend it.

        Returns:
            workspace_fp8: [total_pages, page_size, kv_dim_bytes] tensor (same dtype as real cache)
            workspace_block_table: [batch, max_pages_per_req] int32 tensor
        """
        # Prefix not yet supported: workspace only contains new tokens.
        prefix_lengths = self.attn_inputs.prefix_lengths
        assert prefix_lengths.sum().item() == 0, (
            "kv_cache_sharded (round-robin) with prefix cache is not yet supported. "
            f"Got prefix_lengths sum = {prefix_lengths.sum().item()}"
        )

        page_size = self.token_per_block
        total_kv = restored_ckv.size(0)

        # Determine FP8 kv_dim from the real cache layout
        # kv_cache.kv_cache_base shape: [num_blocks, page_size, kv_dim_bytes]
        kv_dim_bytes = kv_cache.kv_cache_base.size(-1)

        # Allocate temporary FP8 workspace pages
        total_pages = (total_kv + page_size - 1) // page_size
        workspace_fp8 = torch.zeros(
            total_pages, page_size, kv_dim_bytes,
            dtype=kv_cache.kv_cache_base.dtype,
            device=restored_ckv.device,
        )

        # Build contiguous slot_mapping: token i → page (i // page_size), offset (i % page_size)
        slot_mapping = torch.arange(total_kv, dtype=torch.int64, device=restored_ckv.device)

        # Quantize BF16 → FP8 into workspace using the same kernel as real cache write
        scale = torch.tensor(1.0, dtype=torch.float32, device=restored_ckv.device)
        compute_ops.concat_and_cache_mla(
            restored_ckv, restored_k_pe, workspace_fp8, slot_mapping,
            self.kv_cache_write_op.kv_cache_type, scale,
        )

        # Build workspace block_table: each request's pages are contiguous.
        # Since prefix=0, cu_kv_seqlens_global == cumsum(input_lengths), matching
        # the workspace layout where tokens are packed per-request.
        cu_kv = self.cu_kv_seqlens_global.cpu()
        batch_size = cu_kv.size(0) - 1
        max_pages_per_req = 0
        page_starts = []
        page_counts = []
        for i in range(batch_size):
            start_token = cu_kv[i].item()
            end_token = cu_kv[i + 1].item()
            start_page = start_token // page_size
            n_pages = (end_token + page_size - 1) // page_size - start_page
            page_starts.append(start_page)
            page_counts.append(n_pages)
            max_pages_per_req = max(max_pages_per_req, n_pages)

        workspace_block_table = torch.zeros(
            batch_size, max_pages_per_req, dtype=torch.int32, device=restored_ckv.device,
        )
        for i in range(batch_size):
            for j in range(page_counts[i]):
                workspace_block_table[i, j] = page_starts[i] + j

        return workspace_fp8, workspace_block_table

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        topk: Optional[torch.Tensor],
        batch_indice_d: torch.Tensor,
        kv_cache=None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """
        CP prefill forward.

        Both zigzag and round-robin paths:
          1. All-gather KV, restore to global order.
          2. Write to cache (zigzag: full write; round-robin: sharded write with -1 slots).
          3. Run attention via flash_mla_with_kvcache from FP8 cache.
             - zigzag: reads from the real paged cache (full KV written).
             - round-robin: reads from a temporary FP8 workspace (quantized from
               all-gathered BF16), since the real cache is sharded/incomplete.
        """
        # Step 1: All-gather KV and restore to global order (same for both paths)
        gathered_ckv = all_gather(compressed_kv.contiguous(), group=Group.TP)
        gathered_ckv = gathered_ckv.reshape(-1, compressed_kv.size(-1))
        gathered_k_pe = all_gather(k_pe.contiguous(), group=Group.TP)
        gathered_k_pe = gathered_k_pe.reshape(-1, k_pe.size(-1))

        restored_ckv = gathered_ckv[self.kv_restore_unpad_indices]
        restored_k_pe = gathered_k_pe[self.kv_restore_unpad_indices]

        # Step 2: Write cache
        if self.kv_cache_sharded:
            # Round-robin: write only this rank's tokens to sharded cache (for decode).
            # slot_mapping has -1 for non-owned tokens via cp_rank/cp_size in fillParams.
            self.kv_cache_write_op.forward(
                restored_ckv, restored_k_pe, kv_cache, self.mla_params,
            )
        else:
            # Zigzag: write full KV to cache
            self.kv_cache_write_op.forward(
                restored_ckv, restored_k_pe, kv_cache, self.mla_params,
            )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        if topk is None:
            return None

        q0 = q[self.total_local_ids].contiguous()

        # Step 3: Attention
        if self.kv_cache_sharded:
            # Round-robin: real cache is incomplete → build temporary FP8 workspace
            workspace_fp8, workspace_block_table = self._build_workspace_fp8(
                restored_ckv, restored_k_pe, kv_cache,
            )

            # Convert topk indices to workspace page offsets
            workspace_topk = self._convert_topk_indices_to_workspace(
                topk, workspace_block_table,
            )

            # Reshape workspace for flash_mla_with_kvcache: [pages, page_size, 1, kv_dim_bytes]
            kv_cache_flat = workspace_fp8.view(torch.uint8)
            if kv_cache_flat.ndim == 3:
                kv_cache_flat = kv_cache_flat.unsqueeze(-2)

            q_batched = q0.unsqueeze(0)
            if workspace_topk.dim() == 3 and workspace_topk.shape[1] == 1:
                workspace_topk = workspace_topk.squeeze(1)
            indices_batched = workspace_topk.unsqueeze(0)

            part_out, _ = flash_mla_with_kvcache(
                q=q_batched,
                k_cache=kv_cache_flat,
                block_table=workspace_block_table,
                head_dim_v=self.kv_lora_rank,
                cache_seqlens=None,
                tile_scheduler_metadata=self._fp8_kernel_metadata_q0.tile_scheduler_metadata,
                num_splits=self._fp8_kernel_metadata_q0.num_splits,
                is_fp8_kvcache=True,
                indices=indices_batched,
                softmax_scale=self.scale,
            )
            out0 = part_out.squeeze(0)
        else:
            # Zigzag: cache has full KV, use flash_mla_with_kvcache from real paged FP8 cache
            global_topk = self._convert_topk_indices_to_global(topk)

            kv_cache_flat = kv_cache.kv_cache_base.view(
                -1, 1, kv_cache.kv_cache_base.size(-1)
            ).view(torch.uint8)
            if kv_cache_flat.ndim == 3:
                kv_cache_flat = kv_cache_flat.unsqueeze(-2)

            q_batched = q0.unsqueeze(0)
            if global_topk.dim() == 3 and global_topk.shape[1] == 1:
                global_topk = global_topk.squeeze(1)
            indices_batched = global_topk.unsqueeze(0)
            part_out, _ = flash_mla_with_kvcache(
                q=q_batched,
                k_cache=kv_cache_flat,
                block_table=self.block_table,
                head_dim_v=self.kv_lora_rank,
                cache_seqlens=None,
                tile_scheduler_metadata=self._fp8_kernel_metadata_q0.tile_scheduler_metadata,
                num_splits=self._fp8_kernel_metadata_q0.num_splits,
                is_fp8_kvcache=True,
                indices=indices_batched,
                softmax_scale=self.scale,
            )
            out0 = part_out.squeeze(0)

        total_q = q.size(0)
        out = torch.zeros(
            total_q, out0.size(1), out0.size(2), dtype=out0.dtype, device=out0.device
        )
        out[self.total_local_ids] = out0
        return out


class SparseMlaCpImpl(SparseMlaImpl):
    """
    Unified Sparse MLA implementation for both prefill and decode stages.
    Uses the same operator (SparseMlaOp) for both stages with triton-based index conversion.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.cp_info = attn_inputs.context_parallel_info
        attn_inputs.input_lengths = self.cp_info.prefill_actual_input_lengths_cpu
        self._cp_parallelism_config = parallelism_config
        super().__init__(
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            weights=weights,
            cos_sin_cache=cos_sin_cache,
            fmha_config=fmha_config,
            use_trt_fmha=use_trt_fmha,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
            fmha_impl=SparseMlaFp8CPOp,
        )
        self.fmha_impl.kv_cache_write_op = self.kv_cache_write_op
        self.fmha_impl.write_cache_store_impl = self.write_cache_store_impl

    @staticmethod
    def fmha_type() -> FMHAType:
        """Return FMHA type."""
        return FMHAType.CP_SPARSE_FLASHMLA

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create FMHA parameters and pack CP indices into cp_params."""
        self.fmha_params = rtp_llm_ops.SparseMlaParams()
        self.rope_params = self.fmha_params
        self.prepare(attn_inputs)
        # Pack CP indices from fmha_impl for use by indexer and others
        self.cp_params = SimpleNamespace(
            kv_restore_unpad_indices=self.fmha_impl.kv_restore_unpad_indices,
            q0_idx=self.fmha_impl.q0_idx,
            q1_idx=self.fmha_impl.q1_idx,
            q0_idx_global=self.fmha_impl.q0_idx_global,
            q1_idx_global=self.fmha_impl.q1_idx_global,
            total_global_ids=self.fmha_impl.total_global_ids,
            total_local_ids=self.fmha_impl.total_local_ids,
            cu_kv_seqlens_global=self.fmha_impl.cu_kv_seqlens_global,
            kv_cache_sharded=self.fmha_impl.kv_cache_sharded,
        )

    @classmethod
    def support_prefill_cp(cls) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False):
        """Override prepare to pass CP params for kv_cache_sharded mode.

        When kv_cache_sharded=True, slot_mapping has -1 for non-owned tokens.
        This is used by both MLA cache write (sharded) and indexer cache write (sharded).
        Both MLA attention and indexer topk build temporary FP8 workspaces from
        all-gathered data to get complete KV/K for computation.
        """
        assert self.fmha_params is not None

        cp_cfg = self._cp_parallelism_config.prefill_cp_config
        if cp_cfg.kv_cache_sharded and self._cp_parallelism_config.tp_size > 1:
            self.fmha_params.fill_params(
                attn_inputs,
                self.seq_size_per_block,
                forbid_realloc,
                cp_rank=self._cp_parallelism_config.tp_rank,
                cp_size=self._cp_parallelism_config.tp_size,
                kv_cache_sharded=True,
            )
        else:
            self.fmha_params.fill_params(
                attn_inputs, self.seq_size_per_block, forbid_realloc
            )
        self.fmha_impl.plan(self.fmha_params, attn_inputs.kv_cache_block_id_device)

    @classmethod
    def support_parallelism_config(
        cls, parallelism_config: Optional[ParallelismConfig]
    ) -> bool:
        """Support both old CP methods (ALL_GATHER etc.) and PREFILL_CP (sparse MLA CP)."""
        if parallelism_config is None:
            return False
        cp = parallelism_config.prefill_cp_config
        return cp.is_enabled() or cp.is_prefill_enabled()

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass for sparse MLA attention (prefill or decode).

        Args:
            q: Query tensor
                - Prefill: [total_q_len, num_heads, qk_head_dim]
                - Decode: [batch_size, num_heads, qk_head_dim]
            compressed_kv: Compressed KV tensor
                - Prefill: [total_kv_len, kv_lora_rank]
                - Decode: [batch_size, kv_lora_rank] (not used)
            k_pe: Key position encoding
                - Prefill: [total_kv_len, rope_head_dim]
                - Decode: [batch_size, rope_head_dim] (not used)
            kv_cache: KV cache object
            layer_id: Current layer ID
            topk_indices: (topk0, topk1) from indexer CP path, request-local for the two chunks.

        Returns:
            Attention output
                - Prefill: [total_q_len, num_heads, nope_head_dim]
                - Decode: [batch_size, num_heads, nope_head_dim]
        """
        assert self.rope_impl is not None and self.rope_params is not None
        assert kv_cache is not None, "kv_cache is required for sparse MLA"
        assert self.fmha_impl is not None, "fmha_impl is not initialized"

        # Apply RoPE to q_pe and k_pe
        q_pe = q[:, :, self.nope_head_dim :]
        import flashinfer.rope as rope

        if self.fmha_impl.total_local_ids.size(0) > 0:
            q_pe_local = q_pe[self.fmha_impl.total_local_ids]  # element wise
            k_pe_local = k_pe[self.fmha_impl.total_local_ids]  # element wise
            k_rope = k_pe_local.unsqueeze(1)
            pos_ids_q0_global = self.rope_params.positions_d[
                self.fmha_impl.total_global_ids
            ]  # element wise
            rope._apply_rope_pos_ids_cos_sin_cache(
                q=q_pe_local,
                k=k_rope,
                q_rope=q_pe_local,
                k_rope=k_rope,
                cos_sin_cache=self.rope_impl.cos_sin_cache,
                pos_ids=pos_ids_q0_global,
                interleave=not self.rope_impl.is_neox_style,
            )
            k_rope = k_rope.squeeze(1)
            k_pe[self.fmha_impl.total_local_ids] = k_rope  # element wise
            q_pe[self.fmha_impl.total_local_ids] = q_pe_local  # element wise

        # Apply input BMM to transform query
        q_transformed = self._apply_input_bmm(q, layer_id)

        assert self.fmha_params is not None
        attn_output = self.fmha_impl.forward(
            q_transformed,
            compressed_kv,
            k_pe,
            topk_indices,
            self.fmha_params.batch_indice_d,
            kv_cache,
            layer_id=layer_id,
        )

        # Apply output BMM to get final output
        if attn_output is None:
            return None
        output = self._apply_output_bmm(attn_output, layer_id)

        return output
