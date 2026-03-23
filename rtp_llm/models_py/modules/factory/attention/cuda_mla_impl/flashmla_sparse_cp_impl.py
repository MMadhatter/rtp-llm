"""
Unified Sparse MLA implementation for both prefill and decode stages.
Uses flash_mla_sparse_fwd kernel with triton-based index conversion.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

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
from rtp_llm.ops import AttentionConfigs, CPProcessorType, FMHAConfig, FMHAType, ParallelismConfig, compute_ops
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    rtp_llm_ops,
)

from .flashmla_sparse_impl import (
    SparseMlaFp8DecodeParams,
    SparseMlaFp8Op,
    SparseMlaImpl,
)

class ZigZagSparseMlaFp8CPOp(SparseMlaFp8Op):
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
        self.device = torch.cuda.current_device()
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

        # Zig-zag: restore_indices[global_pos] = source_flat_index → global_idx = inv_restore[cp_rank * local_tokens + local_idx]
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
        source_flat_0 = self.prefill_cp_rank * local_tokens + self.q0_idx
        source_flat_1 = self.prefill_cp_rank * local_tokens + self.q1_idx
        self.q0_idx_global = inv_restore[source_flat_0]
        self.q1_idx_global = inv_restore[source_flat_1]

        # Keep only indices where padding_mask is 1 (valid); drop padded positions (0)
        padding_mask_d = padding_mask.to(device=self.device)
        valid_mask_q0 = padding_mask_d[self.q0_idx_global] == 1
        valid_mask_q1 = padding_mask_d[self.q1_idx_global] == 1
        self.q0_idx_global = self.q0_idx_global[valid_mask_q0]
        self.q1_idx_global = self.q1_idx_global[valid_mask_q1]
        self.q0_idx = self.q0_idx[valid_mask_q0]
        self.q1_idx = self.q1_idx[valid_mask_q1]

        # Convert from padded to unpadded coordinate space.
        # inv_restore yields indices in the padded global space (0..padded_total-1),
        # but positions_d / ks / ke / batch_indice_d are sized by unpadded_total.
        pad_to_unpad = torch.cumsum(padding_mask_d, dim=0).long() - 1
        self.q0_idx_global = pad_to_unpad[self.q0_idx_global]
        self.q1_idx_global = pad_to_unpad[self.q1_idx_global]

        self.total_global_ids = torch.cat(
            [self.q0_idx_global, self.q1_idx_global], dim=0
        )
        self.total_local_ids = torch.cat([self.q0_idx, self.q1_idx], dim=0)

        # --- Bounds checks (moved from forward hot-path to avoid device-host sync) ---
        unpadded_total = int(padding_mask.sum().item())
        if self.total_local_ids.numel() > 0:
            max_lid = self.total_local_ids.max().item()
            if max_lid >= local_tokens:
                raise ValueError(
                    f"[plan] total_local_ids out of range: "
                    f"max(total_local_ids)={max_lid}, local_tokens={local_tokens}. "
                    "Check CP plan() local chunk vs actual input size."
                )
        if self.total_global_ids.numel() > 0:
            max_gid = self.total_global_ids.max().item()
            if max_gid >= unpadded_total:
                raise ValueError(
                    f"[plan] total_global_ids out of range: "
                    f"max(total_global_ids)={max_gid}, unpadded_total={unpadded_total}. "
                    "Check padded-to-unpadded coordinate conversion."
                )

        # attention_inputs.cu_kv_seqlens is based on local CP chunk lengths
        # (input_lengths is overwritten by ContextParallelProcessor), but the
        # gather kernel needs cumulative lengths covering the full (global) sequence.
        actual_input_lengths = cp_info.prefill_actual_input_lengths_cpu
        prefix_lengths = self.attn_inputs.prefix_lengths
        kv_lengths = actual_input_lengths.int() + prefix_lengths.int()
        cu_kv_seqlens_cpu = torch.zeros(kv_lengths.shape[0] + 1, dtype=torch.int32)
        cu_kv_seqlens_cpu[1:] = torch.cumsum(kv_lengths, dim=0)
        self.total_kv_len = int(cu_kv_seqlens_cpu[-1])
        self.cu_kv_seqlens_global = cu_kv_seqlens_cpu.to(self.device)

        # get_mla_metadata: num_q_tokens_per_head_k = num_q_tokens * num_heads_q // num_heads_k (for tile scheduling).
        # For q0 and q1 we need separate metadata since each part has different q token count (use filtered counts).
        n_q = self.total_global_ids.size(0)
        tile_sched_q0, num_splits_q0 = get_mla_metadata(
            cache_seqlens=None,
            num_q_tokens_per_head_k=n_q * self.num_heads,
            topk=self.top_k,
            num_heads_q=self.num_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        self._fp8_kernel_metadata_q0 = SparseMlaFp8DecodeParams(
            tile_sched_q0, num_splits_q0
        )

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
        CP prefill forward: all-gather KV, restore, write to kv_cache, then two-part attention.

        Args:
            q: [total_q_len, num_heads, qk_head_dim], already RoPE-applied and input-BMM applied (q_transformed).
            compressed_kv: [total_kv_len, kv_lora_rank], local.
            k_pe: [total_kv_len, rope_head_dim], local.
            topk0: [len(q0_idx), topk] or [len(q0_idx), num_heads, topk], request-local for first CP chunk.
            topk1: [len(q1_idx), topk] or [len(q1_idx), num_heads, topk], request-local for second CP chunk.
            batch_indice_d: [total_q_len], int32, request id per token.
            kv_cache: KV cache to write restored KV into (same paged layout as non-CP).
            layer_id: layer id.

        Returns:
            attn_output: [total_q_len, num_heads, kv_lora_rank], same as non-CP SparseMlaOp.
        """
        # All-gather KV across CP ranks, restore to global order, then write full KV to cache
        gathered_ckv = all_gather(compressed_kv.contiguous(), group=Group.TP)
        gathered_ckv = gathered_ckv.reshape(-1, compressed_kv.size(-1))
        gathered_k_pe = all_gather(k_pe.contiguous(), group=Group.TP)
        gathered_k_pe = gathered_k_pe.reshape(-1, k_pe.size(-1))

        restored_ckv = gathered_ckv[self.kv_restore_unpad_indices]
        restored_k_pe = gathered_k_pe[self.kv_restore_unpad_indices]

        self.kv_cache_write_op.forward(
            restored_ckv, restored_k_pe, kv_cache, self.mla_params,
        )

        # TODO: write cache for each rank
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        if topk is None:
            return None

        # Convert request-local topk0/topk1 to global indices for flash_mla_with_kvcache
        global_topk = self._convert_topk_indices_to_global(topk)

        kv_cache_flat = kv_cache.kv_cache_base.view(
            -1, 1, kv_cache.kv_cache_base.size(-1)
        ).view(torch.uint8)
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)
        
        if layer_id == 0:
            meta = self._fp8_kernel_metadata_q0
            if meta is not None:
                metadata = meta.tile_scheduler_metadata
                if metadata is not None:
                    metadata.tile_scheduler_metadata = None
                    metadata.num_splits = None

        q0 = q[self.total_local_ids].contiguous()

        def run_part(
            q_part: torch.Tensor,
            global_topk: torch.Tensor,
            fp8_kernel_metadata: SparseMlaFp8DecodeParams,
        ) -> torch.Tensor:
            q_batched = q_part.unsqueeze(0)
            if global_topk.dim() == 3 and global_topk.shape[1] == 1:
                global_topk = global_topk.squeeze(1)
            indices_batched = global_topk.unsqueeze(0)
            part_out, _ = flash_mla_with_kvcache(
                q=q_batched,
                k_cache=kv_cache_flat,
                block_table=self.block_table,
                head_dim_v=self.kv_lora_rank,
                cache_seqlens=None,
                tile_scheduler_metadata=fp8_kernel_metadata.tile_scheduler_metadata,  # type: ignore
                num_splits=fp8_kernel_metadata.num_splits,  # type: ignore
                is_fp8_kvcache=True,
                indices=indices_batched,
                softmax_scale=self.scale,
            )
            return part_out.squeeze(0)

        out0 = run_part(q0, global_topk, self._fp8_kernel_metadata_q0)

        total_q = q.size(0)
        out = torch.zeros(
            total_q, out0.size(1), out0.size(2), dtype=out0.dtype, device=out0.device
        )
        out[self.total_local_ids] = out0
        return out


# ---------------------------------------------------------------------------
# RoundRobin CP Op
# ---------------------------------------------------------------------------
class RoundRobinSparseMlaFp8CPOp(SparseMlaFp8Op):
    """Round-robin CP: token i → rank (i % cp_size), sharded cache + FP8 workspace for attention."""

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
        assert self.cp_info is not None, "Context parallel info is required"

        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size
        self.kv_cache_sharded = parallelism_config.prefill_cp_config.kv_cache_sharded
        self.device = torch.cuda.current_device()

        self.kv_restore_unpad_indices = None
        # Round-robin has no q0/q1 split, but keep attrs for indexer compatibility
        self.q0_idx = None
        self.q1_idx = None
        self.q0_idx_global = None
        self.q1_idx_global = None
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

        padding_lengths = cp_info.prefill_cp_padding_lengths
        if hasattr(padding_lengths, "cpu"):
            padding_lengths_list = padding_lengths.cpu().tolist()
        else:
            padding_lengths_list = list(padding_lengths)

        R = self.prefill_cp_rank
        C = self.prefill_cp_size
        local_tokens = sum(chunk_lengths_list)

        # Round-robin: rank R owns tokens at local positions [0, chunk_len) per stream.
        # Their global (unpadded) positions are R + j*C within each stream.
        # Pad tokens sit at the tail of each stream and don't affect causal attention,
        # so we include all local tokens and clamp pad positions to the last valid token.
        _empty = torch.empty(0, device=self.device, dtype=torch.long)
        self.total_local_ids = torch.arange(local_tokens, device=self.device, dtype=torch.long)

        # Compute global (unpadded) position for each local token.
        # actual_seq_len per stream = chunk_length * cp_size - padding_length
        global_pieces: list[torch.Tensor] = []
        unpadded_offset = 0
        for s, cl in enumerate(chunk_lengths_list):
            actual_seq_len = cl * C - int(padding_lengths_list[s])
            js = torch.arange(cl, device=self.device, dtype=torch.long)
            global_pos = js * C + R + unpadded_offset
            # Clamp pad tokens to last valid position (they won't affect causal attention)
            global_pos = torch.clamp(global_pos, max=unpadded_offset + actual_seq_len - 1)
            global_pieces.append(global_pos)
            unpadded_offset += actual_seq_len

        self.total_global_ids = torch.cat(global_pieces) if global_pieces else _empty

        # No q0/q1 split for round-robin; set for indexer compatibility
        self.q0_idx = self.total_local_ids
        self.q1_idx = _empty
        self.q0_idx_global = self.total_global_ids
        self.q1_idx_global = _empty

        # Build global cu_kv_seqlens
        actual_input_lengths = cp_info.prefill_actual_input_lengths_cpu
        prefix_lengths = self.attn_inputs.prefix_lengths
        kv_lengths = actual_input_lengths.int() + prefix_lengths.int()
        cu_kv_seqlens_cpu = torch.zeros(kv_lengths.shape[0] + 1, dtype=torch.int32)
        cu_kv_seqlens_cpu[1:] = torch.cumsum(kv_lengths, dim=0)
        self.cu_kv_seqlens_global = cu_kv_seqlens_cpu.to(self.device)

        # Pre-compute workspace metadata
        self._ws_total_kv = self.kv_restore_unpad_indices.size(0)
        page_size = self.token_per_block
        self._ws_total_pages = (self._ws_total_kv + page_size - 1) // page_size
        self._ws_slot_mapping = torch.arange(
            self._ws_total_kv, dtype=torch.int64, device=self.device
        )

        # Pre-compute workspace_block_table from cu_kv_seqlens_global
        assert prefix_lengths.sum().item() == 0, (
            "kv_cache_sharded (round-robin) with prefix cache is not yet supported. "
            f"Got prefix_lengths sum = {prefix_lengths.sum().item()}"
        )
        cu_kv = self.cu_kv_seqlens_global.cpu()
        batch_size = cu_kv.size(0) - 1
        if batch_size > 0:
            starts = cu_kv[:-1]
            ends = cu_kv[1:]
            start_pages = starts // page_size
            end_pages = (ends + page_size - 1) // page_size
            pages_per_req = (end_pages - start_pages).int()
            max_pages = int(pages_per_req.max().item())
            col_idx = torch.arange(max_pages, dtype=torch.int32)
            self._ws_block_table = (
                start_pages.unsqueeze(1).int() + col_idx.unsqueeze(0)
            ).to(self.device)
            mask = col_idx.unsqueeze(0) >= pages_per_req.unsqueeze(1)
            self._ws_block_table[mask] = 0
        else:
            self._ws_block_table = torch.zeros(0, 0, dtype=torch.int32, device=self.device)

        # Workspace FP8 tensor allocated lazily on first forward
        self._ws_fp8 = None

        # get_mla_metadata for flash_mla kernel scheduling
        n_q = local_tokens
        if n_q > 0:
            tile_sched, num_splits = get_mla_metadata(
                cache_seqlens=None,
                num_q_tokens_per_head_k=n_q * self.num_heads,
                topk=self.top_k,
                num_heads_q=self.num_heads,
                num_heads_k=1,
                is_fp8_kvcache=True,
            )
            self._fp8_kernel_metadata_q0 = SparseMlaFp8DecodeParams(tile_sched, num_splits)
        else:
            self._fp8_kernel_metadata_q0 = None


    def _convert_topk_indices_to_workspace(
        self, topk_indices: torch.Tensor, workspace_block_table: torch.Tensor,
    ) -> torch.Tensor:
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
        return global_indices_2d.unsqueeze(1).expand(num_tokens, h_kv, topk)

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
        # Step 1: All-gather KV and restore to global order
        gathered_ckv = all_gather(compressed_kv.contiguous(), group=Group.TP)
        gathered_ckv = gathered_ckv.reshape(-1, compressed_kv.size(-1))
        gathered_k_pe = all_gather(k_pe.contiguous(), group=Group.TP)
        gathered_k_pe = gathered_k_pe.reshape(-1, k_pe.size(-1))

        restored_ckv = gathered_ckv[self.kv_restore_unpad_indices]
        restored_k_pe = gathered_k_pe[self.kv_restore_unpad_indices]

        # Step 2: Write sharded cache (slot_mapping has -1 for non-owned tokens)
        self.kv_cache_write_op.forward(
            restored_ckv, restored_k_pe, kv_cache, self.mla_params,
        )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        if topk is None:
            return None

        # Step 3: Build temporary FP8 workspace from all-gathered KV
        if self._ws_fp8 is None:
            kv_dim_bytes = kv_cache.kv_cache_base.size(-1)
            self._ws_fp8 = torch.zeros(
                self._ws_total_pages, self.token_per_block, kv_dim_bytes,
                dtype=kv_cache.kv_cache_base.dtype,
                device=self.device,
            )
        else:
            self._ws_fp8.zero_()

        scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        compute_ops.concat_and_cache_mla(
            restored_ckv, restored_k_pe, self._ws_fp8, self._ws_slot_mapping,
            self.kv_cache_write_op.kv_cache_type, scale,
        )

        # Step 4: Attention — all local q tokens attend to full global KV workspace
        workspace_topk = self._convert_topk_indices_to_workspace(
            topk, self._ws_block_table,
        )

        kv_cache_flat = self._ws_fp8.view(torch.uint8)
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)

        q_batched = q.unsqueeze(0)
        if workspace_topk.dim() == 3 and workspace_topk.shape[1] == 1:
            workspace_topk = workspace_topk.squeeze(1)
        indices_batched = workspace_topk.unsqueeze(0)

        attn_out, _ = flash_mla_with_kvcache(
            q=q_batched,
            k_cache=kv_cache_flat,
            block_table=self._ws_block_table,
            head_dim_v=self.kv_lora_rank,
            cache_seqlens=None,
            tile_scheduler_metadata=self._fp8_kernel_metadata_q0.tile_scheduler_metadata,
            num_splits=self._fp8_kernel_metadata_q0.num_splits,
            is_fp8_kvcache=True,
            indices=indices_batched,
            softmax_scale=self.scale,
        )
        return attn_out.squeeze(0)


class SparseMlaCpImpl(SparseMlaImpl):
    """Sparse MLA CP implementation. Selects ZigZag or RoundRobin Op based on processor_type."""

    _OP_MAP = {
        CPProcessorType.ZIG_ZAG: ZigZagSparseMlaFp8CPOp,
        CPProcessorType.ROUND_ROBIN: RoundRobinSparseMlaFp8CPOp,
    }

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
        # Restore input_lengths to global (pre-CP-chunking) lengths.
        # handleInputs() overwrites input_lengths to local chunk sizes, but we
        # need global lengths for:
        #   - fill_params: computes positions 0..seq_len-1 and slot_mapping
        #     (kv_cache_sharded mode sets slot=-1 for non-owned tokens)
        #   - WriteCacheStoreOp (PD separation): C++ writeCacheStore handles
        #     sharding via cp_slot_mapper->localBlockCount(global_len)
        # Pad tokens (from round-robin when seq_len % cp_size != 0) are NOT
        # included here — prefill_actual_input_lengths_cpu is the original
        # unpadded length, and kv_restore_unpad_indices excludes pad tokens
        # from the all-gathered KV before cache write.
        attn_inputs.input_lengths = self.cp_info.prefill_actual_input_lengths_cpu
        self._cp_parallelism_config = parallelism_config

        processor_type = parallelism_config.prefill_cp_config.processor_type
        op_cls = self._OP_MAP.get(processor_type)
        assert op_cls is not None, (
            f"Unsupported CP processor_type: {processor_type}. "
            f"Must be one of {list(self._OP_MAP.keys())}"
        )

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
            fmha_impl=op_cls,
        )
        self.fmha_impl.kv_cache_write_op = self.kv_cache_write_op
        self.fmha_impl.write_cache_store_impl = self.write_cache_store_impl

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.CP_SPARSE_FLASHMLA

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create FMHA parameters and pack CP indices into cp_params."""      
        self.fmha_params = rtp_llm_ops.SparseMlaParams()
        self.rope_params = self.fmha_params
        self.prepare(attn_inputs)
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

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert self.rope_impl is not None and self.rope_params is not None
        assert kv_cache is not None, "kv_cache is required for sparse MLA"
        assert self.fmha_impl is not None, "fmha_impl is not initialized"

        # Apply RoPE to q_pe and k_pe
        q_pe = q[:, :, self.nope_head_dim :]
        import flashinfer.rope as rope

        if self.fmha_impl.total_local_ids.size(0) > 0:
            q_pe_local = q_pe[self.fmha_impl.total_local_ids]
            k_pe_local = k_pe[self.fmha_impl.total_local_ids]
            k_rope = k_pe_local.unsqueeze(1)
            pos_ids_q0_global = self.rope_params.positions_d[
                self.fmha_impl.total_global_ids
            ]
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
            k_pe[self.fmha_impl.total_local_ids] = k_rope
            q_pe[self.fmha_impl.total_local_ids] = q_pe_local

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

        if attn_output is None:
            return None
        return self._apply_output_bmm(attn_output, layer_id)
