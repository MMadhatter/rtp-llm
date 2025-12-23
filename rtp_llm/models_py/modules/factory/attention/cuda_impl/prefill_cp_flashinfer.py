import logging
from enum import Enum, auto
from typing import Any, Dict, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, all_gather, recv, send
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAPrefillImplBase,
)
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOp,
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    PyContextParallelParams,
)

logger = logging.getLogger(__name__)

# Global workspace buffer shared across all wrappers
_g_workspace_buffer = None
_g_workspace_size = 512 * 1024 * 1024  # 512MB


class CPRotateMethod(Enum):
    """Context Parallel rotation method for attention computation."""

    ALL_GATHER = auto()  # Use all_gather with zig-zag load balancing
    ALL_GATHER_WITH_OVERLAP = auto()  # Use all_gather with overlap
    RING = auto()  # Use ring attention with zig-zag load balancing


def get_workspace_buffer(device: torch.device) -> torch.Tensor:
    """Get or create global workspace buffer for FlashInfer."""
    global _g_workspace_buffer
    if _g_workspace_buffer is None:
        _g_workspace_buffer = torch.empty(
            _g_workspace_size,
            dtype=torch.uint8,
            device=device,
        )
    return _g_workspace_buffer


from flashinfer import BatchPrefillWithRaggedKVCacheWrapper


class ContextParallelFlashInferRaggedPrefillOp:
    """
    FlashInfer Ragged KV Cache Prefill Attention Operator for standard MHA.

    This implementation uses BatchPrefillWithRaggedKVCacheWrapper which is
    optimized for variable-length sequences without paging.
    """

    def __init__(
        self,
        config: GptInitModelParameters,
        backend: str = "auto",  # "auto", "fa2", or "fa3"
        causal: bool = True,
        kv_layout: str = "NHD",  # "NHD" or "HND"
    ):
        """
        Initialize FlashInfer Ragged Prefill Operator.

        Args:
            config: Model configuration
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (for GQA/MQA)
            head_dim: Dimension of each head
            backend: FlashInfer backend ("auto", "fa2", or "fa3")
            causal: Whether to use causal masking
            kv_layout: KV cache layout ("NHD" or "HND")
        """
        super().__init__()
        self.config = config
        self.num_qo_heads = config.head_num
        self.num_kv_heads = config.head_num_kv
        self.head_dim = config.size_per_head
        self.backend = backend
        self.causal = causal
        self.kv_layout = kv_layout

        self.device = torch.cuda.current_device()
        self.workspace_buffer = get_workspace_buffer(self.device)
        self.cp_size = config.cp_size
        self.cp_rank = config.cp_rank

        self.rotate_method = CPRotateMethod.ALL_GATHER
        self.prefill_wrappers = {}

        if self.rotate_method == CPRotateMethod.ALL_GATHER:
            # Zig-zag load balancing: when using all_gather, each CP rank's workload
            # is split into two causal attention computations for better load distribution.
            # Part0: wrapper for first portion of the computation
            # Part1: wrapper for second portion of the computation
            self.prefill_wrappers["part0"] = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout=kv_layout,
                backend=backend,
            )
            self.prefill_wrappers["part1"] = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout=kv_layout,
                backend=backend,
            )
        else:
            # ring attention impl: Multi round send/recv for prefill
            self.prefill_wrappers["casual"] = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout=kv_layout,
                backend=backend,
            )
            self.prefill_wrappers["non_casual"] = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout=kv_layout,
                backend=backend,
            )

        self._is_warmed_up = False

    def support(self, attention_inputs: PyAttentionInputs) -> bool:
        """Check if this operator supports the given attention inputs."""
        return attention_inputs.is_prefill and self.config.cp_size > 1

    def _prepare_cp_data(
        self,
        cp_chunk_lengths: torch.Tensor,
        cp_shuffle_indices: torch.Tensor,
        cp_padding_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # return q_lens, kv_lens之类的东西
        return

    def prepare(self, attention_inputs: PyAttentionInputs) -> ParamsBase:
        """
        Prepare context parallel attention computation with zig-zag load balancing.
        Zig-zag Attention Partitioning for Causal Attention:
        For a sequence of length N distributed across cp_size ranks:
        1. Tokens are split using zig-zag shuffle: alternating chunks from start/end
           Example (cp_size=4, N=16, chunk=2): [0,1, 14,15, 2,3, 12,13, 4,5, 10,11, 6,7, 8,9]
                                                 └─┘  └──┘  └─┘  └──┘  └─┘  └──┘  └─┘  └─┘
                                           rank0(r0)   r0  r1   r1    r2   r2    r3   r3
        2. Each rank holds a subset of Q and KV:
           - Rank i has Q_i (queries for its token chunk)
           - Rank i has KV_i (keys/values for its token chunk)

        3. Causal Attention Matrix (Q attends to KV where token_j <= token_i):

                      KV0  KV1  KV2  KV3  KV4  KV5  KV6  KV7  KV8  KV9  KV10 KV11 KV12 KV13 KV14 KV15
           Q0   (r0) [ C   ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ]
           Q1   (r0) [ C   C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ] 2
           Q2   (r1) [ C   C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ]
           Q3   (r1) [ C   C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ] 4
           Q4   (r2) [ C   C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ]
           Q5   (r2) [ C   C    C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ] 6
           Q6   (r3) [ C   C    C    C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ]
           Q7   (r3) [ C   C    C    C    C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ]  8
           Q8   (r3) [ C   C    C    C    C    C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗    ✗   ]
           Q9   (r3) [ C   C    C    C    C    C    C    C    C    C    ✗    ✗    ✗    ✗    ✗    ✗   ] 10
           Q10  (r2) [ C   C    C    C    C    C    C    C    C    C    C    ✗    ✗    ✗    ✗    ✗   ]
           Q11  (r2) [ C   C    C    C    C    C    C    C    C    C    C    C    ✗    ✗    ✗    ✗   ] 12
           Q12  (r1) [ C   C    C    C    C    C    C    C    C    C    C    C    C    ✗    ✗    ✗   ]
           Q13  (r1) [ C   C    C    C    C    C    C    C    C    C    C    C    C    C    ✗    ✗   ] 14
           Q14  (r0) [ C   C    C    C    C    C    C    C    C    C    C    C    C    C    C    ✗   ]
           Q15  (r0) [ C   C    C    C    C    C    C    C    C    C    C    C    C    C    C    C   ] 16
        4.
        - all gather without overlap:
           rank_i_part_0: q_len=chunk_size, kv_len=chunk_size*(rank_id + 1)
           rank_i_part_1: q_len=chunk_size, kv_len=chunk_size*(2 * cp_size - rank_id)
        """
        # Get batch information
        batch_size = attention_inputs.input_lengths.size(0)
        device = attention_inputs.input_lengths.device
        cu_seqlens = attention_inputs.cu_seqlens[
            : attention_inputs.input_lengths.size(0) + 1
        ]
        cp_info = attention_inputs.context_parallel_info
        prefill_cp_chunk_lengths = cp_info.prefill_cp_chunk_lengths
        prefill_shuffle_indices = cp_info.prefill_shuffle_indices
        cp_padding_lengths = cp_info.prefill_cp_padding_lengths

        if self.rotate_method == CPRotateMethod.ALL_GATHER:
            # Plan for both part0 and part1 wrappers
            qo_indptr = cu_seqlens // 2
            kv_indptr_part0 = qo_indptr * (self.cp_rank + 1)
            kv_indptr_part1 = qo_indptr * (2 * self.cp_size - self.cp_rank)
            # Part0: First part of attention computation
            part_0_kv_indptr = cu_seqlens_wo_padding[:-1]
            part_1_kv_indptr = cu_seqlens_wo_padding[:-1]
            self.prefill_wrappers["part0"].plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_part0,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=None,  # Not used in ragged mode
                causal=True,  # Part0 uses causal attention
                q_data_type=attention_inputs.dtype,
            )
            # Part1: Second part of attention computation
            self.prefill_wrappers["part1"].plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_part1,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=None,  # Not used in ragged mode
                causal=True,
                q_data_type=attention_inputs.dtype,
            )
        elif self.rotate_method == CPRotateMethod.RING:
            # Use ring attention with multi-round send/recv
            qo_indptr = cu_seqlens // 2
            self.prefill_wrappers["causal"].plan(
                qo_indptr=cu_seqlens_wo_padding,
                kv_indptr=cu_seqlens_wo_padding,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=None,  # Not used in ragged mode
                causal=True,
                q_data_type=attention_inputs.dtype,
            )

            self.prefill_wrappers["non_causal"].plan(
                qo_indptr=cu_seqlens_wo_padding,
                kv_indptr=cu_seqlens_wo_padding,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=None,  # Not used in ragged mode
                causal=False,
                q_data_type=attention_inputs.dtype,
            )
        elif self.rotate_method == CPRotateMethod.ALL_GATHER_WITH_OVERLAP:
            # 拆成casual和两个非casual
            self.prefill_wrappers["casual"] = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout=kv_layout,
                backend=backend,
            )
            self.prefill_wrappers["non_casual_1"] = (
                BatchPrefillWithRaggedKVCacheWrapper(
                    self.workspace_buffer,
                    kv_layout=kv_layout,
                    backend=backend,
                )
            )
            self.prefill_wrappers["non_casual_2"] = (
                BatchPrefillWithRaggedKVCacheWrapper(
                    self.workspace_buffer,
                    kv_layout=kv_layout,
                    backend=backend,
                )
            )

        return ParamsBase()

    def forward_all_gather(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context_parallel_info: PyContextParallelParams,
    ) -> torch.Tensor:
        # all gather key and value across all CP ranks
        key = all_gather(k, group=Group.CP)
        value = all_gather(v, group=Group.CP)
        # reshuffle

        # Split attention computation into two parts
        # Part0: First part of computation with causal attention
        attn_output_part0 = self.prefill_wrappers["part0"].run(q, k, v)

        # Part1: Second part of computation with non-causal attention
        attn_output_part1 = self.prefill_wrappers["part1"].run(q, k, v)

        # Combine results from both parts
        attn_output = attn_output_part0 + attn_output_part1

    def forward_all_to_all(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context_parallel_info: PyContextParallelParams,
    ) -> torch.Tensor:
        # Ring attention with multi-round send/recv
        for i in range(self.cp_size):
            if i == self.cp_rank:
                attn_output = self.prefill_wrappers["causal"].run(q, k, v)
            else:
                send(k[i], dst=i, group=Group.CP)
                send(v[i], dst=i, group=Group.CP)
                attn_output = recv(k[i], src=i, group=Group.CP)
                attn_output = recv(v[i], src=i, group=Group.CP)
                attn_output = self.prefill_wrappers["non_causal"].run(q, k, v)
                attn_output = attn_output + attn_output

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> torch.Tensor:
        # reshape qkv to q, k, v
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.num_qo_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )
        q = q.reshape(q.shape[0], self.num_qo_heads, self.head_dim)
        k = k.reshape(k.shape[0], self.num_kv_heads, self.head_dim)
        v = v.reshape(v.shape[0], self.num_kv_heads, self.head_dim)

        if self.rotate_method == CPRotateMethod.ALL_GATHER:
            attn_output = self.forward_all_gather(q, k, v)
        elif self.rotate_method == _CPRotateMethod.RING:
            attn_output = self.forward_all_to_all(q, k, v)
        return attn_output


class PrefillContextParallelFlashInferImpl(FMHAPrefillImplBase):
    def __init__(
        self,
        config: GptInitModelParameters,
        attn_inputs: PyAttentionInputs,
    ):
        super().__init__(
            fmha_impl=ContextParallelFlashInferRaggedPrefillOp(config.gpt_init_params),
            rope_kvcache_impl=FusedRopeKVCachePrefillOp(config.gpt_init_params),
            attn_inputs=attn_inputs,
        )

    def support(self) -> bool:
        """Check if this implementation supports current inputs."""
        return self.fmha_impl.support(self.attn_inputs)

    def fmha_type(self) -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return False
