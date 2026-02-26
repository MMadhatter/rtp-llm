"""
- 将序列分成 dilation 个组, 每组内的 tokens 访问模式完全对齐标准卷积的滚动更新
"""

from typing import Optional, Union

import torch
import triton
import triton.language as tl

BLOCK_M = 128  # tokens per block
BLOCK_N = 64  # features per block


@triton.jit
def _causal_conv1d_fwd_kernel_dilated(
    # Pointers to matrices
    x_ptr,
    w_ptr,
    bias_ptr,
    initial_states_ptr,
    block_map_ptr,
    prefix_lengths_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    group_id_ptr,  # dilation group
    o_ptr,
    # Matrix dimensions
    batch: tl.int32,
    dim: tl.constexpr,
    max_block_size: tl.int32,
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    dilation: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQ_SIZE_PER_BLOCK: tl.constexpr,
):
    """
    dilated causal conv1d kernel
    - 将序列分成 dilation 个组（group_id = 0, 1, ..., dilation-1）
    - 每个 kernel instance 处理一个dilation group的连续 tokens
    - 组内 tokens 按 dilation 步长排列
    例如 dilation=2:
    - group 0: token [0, 2, 4, 6, ...]
    - group 1: token [1, 3, 5, 7, ...]
    """

    conv_states_ptr = initial_states_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (KERNEL_WIDTH - 1) * dilation

    # 获取当前处理的序列和组信息
    idx_seq = tl.load(batch_ptr + tl.program_id(0))
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0)).to(tl.int64)
    group_id = tl.load(group_id_ptr + tl.program_id(0)).to(tl.int32)  # 0 到 dilation-1
    prefix_length = tl.load(prefix_lengths_ptr + idx_seq).to(tl.int32)

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1).to(tl.int64)
    seqlen = sequence_end_index - sequence_start_index

    # 计算当前组内的 token offset
    # group_id 决定起始位置，chunk_offset 是组内的 chunk 索引
    token_offset = group_id + chunk_offset * BLOCK_M * dilation

    # 计算这个 chunk 需要处理多少个 token（组内按 dilation 步长）
    remaining_tokens = (seqlen - token_offset + dilation - 1) // dilation
    segment_len = min(BLOCK_M, remaining_tokens)

    if token_offset >= seqlen:
        return

    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    w_base = w_ptr + (idx_feats * stride_w_dim)

    # 历史位置按 dilation 步长计算
    if chunk_offset == 0:
        # 第一个 chunk：从 conv_states 读取历史
        if HAS_CACHE and prefix_length > 0:
            init_state_block_pos = (prefix_length - 1) // SEQ_SIZE_PER_BLOCK
            init_state_block_idx = tl.load(
                block_map_ptr + idx_seq * max_block_size + init_state_block_pos
            ).to(tl.int64)
            conv_states_base = (
                conv_states_ptr
                + (init_state_block_idx * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )
            # 加载最后 state_len 个 tokens
            # 对于 dilated convolution，需要根据 group_id 调整起始位置, group_id 决定了当前组在 dilation 组中的位置
            prior_tokens = (
                conv_states_base
                + (state_len - dilation + group_id) * stride_conv_state_tok
            )
            mask_w = idx_feats < dim

            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - dilation * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - dilation * stride_conv_state_tok
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * dilation * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - dilation * stride_conv_state_tok
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * dilation * stride_conv_state_tok
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * dilation * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # 没有历史，初始化为 0
            mask_w = idx_feats < dim
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    else:
        # 非第一个 chunk：从 x 读取历史（按 dilation 步长）
        prior_tokens = x_base + (token_offset - dilation) * stride_x_token
        mask_w = idx_feats < dim

        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - dilation * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - dilation * stride_x_token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * dilation * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            conv_states_ptrs = prior_tokens
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - dilation * stride_x_token
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * dilation * stride_x_token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * dilation * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    # 预加载 bias
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token

    # 预加载权重
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    mask_x_1d = idx_feats < dim

    # 主循环：处理组内的 tokens（按 dilation 步长前进）
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * dilation * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * dilation * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * dilation * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w

        # 滚动更新历史
        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        # 激活函数
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))

        # 当前 token 在序列中的实际位置
        curr_token_pos = token_offset + idx_token * dilation

        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = (
            o_ptr
            + (sequence_start_index + curr_token_pos) * stride_o_token
            + (idx_feats * stride_o_dim)
        )
        tl.store(o_ptrs, acc, mask=mask_1d)

        # 写回 conv_states
        dest_idx = prefix_length + curr_token_pos
        # 写入条件：
        # case0. 序列到达SEQ_SIZE_PER_BLOCK的位置了，在block_size边界以及往后 dilation-1 个位置
        # case1. 达到当前seq_len长度边界（需要写入以便下次decoding的时候恢复状态）
        # 因为一个组内的 tokens 跨度为 (kernel_width-1)*dilation
        # 对于 dilation > 1，需要检查 dest_idx 到 dest_idx+dilation-1 范围是否跨越 block 边界
        # state_len = (kernel_size -1) * dilation, 当kernel_size=4, dilation=3, state_len = 9,
        # 也就是写[group0_t0, group1_t0, group2_t0, group0_t1, group1_t1, group2_t1, group0_t2,  group1_t2, group2_t2]

        write_to_block = HAS_CACHE and (
            (dest_idx % SEQ_SIZE_PER_BLOCK) + dilation >= SEQ_SIZE_PER_BLOCK
            or dest_idx + dilation >= seqlen
        )
        if write_to_block:
            # case 1:
            if (dest_idx % SEQ_SIZE_PER_BLOCK) + dilation >= SEQ_SIZE_PER_BLOCK and (
                seqlen + prefix_length
            ) >= (dest_idx // SEQ_SIZE_PER_BLOCK + 1) * SEQ_SIZE_PER_BLOCK:
                block_token_offset = (dest_idx + dilation) % SEQ_SIZE_PER_BLOCK
                write_page_idx = tl.load(
                    block_map_ptr
                    + idx_seq * max_block_size
                    + dest_idx // SEQ_SIZE_PER_BLOCK
                )
                base_ptr = (
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                )
                tl.store(
                    base_ptr + block_token_offset * stride_conv_state_tok, col0, mask_1d
                )
                if KERNEL_WIDTH >= 3:
                    tl.store(
                        base_ptr
                        + (block_token_offset + dilation) * stride_conv_state_tok,
                        col1,
                        mask_1d,
                    )
                if KERNEL_WIDTH >= 4:
                    tl.store(
                        base_ptr
                        + (block_token_offset + 2 * dilation) * stride_conv_state_tok,
                        col2,
                        mask_1d,
                    )
                if KERNEL_WIDTH >= 5:
                    tl.store(
                        base_ptr
                        + (block_token_offset + 3 * dilation) * stride_conv_state_tok,
                        col3,
                        mask_1d,
                    )
            # case 2:
            if curr_token_pos + dilation >= seqlen and seqlen % SEQ_SIZE_PER_BLOCK != 0:
                block_token_offset = (curr_token_pos + dilation) % seqlen
                write_page_idx = tl.load(
                    block_map_ptr
                    + idx_seq * max_block_size
                    + (seqlen + prefix_length) // SEQ_SIZE_PER_BLOCK
                )
                base_ptr = (
                    conv_states_ptr
                    + write_page_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                )
                tl.store(
                    base_ptr + block_token_offset * stride_conv_state_tok, col0, mask_1d
                )
                if KERNEL_WIDTH >= 3:
                    tl.store(
                        base_ptr
                        + (block_token_offset + dilation) * stride_conv_state_tok,
                        col1,
                        mask_1d,
                    )
                if KERNEL_WIDTH >= 4:
                    tl.store(
                        base_ptr
                        + (block_token_offset + 2 * dilation) * stride_conv_state_tok,
                        col2,
                        mask_1d,
                    )
                if KERNEL_WIDTH >= 5:
                    tl.store(
                        base_ptr
                        + (block_token_offset + 3 * dilation) * stride_conv_state_tok,
                        col3,
                        mask_1d,
                    )


def causal_conv1d_fn_dilated(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    block_map: Optional[torch.Tensor],
    prefix_lengths: torch.Tensor,
    seq_size_per_block: int,
    dilation: int = 1,
    activation: str = "silu",
    pad_slot_id: int = -1,
):
    """
    计算dilation 个conv group

    """
    original_x_dtype = x.dtype
    x = x.to(weight.dtype)
    out = torch.empty_like(x)

    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = (width - 1) * dilation
    padded_batch = query_start_loc.size(0) - 1

    # 创建所有任务（包括所有组和所有 chunks）
    batch_list = []
    group_id_list = []
    chunk_offset_list = []

    for seq_idx in range(padded_batch):
        seq_start = query_start_loc[seq_idx].item()
        seq_end = query_start_loc[seq_idx + 1].item()
        seq_len = seq_end - seq_start

        # 为每个 dilation group创建任务
        for group_id in range(dilation):
            group_tokens = (seq_len - group_id + dilation - 1) // dilation
            num_chunks = (group_tokens + BLOCK_M - 1) // BLOCK_M

            for chunk_idx in range(num_chunks):
                batch_list.append(seq_idx)
                group_id_list.append(group_id)
                chunk_offset_list.append(chunk_idx)

    if len(batch_list) == 0:
        return out.to(original_x_dtype)
    batch_ptr = torch.tensor(batch_list, dtype=torch.int32, device=x.device)
    group_id_ptr = torch.tensor(group_id_list, dtype=torch.int32, device=x.device)
    token_chunk_offset_ptr = torch.tensor(
        chunk_offset_list, dtype=torch.int32, device=x.device
    )

    # 所有group可以并行独立执行， 放在 grid.x维度
    grid = (len(batch_list), triton.cdiv(dim, BLOCK_N))

    _causal_conv1d_fwd_kernel_dilated[grid](
        x,
        weight,
        bias if bias is not None else torch.empty([0], device=x.device, dtype=x.dtype),
        (
            conv_states
            if conv_states is not None
            else torch.empty([0], device=x.device, dtype=x.dtype)
        ),
        (
            block_map
            if block_map is not None
            else torch.empty([0], device=x.device, dtype=torch.int32)
        ),
        prefix_lengths,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        group_id_ptr,
        out,
        padded_batch,
        dim,
        block_map.size(1) if block_map is not None else 0,
        0,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0) if conv_states is not None else 0,
        conv_states.stride(1) if conv_states is not None else 0,
        conv_states.stride(2) if conv_states is not None else 0,
        0,
        out.stride(0),
        out.stride(1),
        pad_slot_id,
        dilation,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_CACHE=block_map is not None and conv_states is not None,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
        IS_CONTINUOUS_BATCHING=True,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=triton.next_power_of_2(state_len),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out.to(original_x_dtype)
