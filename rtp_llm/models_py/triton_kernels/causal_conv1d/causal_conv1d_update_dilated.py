"""
Dilated causal conv1d update kernel
conv state的访问步长发发生了变化, 目前还没支持speculative decoding
"""

from typing import Optional, Union

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.causal_conv1d.op import cal_block_idx

BLOCK_M = 128  # tokens per block
BLOCK_N = 64  # features per block
PAD_SLOT_ID = -1


@triton.jit()
def _causal_conv1d_dilated_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    block_map_ptr,
    stride_block_map: tl.int32,
    sequence_lengths_ptr,
    query_start_loc_ptr,  # (batch + 1)
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    dilation: tl.constexpr,
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    NP2_STATELEN_TOTAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQ_SIZE_PER_BLOCK: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    # if IS_VARLEN:
    #     query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    #     query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
    #     # revise state_len and seqlen
    #     state_len = state_len - (seqlen - (query_end_index - query_start_index))
    #     seqlen = query_end_index - query_start_index
    #     x_offset = query_start_index * stride_x_token
    #     o_offset = query_start_index * stride_o_token
    # else:
    # query_start_index = idx_seq * seqlen
    # query_end_index = query_start_index + seqlen
    x_offset = idx_seq * stride_x_seq
    o_offset = idx_seq * stride_o_seq

    # if query_start_index >= query_end_index:
    #     return

    sequence_length = tl.load(sequence_lengths_ptr + idx_seq).to(tl.int32)
    read_block_offset = cal_block_idx(sequence_length - 1, SEQ_SIZE_PER_BLOCK)
    read_block_id = tl.load(
        block_map_ptr + idx_seq * stride_block_map + read_block_offset
    ).to(tl.int64)
    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (read_block_id * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + dilation * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = (
            prior_tokens + 2 * dilation * stride_conv_state_tok
        )  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        conv_states_ptrs = (
            prior_tokens + 3 * dilation * stride_conv_state_tok
        )  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        conv_states_ptrs = (
            prior_tokens + 4 * dilation * stride_conv_state_tok
        )  # [BLOCK_N]
        col4 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN_TOTAL)  # [BLOCK_M]

    # the conv_state updates works in a sliding
    # window manner, at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (read_block_id * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + 1) * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = ((idx_tokens + 1) < state_len)[:, None] & (idx_feats < dim)[None, :]
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)
    # without debug barrier, the final conv_state and o is not correct

    VAL = state_len - 1
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)

    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)
    # for seqLen = n, we need to write n block in sequential manner
    for idx in tl.range(seqlen):
        write_block_offset = (cal_block_idx(sequence_length, SEQ_SIZE_PER_BLOCK)) + idx
        write_block_id = tl.load(
            block_map_ptr + idx_seq * stride_block_map + write_block_offset
        ).to(tl.int32)

        conv_state_base = (
            conv_state_ptr
            + (write_block_id * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )  # [BLOCK_N,]

        # base offset
        idx_tokens_offset = idx_tokens - idx

        conv_state_ptrs_target = (
            conv_state_base + (idx_tokens_offset * stride_conv_state_tok)[:, None]
        )  # [BLOCK_M, BLOCK_N]
        mask = (
            (idx_tokens_offset >= 0)[:, None]
            & (idx_tokens_offset < state_len)[:, None]
            & (idx_feats < dim)[None, :]
        )
        tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + (4 * stride_w_width)  # [BLOCK_N] tensor
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + (5 * stride_w_width)  # [BLOCK_N] tensor
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
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
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        # With dilation, simple rolling update doesn't work
        # We need to account for the dilation gap between conv positions
        # For dilation > 1, the next token to enter the conv window
        # is not the one we just loaded (matrix_x), but d tokens before it
        #
        # For example with dilation=2, kernel_width=4:
        # - Current window uses: [t-6, t-4, t-2, t]
        # - Next window uses:    [t-5, t-3, t-1, t+1]
        #
        # The rolling update pattern should reload from memory since
        # we can't reuse the values from the previous iteration directly

        # TODO(serina.wzq): For speculative decoding, this needs to be reimplemented to correctly handle dilation
        # Current simple rolling update is only correct for dilation=1
        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_update_dilated(
    x: torch.Tensor,
    conv_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    dilation: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    if validate_data:
        assert (
            cache_seqlens is None
        ), "cache_seqlens is not supported for speculative decoding"
        assert (
            pad_slot_id is not None
        ), "pad_slot_id is required for speculative decoding"
        assert (
            x.stride(1) == 1
        ), "x is expected to be contiguous along the feature-dimension"
        assert block_map is not None, "block_map is required for speculative decoding"
        assert (
            query_start_loc is None
        ), "query_start_loc is not supported for speculative decoding"

    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_states: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_states.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_states.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_states (currently stride={conv_states.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_states
        assert dim == conv_states.size(1)
        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = torch.empty([batch, seqlen, dim], device=x.device, dtype=x.dtype).transpose(
        1, 2
    )
    stride_w_dim, stride_w_width = weight.stride()

    # X (batch, dim, seqlen)
    stride_x_seq, stride_x_dim, stride_x_token = x.stride()
    stride_o_seq, stride_o_dim, stride_o_token = out.stride()

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_states.stride()

    state_len = (width - 1) * dilation
    np2_statelen = triton.next_power_of_2(state_len)
    # when speculative, we load (state_len - 1) token from conv_states and (seqlen) token from x, then store them in different block
    np2_statelen_total = triton.next_power_of_2(state_len - 1 + seqlen)

    stride_block_map = block_map.size(1) if block_map is not None else 0

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_dilated_update_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_seqlens,
        block_map,
        stride_block_map,
        sequence_lengths,
        query_start_loc,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        dilation,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        NP2_STATELEN=np2_statelen,
        NP2_STATELEN_TOTAL=np2_statelen_total,
        BLOCK_N=256,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)
