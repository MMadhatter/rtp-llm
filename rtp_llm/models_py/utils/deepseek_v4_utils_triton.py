# 一些临时实现，可能被删掉
import torch
import triton
import triton.language as tl


@triton.jit
def prefill_conv1d_padding_kernel(
    x_ptr,
    cu_seqlens_ptr,
    output_ptr,
    hidden_size,
    num_tokens,
    num_seqs,
    padding_size,
    output_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < output_cols

    # 初始化为 0
    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 遍历每个序列
    for seq_idx in range(num_seqs):
        seq_start_in = tl.load(cu_seqlens_ptr + seq_idx)
        seq_end_in = tl.load(cu_seqlens_ptr + seq_idx + 1)
        seq_len = seq_end_in - seq_start_in

        # 输出中该序列的位置
        seq_start_out = seq_start_in + seq_idx * padding_size
        seq_content_start_out = seq_start_out + padding_size
        seq_end_out = seq_start_out + padding_size + seq_len

        # 判断当前 block 的列是否在该序列的内容区域（非 padding 区域）
        in_content_region = (col_offsets >= seq_content_start_out) & (
            col_offsets < seq_end_out
        )

        # 计算输入位置
        in_col = seq_start_in + (col_offsets - seq_content_start_out)
        in_col = tl.where(in_content_region, in_col, 0)

        # 加载数据
        in_offset = row_idx * num_tokens + in_col
        x_val = tl.load(x_ptr + in_offset, mask=in_content_region & mask, other=0.0)

        # 更新结果
        result = tl.where(in_content_region, x_val, result)

    # 写入输出
    output_offset = row_idx * output_cols + col_offsets
    tl.store(output_ptr + output_offset, result, mask=mask)


def create_padding_mask(
    cu_seqlens: torch.Tensor,
    num_tokens: int,
    num_seqs: int,
    padding_size: int,
) -> torch.Tensor:
    """
    生成非 padding 位置的列索引（用于直接 indexing）

    Returns:
        indices: shape [num_tokens]，表示 padded tensor 中哪些列对应原始 tokens
    """
    seq_indices = torch.arange(num_seqs, device=cu_seqlens.device)
    content_starts_out = (
        cu_seqlens[:-1] + (seq_indices + 1) * padding_size
    )  # [num_seqs]

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # [num_seqs]

    max_seq_len = seq_lens.max().item()
    content_offsets = torch.arange(
        max_seq_len, device=cu_seqlens.device
    )  # [max_seq_len]

    content_indices = (
        content_starts_out[:, None] + content_offsets[None, :]
    )  # [num_seqs, max_seq_len]
    valid_mask = content_offsets[None, :] < seq_lens[:, None]  # [num_seqs, max_seq_len]

    indices = content_indices[valid_mask]  # [num_tokens]

    return indices


def conv1d_prefill_padding(
    x_bct: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    hidden_size, num_tokens = x_bct.shape
    num_seqs = cu_seqlens.shape[0] - 1
    output_cols = num_tokens + num_seqs * padding_size

    output = torch.zeros(
        (hidden_size, output_cols), dtype=x_bct.dtype, device=x_bct.device
    )

    BLOCK_SIZE = 256
    grid = (hidden_size, triton.cdiv(output_cols, BLOCK_SIZE))

    prefill_conv1d_padding_kernel[grid](
        x_bct,
        cu_seqlens,
        output,
        hidden_size,
        num_tokens,
        num_seqs,
        padding_size,
        output_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    indices = create_padding_mask(cu_seqlens, num_tokens, num_seqs, padding_size)

    return output, indices
