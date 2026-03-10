import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper


def plan_prefix_paged_attention(
    wrapper: BatchPrefillWithPagedKVCacheWrapper,
    qo_indptr: torch.Tensor,
    prefix_lengths: torch.Tensor,
    params,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    device,
) -> None:
    """Plan paged attention for the prefix portion of KV cache.

    All prefix positions precede every new-token position, so causal=False.
    This is shared by all CP implementations that need prefix cache support.
    """
    batch_size = params.kvlen_h.shape[0]
    prefix_lens = prefix_lengths.cpu().to(torch.int32)
    assert (prefix_lens % page_size == 0).all(), (
        f"prefix lengths must be multiples of page_size({page_size}), "
        f"got {prefix_lens}"
    )

    prefix_pages = prefix_lens // page_size
    page_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    page_indptr[1:] = prefix_pages.cumsum(0)

    full_page_starts = params.decode_page_indptr_h[:batch_size].to(torch.int32)
    all_page_indices = params.page_indice_d

    total_pages = page_indptr[-1].item()
    if total_pages > 0:
        expanded_starts = torch.repeat_interleave(full_page_starts, prefix_pages)
        local_offsets = torch.arange(
            total_pages, dtype=torch.int32
        ) - torch.repeat_interleave(page_indptr[:batch_size], prefix_pages)
        gather_idx = (expanded_starts + local_offsets).long().to(device)
        prefix_page_indices = all_page_indices[gather_idx]
    else:
        prefix_page_indices = torch.tensor([], dtype=torch.int32, device=device)

    last_page_len = torch.full([batch_size], page_size, dtype=torch.int32)

    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=page_indptr.to(device),
        paged_kv_indices=prefix_page_indices,
        paged_kv_last_page_len=last_page_len.to(device),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=False,
        q_data_type=torch.bfloat16,
    )


# generate kv_indices for zigzag load balance with partial q and full kv
def generate_kv_indices(cp_chunk_lengths, cp_rank, cp_size, is_non_local=False):
    """
    Generate kv_indices for zigzag load balance with partial q and full kv
    Args:
        cp_chunk_lengths: List of chunk lengths for each CP rank
        cp_rank: Rank of the current process
        cp_size: Size of the context parallel group
        is_non_local: Whether the kv_indices include the local kv index
    Returns:
        kv_part_0_indices: List of indices for the first part of the kv
        kv_part_1_indices: List of indices for the second part of the kv
    """
    total_seq_lengths = [x * cp_size for x in cp_chunk_lengths]

    kv_part_0_indices = []
    kv_part_1_indices = []
    seq_offset = 0
    for i in range(len(total_seq_lengths)):
        assert cp_chunk_lengths[i] % 2 == 0
        half_chunk_len = cp_chunk_lengths[i] // 2
        # with out prefix cache, the start kv position is always 0
        start_pos_part0 = 0
        end_pos_part0 = half_chunk_len * (cp_rank + 1 - int(is_non_local))
        start_pos_part1 = 0
        end_pos_part1 = half_chunk_len * (2 * cp_size - cp_rank - int(is_non_local))

        if end_pos_part0 > start_pos_part0:
            kv_part_0_indices.extend(
                range(start_pos_part0 + seq_offset, end_pos_part0 + seq_offset)
            )
        if end_pos_part1 > start_pos_part1:
            kv_part_1_indices.extend(
                range(start_pos_part1 + seq_offset, end_pos_part1 + seq_offset)
            )
        seq_offset += total_seq_lengths[i]
    return kv_part_0_indices, kv_part_1_indices


# generate q_indices for zigzag load balance with partial q and full kv
def generate_q_indices(cp_chunk_lengths):
    """Generate two sets of indices by splitting each chunk in half.
    Args:
        cp_chunk_lengths: List of chunk lengths for each CP rank
    Returns:
        indices0: List of first half indices from each chunk
        indices1: List of second half indices from each chunk
    Example 1:
        cp_chunk_lengths = [8, 4, 4]
        indices0 = [0, 1, 2, 3, 8, 9, 12, 13]
        indices1 = [4, 5, 6, 7, 10, 11, 14, 15]
    """
    indices0 = []
    indices1 = []
    offset = 0
    for chunk_len in cp_chunk_lengths:
        # Use ceiling division for first half (gets extra element if odd)
        half0 = (chunk_len + 1) // 2
        indices0.extend(range(offset, offset + half0))
        indices1.extend(range(offset + half0, offset + chunk_len))
        offset += chunk_len

    return indices0, indices1


# for all2all mode with zigzag loadbalance
def generate_half_q_indices(cp_chunk_lengths):
    """
    Generate half q indices for all2all with zigzag loadbalance
    Args:
        cp_chunk_lengths: List of chunk lengths for each CP rank
    Returns:
        half_q_indices: List of indices for the first half of the q
    """
    half_q_indices = []
    offset = 0
    for chunk_len in cp_chunk_lengths:
        assert chunk_len % 2 == 0
        half_q_indices.extend(range(offset + (chunk_len) // 2, offset + chunk_len))
        offset += chunk_len
    return half_q_indices


# for all2all mode with zigzag loadbalance
def generate_half_kv_indices(cp_chunk_lengths):
    """
    Generate half kv indices for all2all with zigzag loadbalance
    Args:
        cp_chunk_lengths: List of chunk lengths for each CP rank
    Returns:
        half_kv_indices: List of indices for the first half of the kv
    """
    half_kv_indices = []
    offset = 0
    for chunk_len in cp_chunk_lengths:
        assert chunk_len % 2 == 0
        half_kv_indices.extend(range(offset, offset + (chunk_len) // 2))
        offset += chunk_len
    return half_kv_indices
