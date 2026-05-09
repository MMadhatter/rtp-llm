"""Context-Parallel helpers for DeepSeek-V4.

RTP-LLM's CP repurposes the TP process group as the CP group (see
``ParallelismConfig::get_attn_tp_size`` — returns 1 when CP enabled).
The C++ ``ZigZagProcessor`` splits the padded prefill tokens across the
CP group with a zigzag layout and attaches
``attention_inputs.context_parallel_info`` carrying:

- ``prefill_qkv_padding_mask``  : ``[padded_seq_len] int32`` — 1 for
  real tokens, 0 for padding. Padding sits at global positions
  ``[input_len, padded_seq_len)``.
- ``prefill_qkv_restore_indice``: ``[padded_seq_len] int32`` —
  ``restore[global_pos] = gathered_flat_idx`` (where gathered is the
  concat of per-rank chunks: ``cp_rank * chunk_len + local_idx``).

V4's model-side CP work:
- Attention: rank-local Q (chunk_len tokens) × FULL KV (stripped to
  ``input_len``).  RoPE uses **global** positions.
- Compressor / Indexer: all-gather kv/score before S-pool, so every
  rank's ``kv_cache`` buffer holds the full compressed KV for decode.
- MoE: stays rank-local — DeepEP dispatches the ``chunk_len`` tokens
  naturally; no gather needed.

All derived per-forward quantities are bundled into ``CPContext`` and
stashed on each module via ``_cp_ctx`` before ``forward`` runs.  A
``None`` value means "no CP" and every module falls through to the
single-rank path unchanged.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather


@dataclass
class CPContext:
    """Derived CP metadata for one prefill forward."""

    cp_size: int
    cp_rank: int
    # Rank-local chunk length (== input_ids.size(1) on the adapter side;
    # framework-padded multiple of ``cp_size*2/cp_size = 2``).
    chunk_length: int
    # Total padded global seqlen = cp_size * chunk_length.
    padded_seq_len: int
    # Real un-padded global seqlen (= user's prefill input length).
    seq_len_full: int
    # [chunk_length] int64 — local idx i -> position in the padded
    # request-concat stream consumed by ``prefill_qkv_padding_mask`` /
    # ``prefill_qkv_restore_indice``.
    relative_positions: torch.Tensor
    # Prefix length already resident in KV cache for continuation prefill.
    prefix_length: int
    # [chunk_length] int64 — local idx i -> absolute sequence position
    # (prefix_length + relative_positions).  For padding local idxs, the
    # position is clamped to a valid token in the same request; their
    # attention output is discarded by the framework's strip-pad gather.
    global_positions: torch.Tensor
    # [chunk_length] int32 — request id for each rank-local token.  Under
    # B==1 this is all zeros.  Under B>1 it mirrors the framework's flat
    # rank-local order: all local tokens for request 0, then request 1, ...
    req_id_per_token: Optional[torch.Tensor]
    # [B] int64 — per-request prefix/start position.  Kept because CP+B>1
    # cannot be represented by the legacy scalar ``prefix_length``.
    prefix_lengths: Optional[torch.Tensor]
    # [chunk_length] bool — True if local idx maps to a real token
    # (padding_mask[relative_pos] == 1), False for padding slots.
    local_is_real: torch.Tensor
    # [seq_len_full] int64 — gathered-flat index for each real token in
    # GLOBAL order. ``gathered.index_select(0, unpad_restore)`` yields the
    # full un-padded sequence.
    unpad_restore: torch.Tensor
    # Total sequence length after this prefill step (prefix + current input).
    seq_len_total: int
    # Raw cp_info, kept for any caller needing extra fields.
    cp_info: object
    # ----- Pool-write side (CP "global" view) ------------------------------
    # The framework's ``attention_inputs.{cu_seqlens, input_lengths}`` are
    # rank-local (already split by ZigZagProcessor). After ``cp_all_gather_full``
    # the KV tensor has length ``seq_len_full`` in GLOBAL request order, so
    # the SWA paged-tail slot_mapping (and any other write-side meta keyed
    # off cu_seqlens) must use the global view instead. These two tensors
    # are derived once per forward from
    # ``cp_info.prefill_actual_input_lengths_cpu`` and reused by every
    # FP8 SWA layer's write meta.
    #
    # ``input_lengths_global``  : ``[B] int32`` per-req full sequence length
    # ``cu_seqlens_global``     : ``[B+1] int32`` cumulative prefix sum
    # Both are ``None`` outside of a CP-active forward (cp_size <= 1) — the
    # consumer falls back to the rank-local tensors in that case.
    input_lengths_global: Optional[torch.Tensor]
    cu_seqlens_global: Optional[torch.Tensor]


def build_cp_context(
    cp_info,
    cp_size: int,
    cp_rank: int,
    chunk_length: int,
    device: torch.device,
    position_offset: Union[int, torch.Tensor] = 0,
) -> CPContext:
    """Compute the per-forward derived CPContext from framework metadata."""
    padding_mask = cp_info.prefill_qkv_padding_mask
    restore_indices = cp_info.prefill_qkv_restore_indice
    if padding_mask.device != device:
        padding_mask = padding_mask.to(device)
        restore_indices = restore_indices.to(device)
    padded_seq_len = int(padding_mask.shape[0])

    assert cp_size * chunk_length == padded_seq_len, (
        f"cp_size({cp_size}) * chunk_length({chunk_length}) != "
        f"padded_seq_len({padded_seq_len})"
    )

    actual_input_lengths_cpu = getattr(
        cp_info, "prefill_actual_input_lengths_cpu", None
    )
    input_lengths_global: Optional[torch.Tensor] = None
    cu_seqlens_global: Optional[torch.Tensor] = None
    if actual_input_lengths_cpu is not None and actual_input_lengths_cpu.numel() > 0:
        input_lengths_global = actual_input_lengths_cpu.to(
            device=device, dtype=torch.int32
        ).contiguous()
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        cu_seqlens_global = torch.cat(
            [zero, torch.cumsum(input_lengths_global, dim=0).to(torch.int32)]
        ).contiguous()

    chunk_lengths_obj = getattr(cp_info, "prefill_cp_chunk_lengths", None)
    if chunk_lengths_obj is not None and chunk_lengths_obj.numel() > 0:
        chunk_lengths = [int(v) for v in chunk_lengths_obj.detach().cpu().tolist()]
    elif input_lengths_global is not None and input_lengths_global.numel() == 1:
        chunk_lengths = [chunk_length]
    else:
        # Test/legacy fallback: a single stream with the caller-provided
        # aggregate rank-local chunk length.
        chunk_lengths = [chunk_length]

    assert sum(chunk_lengths) == chunk_length, (
        f"sum(prefill_cp_chunk_lengths)={sum(chunk_lengths)} != "
        f"chunk_length={chunk_length}"
    )
    for i, per_req_chunk in enumerate(chunk_lengths):
        assert per_req_chunk % 2 == 0, (
            f"prefill_cp_chunk_lengths[{i}]={per_req_chunk} must be even "
            "for zigzag CP"
        )

    if input_lengths_global is not None:
        B = int(input_lengths_global.numel())
    else:
        B = len(chunk_lengths)
    assert B == len(
        chunk_lengths
    ), f"num global lengths ({B}) != num CP chunks ({len(chunk_lengths)})"

    if isinstance(position_offset, torch.Tensor):
        prefix_lengths = position_offset.to(
            device=device, dtype=torch.long
        ).contiguous()
        if prefix_lengths.numel() == 1 and B > 1:
            prefix_lengths = prefix_lengths.expand(B).contiguous()
    else:
        prefix_lengths = torch.full(
            (B,), int(position_offset), dtype=torch.long, device=device
        )
    assert (
        prefix_lengths.numel() >= B
    ), f"prefix_lengths has {prefix_lengths.numel()} entries, expected at least {B}"
    prefix_lengths = prefix_lengths[:B].contiguous()

    if input_lengths_global is not None:
        real_lengths = input_lengths_global.to(device=device, dtype=torch.long)
    else:
        # No actual lengths means no padding information beyond the mask.
        # For the single-stream fallback this collapses to seq_len_full below.
        real_lengths = torch.tensor(
            [int((padding_mask == 1).sum().item())],
            dtype=torch.long,
            device=device,
        )

    # C++ ZigZagProcessor applies the zigzag plan independently per prefill
    # stream/request, then concatenates the rank-local chunks.  Generate the
    # same padded-concat coordinates here; the previous single-stream formula
    # is only valid for B==1.
    padded_positions = []
    per_req_positions = []
    req_ids = []
    padded_seq_offset = 0
    for req_id, per_req_chunk in enumerate(chunk_lengths):
        pair_size = per_req_chunk // 2
        padded_len = per_req_chunk * cp_size
        arange_pair = torch.arange(pair_size, dtype=torch.long, device=device)
        even_padded = padded_seq_offset + cp_rank * pair_size + arange_pair
        odd_padded = (
            padded_seq_offset + padded_len - (cp_rank + 1) * pair_size + arange_pair
        )
        req_relative = torch.cat(
            [even_padded - padded_seq_offset, odd_padded - padded_seq_offset]
        )
        if req_id < int(real_lengths.numel()):
            max_real_pos = max(int(real_lengths[req_id].item()) - 1, 0)
        else:
            max_real_pos = max(padded_len - 1, 0)
        per_req_positions.append(req_relative.clamp_max(max_real_pos))
        padded_positions.append(torch.cat([even_padded, odd_padded]))
        req_ids.append(
            torch.full((per_req_chunk,), req_id, dtype=torch.int32, device=device)
        )
        padded_seq_offset += padded_len

    relative_positions = torch.cat(padded_positions).contiguous()
    local_positions = torch.cat(per_req_positions).contiguous()
    req_id_per_token = torch.cat(req_ids).contiguous()

    local_is_real = padding_mask[relative_positions] == 1  # [chunk_length] bool
    unpad_restore = restore_indices[padding_mask == 1].to(torch.long)  # [seq_len_full]
    seq_len_full = int(unpad_restore.shape[0])
    prefix_per_token = prefix_lengths.gather(0, req_id_per_token.to(torch.long))
    global_positions = (prefix_per_token + local_positions).contiguous()
    prefix_length = int(prefix_lengths[0].item()) if prefix_lengths.numel() > 0 else 0
    if input_lengths_global is not None:
        seq_len_total = int((prefix_lengths + real_lengths[:B]).max().item())
    else:
        seq_len_total = prefix_length + seq_len_full

    return CPContext(
        cp_size=int(cp_size),
        cp_rank=int(cp_rank),
        chunk_length=int(chunk_length),
        padded_seq_len=padded_seq_len,
        seq_len_full=seq_len_full,
        relative_positions=relative_positions,
        prefix_length=prefix_length,
        global_positions=global_positions,
        req_id_per_token=req_id_per_token,
        prefix_lengths=prefix_lengths,
        local_is_real=local_is_real,
        unpad_restore=unpad_restore,
        seq_len_total=seq_len_total,
        cp_info=cp_info,
        input_lengths_global=input_lengths_global,
        cu_seqlens_global=cu_seqlens_global,
    )


def cp_all_gather_full(
    local: torch.Tensor,
    cp_ctx: CPContext,
) -> torch.Tensor:
    """**Legacy B==1 path** (used under ``DSV4_VARLEN_PREFILL=0`` and on
    the standalone test harness): all-gather a rank-local
    ``[B, chunk_length, *F]`` tensor across the CP (== TP) group and
    strip padding → ``[B, seq_len_full, *F]`` in GLOBAL logical order.

    ``B`` must be 1. The varlen path (``DSV4_VARLEN_PREFILL=1`` —
    default; supports B>=1 multi-request prefill) calls
    :func:`cp_all_gather_full_varlen` instead, which operates on a flat
    ``[chunk_length, *F]`` tensor without the leading B dim."""
    assert local.dim() >= 2
    B = local.size(0)
    assert B == 1
    assert (
        local.size(1) == cp_ctx.chunk_length
    ), f"local.size(1)={local.size(1)} != chunk_length={cp_ctx.chunk_length}"
    trailing = local.shape[2:]

    # collective_torch.all_gather concatenates along dim 0 for 1-D/2-D
    # tensors.  Flatten trailing dims so the helper sees a simple 2-D.
    local_flat = local.reshape(cp_ctx.chunk_length, -1).contiguous()
    gathered = all_gather(local_flat, group=Group.TP)
    # gathered: [cp_size * chunk_length, prod(F)]
    full = gathered.index_select(0, cp_ctx.unpad_restore)  # [seq_len_full, prod(F)]
    return full.view((1, cp_ctx.seq_len_full) + trailing)


def cp_all_gather_full_varlen(
    local_flat: torch.Tensor,
    cp_ctx: CPContext,
) -> torch.Tensor:
    """**Varlen B>=1 path** (default ``DSV4_VARLEN_PREFILL=1``):
    all-gather a flat ``[chunk_length, *F]`` rank-local tensor across the
    CP (== TP) group and strip padding → ``[seq_len_full, *F]`` in
    GLOBAL per-request-concat order. No leading B dim.

    For B==1 this is mathematically identical to
    :func:`cp_all_gather_full` (just without the extra unsqueeze). For
    B>1 ``seq_len_full`` is the sum of per-request real lengths and the
    output is the request-concatenated KV stream — every consumer in
    the FP8 prefill stack (compressor/indexer/SWA write/attention)
    treats this as a single virtual sequence (matching the existing
    non-CP B>1 behaviour documented in
    ``prefill/forward.py::forward_layers``).
    """
    assert local_flat.dim() >= 1
    assert (
        local_flat.size(0) == cp_ctx.chunk_length
    ), f"local_flat.size(0)={local_flat.size(0)} != chunk_length={cp_ctx.chunk_length}"
    trailing = local_flat.shape[1:]
    local_2d = local_flat.reshape(cp_ctx.chunk_length, -1).contiguous()
    gathered = all_gather(local_2d, group=Group.TP)
    full = gathered.index_select(0, cp_ctx.unpad_restore)
    return full.view((cp_ctx.seq_len_full,) + trailing)


def cp_freqs_cis_local(
    freqs_cis: torch.Tensor,
    cp_ctx: CPContext,
) -> torch.Tensor:
    """Select ``freqs_cis`` rows at the GLOBAL positions of this rank's
    local tokens → ``[chunk_length, rope_dim/2]`` complex tensor suitable
    for ``apply_rotary_emb`` against a rank-local Q/K of length
    ``chunk_length``.

    Padding slots pick a valid (in-range) row from ``freqs_cis``; their
    attention output is discarded by the framework after the exit
    all-gather, so the specific rotation angle doesn't matter as long as
    it doesn't NaN or OOB."""
    pos = cp_ctx.global_positions
    if pos.device != freqs_cis.device:
        pos = pos.to(freqs_cis.device)
    # freqs_cis is complex [max_seq_len, rope_dim//2].  Clamp positions
    # to a valid range — padding slots compute a RoPE that gets thrown
    # away, but must not index past the end.
    max_s = freqs_cis.size(0)
    pos = pos.clamp_max(max_s - 1)
    return freqs_cis.index_select(0, pos)


# Legacy shim kept so existing callers (old scaffold) still link while
# we move them over.  New code should use CPContext + cp_all_gather_full.
def cp_all_gather_to_full(
    local: torch.Tensor,
    cp_info,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Deprecated: use ``build_cp_context`` + ``cp_all_gather_full``."""
    device = local.device
    chunk_length = local.size(1)
    ctx = build_cp_context(cp_info, cp_size, cp_rank, chunk_length, device)
    return cp_all_gather_full(local, ctx)


def combine_topk_swa_indices_cp_varlen(
    topk_indices: torch.Tensor,
    global_positions: torch.Tensor,
    sp_int: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    req_id_per_token: Optional[torch.Tensor] = None,
    prefix_lengths: Optional[torch.Tensor] = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """**Varlen B>=1 path** (default ``DSV4_VARLEN_PREFILL=1``):
    multi-request-aware vectorized replacement for
    ``_swa_ops_triton.combine_topk_swa_indices`` under CP.

    For B==1 single-request prefill this is bit-equal to
    :func:`combine_topk_swa_indices_cp_b1`. For B>1 it emits indices in
    the flattened workspace coordinate system:

      * compressed slots: ``req_id * M + topk_indices[t, k]``
      * SWA slots: ``req_id * M + N + local_slot``

    ``global_positions`` are absolute per-request positions
    (``prefix_lengths[req] + local_pos``), not padded-concat positions.
    """
    if req_id_per_token is None or prefix_lengths is None:
        return combine_topk_swa_indices_cp_b1(
            topk_indices=topk_indices,
            global_positions=global_positions,
            sp_int=sp_int,
            window_size=window_size,
            compress_ratio=compress_ratio,
            topk=topk,
            M=M,
            N=N,
        )

    device = topk_indices.device
    T = int(global_positions.shape[0])
    combined_topk_padded = ((topk + window_size + 127) // 128) * 128
    combined_indices = torch.full(
        (T, combined_topk_padded), -1, dtype=torch.int32, device=device
    )
    combined_lens = torch.zeros((T,), dtype=torch.int32, device=device)
    if T == 0:
        return combined_indices, combined_lens

    req = req_id_per_token.to(device=device, dtype=torch.int64)
    prefix = prefix_lengths.to(device=device, dtype=torch.int64)
    gp = global_positions.to(device=device, dtype=torch.int64)
    sp_b = prefix.gather(0, req)
    P_b = torch.clamp_max(sp_b, window_size - 1)
    gather_start = sp_b - P_b

    topk_len = torch.minimum((gp + 1) // compress_ratio, gp.new_full((), topk)).to(
        torch.int64
    )
    swa_len = torch.minimum(gp + 1, gp.new_full((), window_size)).to(torch.int64)
    combined_lens = (topk_len + swa_len).to(torch.int32).contiguous()
    req_base = req * int(M)

    if topk > 0:
        off_k = torch.arange(topk, device=device, dtype=torch.int64).unsqueeze(0)
        mask_k = off_k < topk_len.unsqueeze(1)
        topk_vals = req_base.unsqueeze(1) + topk_indices.to(torch.int64)
        combined_indices[:, :topk] = torch.where(
            mask_k,
            topk_vals.to(torch.int32),
            torch.full((T, topk), -1, dtype=torch.int32, device=device),
        )

    off_w = torch.arange(window_size, device=device, dtype=torch.int64).unsqueeze(0)
    mask_w = off_w < swa_len.unsqueeze(1)
    swa_val = (
        req_base.unsqueeze(1)
        + int(N)
        + (gp - gather_start).unsqueeze(1)
        - swa_len.unsqueeze(1)
        + 1
        + off_w
    ).to(torch.int32)
    target_col = topk_len.unsqueeze(1) + off_w
    flat_target = (
        torch.arange(T, device=device, dtype=torch.int64).unsqueeze(1)
        * combined_topk_padded
        + target_col
    )
    combined_indices.view(-1).scatter_(0, flat_target[mask_w], swa_val[mask_w])
    return combined_indices, combined_lens


def build_cp_full_prefill_positions(
    cp_ctx: CPContext,
    device: torch.device,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Return full gathered per-token compressor metadata for CP prefill.

    Output order matches ``cp_all_gather_full[_varlen]`` after
    ``unpad_restore``: request 0 real tokens, request 1 real tokens, ...

    Returns ``(positions, b_idx, seq_start_per_req, cu_seq_per_req)``.
    """
    assert (
        cp_ctx.input_lengths_global is not None
    ), "CP full prefill positions require input_lengths_global"
    lengths = cp_ctx.input_lengths_global.to(device=device, dtype=torch.long)
    if cp_ctx.prefix_lengths is not None:
        prefixes = cp_ctx.prefix_lengths.to(device=device, dtype=torch.long)
    else:
        prefixes = torch.full(
            (int(lengths.numel()),),
            int(cp_ctx.prefix_length),
            dtype=torch.long,
            device=device,
        )

    positions = []
    b_idx = []
    for req_id in range(int(lengths.numel())):
        length = int(lengths[req_id].item())
        start = int(prefixes[req_id].item())
        if length <= 0:
            continue
        positions.append(
            torch.arange(start, start + length, dtype=torch.long, device=device)
        )
        b_idx.append(torch.full((length,), req_id, dtype=torch.long, device=device))
    if positions:
        pos = torch.cat(positions).contiguous()
        req = torch.cat(b_idx).contiguous()
    else:
        pos = torch.empty((0,), dtype=torch.long, device=device)
        req = torch.empty((0,), dtype=torch.long, device=device)

    zero = torch.zeros(1, dtype=torch.int32, device=device)
    cu_seq = torch.cat(
        [zero, torch.cumsum(lengths.to(torch.int32), dim=0)]
    ).contiguous()
    return (
        pos,
        req,
        prefixes.to(device=device, dtype=torch.int32).contiguous(),
        cu_seq,
    )


def combine_topk_swa_indices_cp_b1(
    topk_indices: torch.Tensor,
    global_positions: torch.Tensor,
    sp_int: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """**Legacy B==1 path** (used under ``DSV4_VARLEN_PREFILL=0``):
    vectorized replacement for ``_swa_ops_triton.combine_topk_swa_indices``.

    The Triton kernel computes ``pos = start_pos + token_idx_in_query``
    assuming Q is a contiguous sequence slice; under zigzag CP the
    rank-local Q rows correspond to non-contiguous global positions, so
    that formula breaks. This helper consumes ``cp_ctx.global_positions``
    directly to compute per-token ``pos`` and emits the same ``[T,
    align(topk+win, 128)]`` int32 layout — sentinel ``-1`` in the unused
    tail — plus ``[T]`` int32 ``combined_lens = topk_len + swa_len``.

    For each Q-row ``t`` with global position ``gp[t]``:
      * ``topk_len = min((gp+1) // ratio, topk)``
      * ``swa_len  = min(gp+1, win)``
      * row[0:topk_len]                 = topk_indices[t, 0:topk_len]
      * row[topk_len:topk_len+swa_len]  = N + (gp - gather_start) - swa_len + 1 + off
        where ``gather_start = sp_int - P``, ``P = min(sp_int, win-1)``.

    Workspace stride ``M`` is currently unused on this legacy B==1 helper
    (``M*batch_idx == 0``), but kept in the signature so callers can share the
    same dispatch shape as the B>=1 varlen helper.
    """
    device = topk_indices.device
    T = int(global_positions.shape[0])
    combined_topk_padded = ((topk + window_size + 127) // 128) * 128
    combined_indices = torch.full(
        (T, combined_topk_padded), -1, dtype=torch.int32, device=device
    )
    combined_lens = torch.zeros((T,), dtype=torch.int32, device=device)
    if T == 0:
        return combined_indices, combined_lens

    gp = global_positions.to(device=device, dtype=torch.int64)
    P = min(sp_int, window_size - 1)
    gather_start = sp_int - P

    topk_len = torch.minimum((gp + 1) // compress_ratio, gp.new_full((), topk)).to(
        torch.int64
    )
    swa_len = torch.minimum(gp + 1, gp.new_full((), window_size)).to(torch.int64)
    combined_lens = (topk_len + swa_len).to(torch.int32).contiguous()

    if topk > 0:
        off_k = torch.arange(topk, device=device, dtype=torch.int64).unsqueeze(0)
        mask_k = off_k < topk_len.unsqueeze(1)
        combined_indices[:, :topk] = torch.where(
            mask_k,
            topk_indices.to(torch.int32),
            torch.full_like(topk_indices, -1, dtype=torch.int32),
        )

    off_w = torch.arange(window_size, device=device, dtype=torch.int64).unsqueeze(0)
    mask_w = off_w < swa_len.unsqueeze(1)
    swa_val = (
        N + (gp - gather_start).unsqueeze(1) - swa_len.unsqueeze(1) + 1 + off_w
    ).to(torch.int32)
    target_col = topk_len.unsqueeze(1) + off_w
    flat_target = (
        torch.arange(T, device=device, dtype=torch.int64).unsqueeze(1)
        * combined_topk_padded
        + target_col
    )
    flat_target = flat_target[mask_w]
    flat_val = swa_val[mask_w]
    combined_indices.view(-1).scatter_(0, flat_target, flat_val)
    return combined_indices, combined_lens


def cp_should_gather(cp_ctx: Optional[CPContext], start_pos: int) -> bool:
    """Prefill-only gate: gather runs iff a CPContext is bound.

    ``start_pos`` is intentionally ignored: CP continuation prefill also
    receives only the rank-local suffix, so compressor/indexer state must still
    all-gather the current input before pooling.
    """
    return cp_ctx is not None and cp_ctx.cp_size > 1
