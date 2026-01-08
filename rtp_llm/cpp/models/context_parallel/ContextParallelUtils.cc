#include "rtp_llm/cpp/models/context_parallel/ContextParallelUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <numeric>
#include <vector>

using namespace std;

namespace rtp_llm {

std::vector<int> generateZigZagShuffleIndices(int num_padded_input_tokens, int cp_size) {
    std::vector<int> shuffle_indices(num_padded_input_tokens);

    // pair_size = num_input_tokens / (2 * cp_size)
    const int chunk_num = cp_size * 2;
    RTP_LLM_CHECK_WITH_INFO(
        num_padded_input_tokens % chunk_num == 0,
        "num_padded_input_tokens must be multiple of (cp_size * 2), got num_padded_input_tokens=%d, cp_size=%d",
        num_padded_input_tokens,
        cp_size);

    const int pair_size = num_padded_input_tokens / chunk_num;

    // Direct calculation: O(n) with optimal cache locality
    // Zig-zag: alternately take groups from start (forward) and end (backward) of entire sequence
    for (int i = 0; i < num_padded_input_tokens; ++i) {
        const int pair_idx    = i / pair_size;  // which group in output (0,1,2,3,...)
        const int pair_offset = i % pair_size;  // offset within group
        const int half_pos    = pair_idx >> 1;  // pair_idx / 2
        // Zig-zag: even group indices from start, odd group indices from end
        int target_idx;
        if (pair_idx & 1) {
            // Odd group index: take from end, counting backwards
            // pair_idx=1, half_pos=0: take last group from entire sequence
            // pair_idx=3, half_pos=1: take 2nd-to-last group from entire sequence
            target_idx = num_padded_input_tokens - pair_size * (half_pos + 1) + pair_offset;
        } else {
            // Even group index: take from start, counting forwards
            // pair_idx=0, half_pos=0: take first group from entire sequence
            // pair_idx=2, half_pos=1: take second group from entire sequence
            target_idx = half_pos * pair_size + pair_offset;
        }
        shuffle_indices[i] = target_idx;
    }
    return shuffle_indices;
}

bool contextParallelLoadBalanceSplit(const std::vector<int>& total_input_tokens,
                                     std::vector<int>&       input_tokens,
                                     std::vector<int>&       shuffle_indices,
                                     int                     cp_rank,
                                     int                     cp_size,
                                     int                     cp_chunk_size,
                                     int                     cp_padding_size) {
    const int input_token_size      = static_cast<int>(total_input_tokens.size());
    const int padded_seq_token_size = input_token_size + cp_padding_size;
    RTP_LLM_CHECK(cp_rank >= 0 && cp_rank < cp_size);
    // Generate zig-zag shuffle indices
    const auto zigzag_indices = generateZigZagShuffleIndices(padded_seq_token_size, cp_size);

    // Calculate this rank's chunk range in the shuffled sequence
    const int start_pos = cp_rank * cp_chunk_size;
    const int end_pos   = start_pos + cp_chunk_size;

    // Validate range
    if (start_pos >= padded_seq_token_size) {
        return false;
    }
    // Copy this rank's chunk using shuffled indices
    for (int i = 0, j = start_pos; j < end_pos && i < cp_chunk_size; ++i, ++j) {
        const int src_idx = zigzag_indices[j];
        if (src_idx < input_token_size) {  // Skip padding tokens
            input_tokens[i] = total_input_tokens[src_idx];
        }
        shuffle_indices[i] = src_idx;  // include padding tokens
    }

    return true;
}

torch::Tensor generateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths, int cp_size) {
    int           num_prefill_streams = prefill_cp_chunk_lengths.size(0);
    int           total_token_size    = torch::sum(prefill_cp_chunk_lengths).item<int>();
    torch::Tensor qkv_restore_indices =
        torch::empty({cp_size, total_token_size}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));

    int* qkv_data = qkv_restore_indices.data_ptr<int>();

    // Fill restore indices for each cp rank
    int chunk_offset = 0;
    for (int i = 0; i < num_prefill_streams; i++) {
        int chunk_length    = prefill_cp_chunk_lengths[i].item<int>();
        int prefill_qkv_len = chunk_length * cp_size;

        std::vector<int> shuffle_indices = generateZigZagShuffleIndices(prefill_qkv_len, cp_size);

        // Directly copy data for each rank using pointer arithmetic
        for (int cp_rank = 0; cp_rank < cp_size; cp_rank++) {
            const int* src = shuffle_indices.data() + cp_rank * chunk_length;
            int*       dst = qkv_data + cp_rank * total_token_size + chunk_offset;
            std::memcpy(dst, src, chunk_length * sizeof(int));
        }

        chunk_offset += chunk_length;
    }

    torch::Tensor qkv_restore_indices_1d = qkv_restore_indices.reshape({-1});
    torch::Tensor sorted_indices         = torch::argsort(qkv_restore_indices_1d);
    return sorted_indices;
}

torch::Tensor generateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                     const torch::Tensor& prefill_cp_padding_lengths,
                                     int                  cp_size) {
    int num_prefill_streams = prefill_cp_chunk_lengths.size(0);

    // Calculate padded sequence lengths: chunk_length * cp_size
    auto padded_seq_lengths = prefill_cp_chunk_lengths * cp_size;

    // Calculate total mask size
    int total_size = torch::sum(padded_seq_lengths).item<int>();

    // Create output tensor on CPU first for efficient construction
    torch::Tensor padding_mask = torch::empty({total_size}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    int*          mask_data    = padding_mask.data_ptr<int>();

    // Fill mask for each stream
    int offset = 0;
    for (int i = 0; i < num_prefill_streams; i++) {
        int padded_length = padded_seq_lengths[i].item<int>();
        int padding_count = prefill_cp_padding_lengths[i].item<int>();
        int valid_count   = padded_length - padding_count;

        // Set valid tokens to 1
        std::fill_n(mask_data + offset, valid_count, 1);
        // Set padding tokens to 0
        std::fill_n(mask_data + offset + valid_count, padding_count, 0);

        offset += padded_length;
    }
    return padding_mask;
}

}  // namespace rtp_llm