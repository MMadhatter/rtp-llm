#pragma once
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include <vector>

namespace rtp_llm {

/// @brief Generate zig-zag shuffle indices for context parallel load balancing
/// @param num_input_tokens Total number of input tokens
/// @param cp_size Context parallel size
/// @return Shuffle indices vector
///
/// For num_input_tokens <= cp_size: returns sequential indices [0, 1, 2, ..., n-1]
/// For num_input_tokens > cp_size: returns zig-zag pattern with pair_size = num_input_tokens / (2 * cp_size)
///   Example 1: cp_size=4, num_input_tokens=16, pair_size=2 → [0, 1, 14, 15, 2, 3, 12, 13, 4, 5, 10, 11, 6, 7, 8, 9]
///   Example 2: cp_size=2, num_input_tokens=16, pair_size=4 → [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]
std::vector<int> generateZigZagShuffleIndices(int num_padded_input_tokens, int cp_size);

/// @brief Split and balance input tokens across context parallel ranks
/// @param total_input_tokens All input tokens before splitting
/// @param input_tokens Output: tokens for this rank (pre-allocated)
/// @param shuffle_indices Output: shuffle indices for later reshuffle
/// @param cp_rank Current context parallel rank
/// @param cp_size Total context parallel size
/// @param cp_chunk_size Chunk size per rank
/// @param cp_padding_size Padding size to add for context parallel
/// @return true if split successful, false otherwise
bool contextParallelLoadBalanceSplit(const std::vector<int>& total_input_tokens,
                                     std::vector<int>&       input_tokens,
                                     std::vector<int>&       shuffle_indices,
                                     int                     cp_rank,
                                     int                     cp_size,
                                     int                     cp_chunk_size,
                                     int                     cp_padding_size);

}  // namespace rtp_llm
