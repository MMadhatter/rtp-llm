#pragma once

#include <cstdint>
#include <string>

#include "absl/status/status.h"

namespace rtp_llm {

struct PinnedHostAllocatorSnapshot {
    int64_t active_bytes    = 0;
    int64_t active_requests = 0;
    int64_t allocations     = 0;
    int64_t allocated_bytes = 0;

    // allocations/allocated_bytes are the authoritative backing-block
    // counters. Some torch versions leave active_* stale after no-event frees.
    bool empty() const;
    bool hasStaleActiveCounters() const;
    std::string debugString() const;
};

// Pure validation entry point kept separate so the Level-3 zero-allocation
// contract can be unit-tested without initializing CUDA.
absl::Status verifyPinnedHostAllocatorSnapshotIsEmpty(const PinnedHostAllocatorSnapshot& snapshot);

// Flushes PyTorch's CUDA caching host allocator, logs its before/after state,
// and enforces that no cached or live pinned allocations remain. On non-CUDA
// builds this is a no-op that returns OK.
absl::Status flushAndVerifyCudaPinnedHostMemory(PinnedHostAllocatorSnapshot* before = nullptr,
                                                PinnedHostAllocatorSnapshot* after = nullptr);

}  // namespace rtp_llm
