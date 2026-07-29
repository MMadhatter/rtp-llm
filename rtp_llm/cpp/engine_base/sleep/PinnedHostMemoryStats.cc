#include "rtp_llm/cpp/engine_base/sleep/PinnedHostMemoryVerifier.h"

#include <sstream>

namespace rtp_llm {

bool PinnedHostAllocatorSnapshot::empty() const {
    return allocations == 0 && allocated_bytes == 0;
}

bool PinnedHostAllocatorSnapshot::hasStaleActiveCounters() const {
    return empty() && (active_bytes != 0 || active_requests != 0);
}

std::string PinnedHostAllocatorSnapshot::debugString() const {
    std::ostringstream oss;
    oss << "active_bytes=" << active_bytes << " active_requests=" << active_requests
        << " allocations=" << allocations << " allocated_bytes=" << allocated_bytes;
    return oss.str();
}

absl::Status verifyPinnedHostAllocatorSnapshotIsEmpty(const PinnedHostAllocatorSnapshot& snapshot) {
    if (snapshot.empty()) {
        return absl::OkStatus();
    }
    return absl::FailedPreconditionError(
        "CUDA pinned-host backing allocations remain after cache flush: " + snapshot.debugString());
}

}  // namespace rtp_llm
