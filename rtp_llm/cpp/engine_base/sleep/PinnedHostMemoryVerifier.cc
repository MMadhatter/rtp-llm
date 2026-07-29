#include "rtp_llm/cpp/engine_base/sleep/PinnedHostMemoryVerifier.h"

#include <exception>
#include <string>

#if USING_CUDA
#include <ATen/cuda/CachingHostAllocator.h>
#include <cuda_runtime_api.h>
#endif

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

#if USING_CUDA
namespace {

PinnedHostAllocatorSnapshot snapshotAllocator(const at::HostStats& stats) {
    PinnedHostAllocatorSnapshot snapshot;
    snapshot.active_bytes    = stats.active_bytes.current;
    snapshot.active_requests = stats.active_requests.current;
    snapshot.allocations     = stats.allocations.current;
    snapshot.allocated_bytes = stats.allocated_bytes.current;
    return snapshot;
}

}  // namespace
#endif

absl::Status flushAndVerifyCudaPinnedHostMemory(PinnedHostAllocatorSnapshot* before,
                                                PinnedHostAllocatorSnapshot* after) {
#if USING_CUDA
    try {
        auto* allocator = at::getHostAllocator(at::kCUDA);
        if (allocator == nullptr) {
            return absl::FailedPreconditionError("CUDA pinned-host allocator is unavailable");
        }

        const auto before_snapshot = snapshotAllocator(allocator->get_stats());
        allocator->empty_cache();
        const auto initial_flush_snapshot = snapshotAllocator(allocator->get_stats());
        const bool event_drain_retry      = !initial_flush_snapshot.empty();
        if (event_drain_retry) {
            // Pinned tensor destruction records allocator events after the
            // earlier sleep-path synchronization. Drain those events before
            // asking the allocator to release their backing blocks.
            RTP_LLM_LOG_WARNING(
                "[PinnedHost][verify] backing remains after initial flush; "
                "action=synchronize_and_retry initial={%s}",
                initial_flush_snapshot.debugString().c_str());
            const cudaError_t sync_error = cudaDeviceSynchronize();
            if (sync_error != cudaSuccess) {
                const std::string message = cudaGetErrorString(sync_error);
                (void)cudaGetLastError();
                if (before != nullptr) {
                    *before = before_snapshot;
                }
                if (after != nullptr) {
                    *after = initial_flush_snapshot;
                }
                return absl::InternalError(
                    "CUDA pinned-host event drain synchronize failed: " + message);
            }
            allocator->empty_cache();
        }
        const auto after_snapshot = snapshotAllocator(allocator->get_stats());
        if (before != nullptr) {
            *before = before_snapshot;
        }
        if (after != nullptr) {
            *after = after_snapshot;
        }

        RTP_LLM_LOG_INFO("[PinnedHost][verify] before={%s} initial_flush={%s} "
                         "event_drain_retry=%d after={%s}",
                         before_snapshot.debugString().c_str(),
                         initial_flush_snapshot.debugString().c_str(),
                         static_cast<int>(event_drain_retry),
                         after_snapshot.debugString().c_str());
        if (after_snapshot.hasStaleActiveCounters()) {
            RTP_LLM_LOG_WARNING(
                "[PinnedHost][verify] torch host allocator active counters are stale after all "
                "backing allocations were released; ignoring active-only counters: after={%s}",
                after_snapshot.debugString().c_str());
        }
        return verifyPinnedHostAllocatorSnapshotIsEmpty(after_snapshot);
    } catch (const std::exception& e) {
        return absl::InternalError(std::string("CUDA pinned-host allocator flush failed: ") + e.what());
    } catch (...) {
        return absl::InternalError("CUDA pinned-host allocator flush failed: unknown exception");
    }
#else
    if (before != nullptr) {
        *before = {};
    }
    if (after != nullptr) {
        *after = {};
    }
    RTP_LLM_LOG_INFO("[PinnedHost][verify] CUDA is disabled; skipping CUDA pinned-host allocator flush");
    return absl::OkStatus();
#endif
}

}  // namespace rtp_llm
