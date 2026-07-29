#include "rtp_llm/cpp/engine_base/sleep/PinnedHostMemoryVerifier.h"

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(PinnedHostMemoryVerifierTest, AcceptsZeroSnapshot) {
    EXPECT_TRUE(verifyPinnedHostAllocatorSnapshotIsEmpty({}).ok());
}

TEST(PinnedHostMemoryVerifierTest, RejectsEachRemainingBackingAllocationCounter) {
    PinnedHostAllocatorSnapshot snapshot;

    snapshot.active_bytes = 1;
    EXPECT_TRUE(verifyPinnedHostAllocatorSnapshotIsEmpty(snapshot).ok());
    EXPECT_TRUE(snapshot.hasStaleActiveCounters());

    snapshot                 = {};
    snapshot.active_requests = 1;
    EXPECT_TRUE(verifyPinnedHostAllocatorSnapshotIsEmpty(snapshot).ok());
    EXPECT_TRUE(snapshot.hasStaleActiveCounters());

    snapshot             = {};
    snapshot.allocations = 1;
    EXPECT_EQ(verifyPinnedHostAllocatorSnapshotIsEmpty(snapshot).code(), absl::StatusCode::kFailedPrecondition);

    snapshot                 = {};
    snapshot.allocated_bytes = 1;
    EXPECT_EQ(verifyPinnedHostAllocatorSnapshotIsEmpty(snapshot).code(), absl::StatusCode::kFailedPrecondition);
}

TEST(PinnedHostMemoryVerifierTest, AcceptsStaleActiveCountersAfterBackingRelease) {
    const PinnedHostAllocatorSnapshot snapshot{223132, 603, 0, 0};
    EXPECT_TRUE(snapshot.empty());
    EXPECT_TRUE(snapshot.hasStaleActiveCounters());
    EXPECT_TRUE(verifyPinnedHostAllocatorSnapshotIsEmpty(snapshot).ok());
}

TEST(PinnedHostMemoryVerifierTest, ReportsAllRemainingCounters) {
    const PinnedHostAllocatorSnapshot snapshot{1, 2, 3, 4};
    const auto                        status = verifyPinnedHostAllocatorSnapshotIsEmpty(snapshot);

    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("active_bytes=1"), std::string::npos);
    EXPECT_NE(status.message().find("active_requests=2"), std::string::npos);
    EXPECT_NE(status.message().find("allocations=3"), std::string::npos);
    EXPECT_NE(status.message().find("allocated_bytes=4"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
