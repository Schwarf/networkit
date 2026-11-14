// networkit-format
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <networkit/graph/gpu/DeviceGraphCSR.hpp>

#include "DeviceStorageTestHelpers.hpp"

namespace NetworKit::GPU {
namespace {

using Test::read;
using Test::TestBackend;
using CSR = DeviceGraphCSR<TestBackend, uint32_t, uint64_t, double>;

TEST(DeviceGraphCSRGTest, testDimensionsAndEmptyGraph) {
    TestBackend::Context context;
    CSR graph(context, 4, 7);
    EXPECT_EQ(graph.numberOfNodes(), 4u);
    EXPECT_EQ(graph.numberOfEntries(), 7u);
    EXPECT_EQ(graph.rowOffsets().size(), 5u);
    EXPECT_EQ(graph.columnIndices().size(), 7u);
    EXPECT_EQ(graph.weights().size(), 7u);

    CSR empty(context, 0, 0);
    EXPECT_EQ(empty.numberOfNodes(), 0u);
    EXPECT_EQ(empty.numberOfEntries(), 0u);
    EXPECT_EQ(empty.rowOffsets().size(), 1u);
    EXPECT_EQ(empty.columnIndices().size(), 0u);
    EXPECT_EQ(empty.weights().size(), 0u);
}

TEST(DeviceGraphCSRGTest, testCommitAllBuffers) {
    TestBackend::Context context;
    CSR graph(context, 1, 1);
    graph.rowOffsets().hostWrite()[0] = 0;
    graph.rowOffsets().hostWrite()[1] = 1;
    graph.columnIndices().hostWrite()[0] = 0;
    graph.weights().hostWrite()[0] = 2.5;
    graph.commitHostWrites();
    const auto &view = graph;
    EXPECT_EQ(read(view.rowOffsets()), (std::vector<uint64_t>{0, 1}));
    EXPECT_EQ(read(view.columnIndices()), (std::vector<uint32_t>{0}));
    EXPECT_EQ(read(view.weights()), (std::vector<double>{2.5}));
    EXPECT_EQ(context.commits, 3u);
}

TEST(DeviceGraphCSRGTest, testRejectCapacityOverflowBeforeAllocation) {
    // NetworKit requires a 64-bit platform, so uint32_t limits can be exceeded.
    static_assert(std::numeric_limits<std::size_t>::digits > 32);
    using SmallCSR = DeviceGraphCSR<TestBackend, uint32_t, uint32_t, float>;
    TestBackend::Context context;
    const auto beyond32Bits = std::size_t{std::numeric_limits<uint32_t>::max()} + 1;
    EXPECT_THROW((SmallCSR(context, beyond32Bits + 1, 0)), std::length_error);
    EXPECT_THROW((SmallCSR(context, 1, beyond32Bits)), std::length_error);
    EXPECT_THROW((CSR(context, std::numeric_limits<std::size_t>::max(), 0)), std::length_error);
    EXPECT_EQ(context.allocationAttempts, 0u);
}

TEST(DeviceGraphCSRGTest, testMovesTransferAllAllocations) {
    static_assert(!std::is_copy_constructible_v<CSR>);
    static_assert(!std::is_copy_assignable_v<CSR>);
    static_assert(std::is_nothrow_move_constructible_v<CSR>);
    static_assert(std::is_nothrow_move_assignable_v<CSR>);
    TestBackend::Context context;
    {
        CSR source(context, 2, 1);
        const auto offsets = source.rowOffsets().nativeHandle();
        const auto columns = source.columnIndices().nativeHandle();
        const auto weights = source.weights().nativeHandle();

        CSR moved(std::move(source));
        EXPECT_EQ(source.numberOfNodes(), 0u);
        EXPECT_EQ(source.numberOfEntries(), 0u);
        EXPECT_EQ(context.liveAllocations, 3u);

        CSR target(context, 3, 2);
        target = std::move(moved);
        EXPECT_EQ(moved.numberOfNodes(), 0u);
        EXPECT_EQ(moved.numberOfEntries(), 0u);
        EXPECT_EQ(target.numberOfNodes(), 2u);
        EXPECT_EQ(target.numberOfEntries(), 1u);
        EXPECT_EQ(target.rowOffsets().nativeHandle(), offsets);
        EXPECT_EQ(target.columnIndices().nativeHandle(), columns);
        EXPECT_EQ(target.weights().nativeHandle(), weights);
        EXPECT_EQ(context.liveAllocations, 3u);
        EXPECT_EQ(context.releases, 3u);
    }
    EXPECT_EQ(context.liveAllocations, 0u);
    EXPECT_EQ(context.releases, 6u);
}

TEST(DeviceGraphCSRGTest, testPartialConstructionReleasesEarlierBuffers) {
    for (std::size_t failure = 1; failure <= 3; ++failure) {
        SCOPED_TRACE(failure);
        TestBackend::Context context;
        context.failOnAllocation = failure;
        EXPECT_THROW((CSR(context, 2, 1)), std::bad_alloc);
        EXPECT_EQ(context.allocationAttempts, failure);
        EXPECT_EQ(context.liveAllocations, 0u);
        EXPECT_EQ(context.releases, failure - 1);
    }
}

} // namespace
} // namespace NetworKit::GPU
