// networkit-format
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <networkit/graph/Graph.hpp>
#include <networkit/graph/gpu/DeviceGraphCSRBuilder.hpp>

#include "DeviceStorageTestHelpers.hpp"

namespace NetworKit::GPU {
namespace {

using Test::read;
using Test::TestBackend;

TEST(DeviceGraphCSRBuilderGTest, testDirectedOrientationAndRemovedNodes) {
    Graph source(4, true, true);
    source.removeNode(1);
    source.addEdge(0, 2, 2.5);
    source.addEdge(2, 3, -1.);
    source.addEdge(3, 3, 4.);
    TestBackend::Context context;

    auto outgoing = buildDeviceGraphCSR<TestBackend, uint32_t, uint64_t, double>(context, source);
    EXPECT_TRUE(outgoing.directed);
    EXPECT_EQ(outgoing.adjacency, CSRAdjacency::OUTGOING);
    EXPECT_EQ(outgoing.originalNodeIds, (std::vector<node>{0, 2, 3}));
    EXPECT_EQ(read(outgoing.graph.rowOffsets()), (std::vector<uint64_t>{0, 1, 2, 3}));
    EXPECT_EQ(read(outgoing.graph.columnIndices()), (std::vector<uint32_t>{1, 2, 2}));
    EXPECT_EQ(read(outgoing.graph.weights()), (std::vector<double>{2.5, -1., 4.}));

    auto incoming = buildDeviceGraphCSR<TestBackend, uint32_t, uint32_t, float>(
        context, source, CSRAdjacency::INCOMING);
    EXPECT_EQ(incoming.adjacency, CSRAdjacency::INCOMING);
    EXPECT_EQ(read(incoming.graph.rowOffsets()), (std::vector<uint32_t>{0, 0, 1, 3}));
    EXPECT_EQ(read(incoming.graph.columnIndices()), (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(read(incoming.graph.weights()), (std::vector<float>{2.5f, -1.f, 4.f}));
    EXPECT_EQ(context.uploads, 6u);
}

TEST(DeviceGraphCSRBuilderGTest, testUndirectedUnweightedSelfLoopAndIsolatedNode) {
    Graph source(3);
    source.addEdge(0, 1);
    source.addEdge(1, 1);
    TestBackend::Context context;
    for (auto adjacency : {CSRAdjacency::OUTGOING, CSRAdjacency::INCOMING}) {
        auto result =
            buildDeviceGraphCSR<TestBackend, uint64_t, uint32_t, float>(context, source, adjacency);
        EXPECT_FALSE(result.directed);
        EXPECT_EQ(result.graph.numberOfNodes(), 3u);
        EXPECT_EQ(result.graph.numberOfEntries(), 3u);
        EXPECT_EQ(read(result.graph.rowOffsets()), (std::vector<uint32_t>{0, 1, 3, 3}));
        EXPECT_EQ(read(result.graph.columnIndices()), (std::vector<uint64_t>{1, 0, 1}));
        EXPECT_EQ(read(result.graph.weights()), (std::vector<float>{1.f, 1.f, 1.f}));
    }
}

TEST(DeviceGraphCSRBuilderGTest, testEmptyGraph) {
    Graph source(2);
    source.removeNode(0);
    source.removeNode(1);
    TestBackend::Context context;
    auto result = buildDeviceGraphCSR<TestBackend, uint32_t, uint64_t, double>(context, source);
    EXPECT_EQ(result.graph.numberOfNodes(), 0u);
    EXPECT_EQ(result.graph.numberOfEntries(), 0u);
    EXPECT_TRUE(result.originalNodeIds.empty());
    EXPECT_EQ(read(result.graph.rowOffsets()), (std::vector<uint64_t>{0}));
    EXPECT_EQ(context.uploads, 3u);
}

TEST(DeviceGraphCSRBuilderGTest, testDifferentSourceTypes) {
    AdjListGraph<uint32_t, float> source(2, true, true);
    source.addEdge(1, 0, 3.f);
    TestBackend::Context context;
    auto result = buildDeviceGraphCSR<TestBackend, uint64_t, uint64_t, int>(context, source);
    EXPECT_EQ(result.originalNodeIds, (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(read(result.graph.rowOffsets()), (std::vector<uint64_t>{0, 0, 1}));
    EXPECT_EQ(read(result.graph.weights()), (std::vector<int>{3}));
}

TEST(DeviceGraphCSRBuilderGTest, testRejectUnrepresentableWeights) {
    Graph source(2, true, true);
    TestBackend::Context context;
    for (const auto weight : {0.5, -1., std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN(), 1e100}) {
        if (source.hasEdge(0, 1)) {
            source.setWeight(0, 1, weight);
        } else {
            source.addEdge(0, 1, weight);
        }
        EXPECT_THROW(
            (buildDeviceGraphCSR<TestBackend, uint32_t, uint32_t, uint32_t>(context, source)),
            std::invalid_argument);
    }
    EXPECT_EQ(context.uploads, 0u);
}

} // namespace
} // namespace NetworKit::GPU
