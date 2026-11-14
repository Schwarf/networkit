// networkit-format
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <networkit/graph/Graph.hpp>
#include <networkit/graph/gpu/DeviceGraphCSRBuilder.hpp>

#include "DeviceStorageTestHelpers.hpp"

namespace NetworKit::GPU {
namespace {

using Test::read;
using Test::TestBackend;

template <class DeviceEdgeWeightT, class HostEdgeWeightT>
void expectWeightErrorDetails(HostEdgeWeightT weight, std::string_view reason,
                              std::string_view value, std::string_view limit = {}) {
    for (const auto adjacency : {CSRAdjacency::OUTGOING, CSRAdjacency::INCOMING}) {
        // Error messages must identify original node IDs rather than compact ones.
        AdjListGraph<node, HostEdgeWeightT> source(5, true, true);
        source.removeNode(1);
        source.addEdge(2, 4, weight);
        TestBackend::Context context;
        try {
            (void)buildDeviceGraphCSR<TestBackend, uint32_t, uint64_t, DeviceEdgeWeightT>(
                context, source, adjacency);
            FAIL() << "Expected a weight conversion error";
        } catch (const std::invalid_argument &error) {
            const std::string message = error.what();
            SCOPED_TRACE(message);
            EXPECT_NE(message.find(reason), std::string::npos);
            EXPECT_NE(message.find(value), std::string::npos);
            EXPECT_NE(message.find(limit), std::string::npos);
            EXPECT_NE(message.find("source edge 2 -> 4"), std::string::npos);
            EXPECT_NE(message.find(adjacency == CSRAdjacency::OUTGOING ? "CSR row 1 (outgoing)"
                                                                       : "CSR row 3 (incoming)"),
                      std::string::npos);
        }
        EXPECT_EQ(context.uploads, 0u);
        EXPECT_EQ(context.liveAllocations, 0u);
    }
}

TEST(DeviceGraphCSRBuilderGTest, testIntegerWeightErrorDetails) {
    expectWeightErrorDetails<uint32_t>(int64_t{-1}, "below DeviceEdgeWeightT minimum", "weight -1",
                                       "minimum 0");
    expectWeightErrorDetails<int32_t>(int64_t{-2147483649LL}, "below DeviceEdgeWeightT minimum",
                                      "weight -2147483649", "minimum -2147483648");
    expectWeightErrorDetails<int32_t>(int64_t{2147483648LL}, "exceeds DeviceEdgeWeightT maximum",
                                      "weight 2147483648", "maximum 2147483647");
    expectWeightErrorDetails<int64_t>(std::numeric_limits<uint64_t>::max(),
                                      "exceeds DeviceEdgeWeightT maximum",
                                      "weight 18446744073709551615", "maximum 9223372036854775807");
}

TEST(DeviceGraphCSRBuilderGTest, testFloatingPointWeightErrorDetails) {
    expectWeightErrorDetails<float>(std::numeric_limits<double>::quiet_NaN(), "non-finite weight",
                                    "a finite weight is required");
    expectWeightErrorDetails<float>(std::numeric_limits<double>::infinity(), "non-finite weight",
                                    "a finite weight is required");
    expectWeightErrorDetails<float>(-std::numeric_limits<double>::infinity(), "non-finite weight",
                                    "a finite weight is required");
    expectWeightErrorDetails<int32_t>(-2147483649.0, "below integer DeviceEdgeWeightT minimum",
                                      "weight -2147483649", "minimum -2147483648");
    expectWeightErrorDetails<int32_t>(2147483648.0, "exclusive upper bound", "weight 2147483648",
                                      "bound 2147483648");
    expectWeightErrorDetails<int32_t>(0.5, "fractional weight 0.5", "without truncation");
    expectWeightErrorDetails<float>(1e100, "exceeds floating-point DeviceEdgeWeightT maximum",
                                    "weight ", "maximum 3.40282347e+38");
    expectWeightErrorDetails<float>(-1e100, "below floating-point DeviceEdgeWeightT minimum",
                                    "weight -", "minimum -3.40282347e+38");
}

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

TEST(DeviceGraphCSRBuilderGTest, testRejectInvalidAdjacencyBeforeAllocation) {
    Graph source(2);
    TestBackend::Context context;
    EXPECT_THROW((buildDeviceGraphCSR<TestBackend, uint32_t, uint32_t, float>(
                     context, source, static_cast<CSRAdjacency>(-1))),
                 std::invalid_argument);
    EXPECT_EQ(context.allocationAttempts, 0u);
    EXPECT_EQ(context.uploads, 0u);
}

template <class DeviceEdgeWeightT>
class DeviceGraphCSRBuilderWeightGTest : public ::testing::Test {
protected:
    template <class HostEdgeWeightT>
    void expectAccepted(HostEdgeWeightT weight, DeviceEdgeWeightT expected) {
        for (const auto adjacency : {CSRAdjacency::OUTGOING, CSRAdjacency::INCOMING}) {
            AdjListGraph<uint32_t, HostEdgeWeightT> source(2, true, true);
            source.addEdge(0, 1, weight);
            TestBackend::Context context;
            {
                const auto result =
                    buildDeviceGraphCSR<TestBackend, uint32_t, uint64_t, DeviceEdgeWeightT>(
                        context, source, adjacency);
                const auto weights = ::NetworKit::GPU::Test::read(result.graph.weights());
                ASSERT_EQ(weights.size(), 1u);
                EXPECT_EQ(weights[0], expected);
                EXPECT_EQ(context.uploads, 3u);
            }
            EXPECT_EQ(context.liveAllocations, 0u);
        }
    }

    template <class HostEdgeWeightT>
    void expectRejected(HostEdgeWeightT weight) {
        for (const auto adjacency : {CSRAdjacency::OUTGOING, CSRAdjacency::INCOMING}) {
            AdjListGraph<uint32_t, HostEdgeWeightT> source(2, true, true);
            source.addEdge(0, 1, weight);
            TestBackend::Context context;
            EXPECT_THROW((buildDeviceGraphCSR<TestBackend, uint32_t, uint64_t, DeviceEdgeWeightT>(
                             context, source, adjacency)),
                         std::invalid_argument);
            EXPECT_EQ(context.uploads, 0u);
            EXPECT_EQ(context.liveAllocations, 0u);
            EXPECT_EQ(context.releases, 3u);
        }
    }
};

using DeviceWeightTypes = ::testing::Types<float, double, int32_t, uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(DeviceGraphCSRBuilderWeightGTest, DeviceWeightTypes);

TYPED_TEST(DeviceGraphCSRBuilderWeightGTest, testRejectNonfiniteWeights) {
    this->expectRejected(std::numeric_limits<double>::quiet_NaN());
    this->expectRejected(std::numeric_limits<double>::infinity());
    this->expectRejected(-std::numeric_limits<double>::infinity());
}

TYPED_TEST(DeviceGraphCSRBuilderWeightGTest, testAcceptExactDestinationLimits) {
    using Limits = std::numeric_limits<TypeParam>;
    // Keep host weights in the exact integer type: converting max() to double
    // first would round the 64-bit integer limits and weaken this test.
    this->expectAccepted(Limits::lowest(), Limits::lowest());
    this->expectAccepted(Limits::max(), Limits::max());
    this->expectAccepted(TypeParam{0}, TypeParam{0});
}

TYPED_TEST(DeviceGraphCSRBuilderWeightGTest, testFractionalConversionPolicy) {
    if constexpr (std::is_integral_v<TypeParam>) {
        this->expectRejected(0.5);
        this->expectRejected(-0.5);
        this->expectAccepted(42.0, TypeParam{42});
    } else {
        this->expectAccepted(0.5, TypeParam{0.5});
        this->expectAccepted(-0.5, TypeParam{-0.5});
        // Inexact floating-point narrowing is intentionally allowed.
        const auto weight = 1.L / 3.L;
        this->expectAccepted(weight, static_cast<TypeParam>(weight));
    }
}

TYPED_TEST(DeviceGraphCSRBuilderWeightGTest, testValuesOutsideDestinationRange) {
    if constexpr (std::is_integral_v<TypeParam>) {
        const auto upper = std::ldexp(1.0, std::numeric_limits<TypeParam>::digits);
        const auto lower = std::is_signed_v<TypeParam> ? -upper : 0.0;
        this->expectRejected(upper);
        this->expectRejected(std::nextafter(lower, -std::numeric_limits<double>::infinity()));
        this->expectAccepted(lower, std::numeric_limits<TypeParam>::lowest());
        // Largest integral double below the exclusive upper limit.
        const auto inside = std::floor(std::nextafter(upper, 0.0));
        this->expectAccepted(inside, static_cast<TypeParam>(inside));
    } else {
        // long double can expose finite values just outside float/double bounds.
        // On platforms where it has no wider range, infinity is tested above.
        const auto above =
            std::nextafter(static_cast<long double>(std::numeric_limits<TypeParam>::max()),
                           std::numeric_limits<long double>::infinity());
        const auto below =
            std::nextafter(static_cast<long double>(std::numeric_limits<TypeParam>::lowest()),
                           -std::numeric_limits<long double>::infinity());
        if (std::isfinite(above)) {
            this->expectRejected(above);
        }
        if (std::isfinite(below)) {
            this->expectRejected(below);
        }
        if constexpr (std::is_same_v<TypeParam, float>) {
            this->expectRejected(std::numeric_limits<double>::max());
            this->expectRejected(std::numeric_limits<double>::lowest());
            this->expectAccepted(static_cast<double>(std::numeric_limits<float>::max()),
                                 std::numeric_limits<float>::max());
            this->expectAccepted(static_cast<double>(std::numeric_limits<float>::lowest()),
                                 std::numeric_limits<float>::lowest());
        }
    }
}

TYPED_TEST(DeviceGraphCSRBuilderWeightGTest, testSignedUnsignedAndNarrowingConversions) {
    this->expectAccepted(int64_t{42}, TypeParam{42});
    this->expectAccepted(uint64_t{42}, TypeParam{42});
    if constexpr (std::is_integral_v<TypeParam>) {
        if constexpr (std::is_signed_v<TypeParam>) {
            this->expectAccepted(int64_t{-1}, TypeParam{-1});
            this->expectRejected(std::numeric_limits<uint64_t>::max());
            if constexpr (sizeof(TypeParam) < sizeof(int64_t)) {
                this->expectRejected(std::numeric_limits<int64_t>::max());
                this->expectRejected(std::numeric_limits<int64_t>::lowest());
            }
        } else {
            this->expectRejected(int64_t{-1});
            if constexpr (sizeof(TypeParam) < sizeof(uint64_t)) {
                this->expectRejected(std::numeric_limits<uint64_t>::max());
                this->expectRejected(std::numeric_limits<int64_t>::max());
            } else {
                this->expectAccepted(std::numeric_limits<int64_t>::max(),
                                     static_cast<TypeParam>(std::numeric_limits<int64_t>::max()));
            }
        }
    } else {
        // Rounding is allowed when integer host weights become floating-point.
        this->expectAccepted(std::numeric_limits<uint64_t>::max(),
                             static_cast<TypeParam>(std::numeric_limits<uint64_t>::max()));
        this->expectAccepted(std::numeric_limits<int64_t>::lowest(),
                             static_cast<TypeParam>(std::numeric_limits<int64_t>::lowest()));
    }
}

template <class HostValueT>
class DeviceGraphCSRBuilderCallbackGTest : public ::testing::Test {};

using MatchingNodeAndWeightTypes = ::testing::Types<uint32_t, uint64_t>;
TYPED_TEST_SUITE(DeviceGraphCSRBuilderCallbackGTest, MatchingNodeAndWeightTypes);

TYPED_TEST(DeviceGraphCSRBuilderCallbackGTest, testIdenticalNodeAndWeightTypes) {
    const auto maxWeight = std::numeric_limits<TypeParam>::max();
    for (const bool indexed : {false, true}) {
        SCOPED_TRACE(indexed);
        AdjListGraph<TypeParam, TypeParam> source(3, true, true);
        source.addEdge(0, 2, TypeParam{17});
        source.addEdge(2, 1, maxWeight);
        source.addEdge(1, 0, TypeParam{0});
        if (indexed) {
            source.indexEdges();
        }
        for (const auto adjacency : {CSRAdjacency::OUTGOING, CSRAdjacency::INCOMING}) {
            TestBackend::Context context;
            const auto result = buildDeviceGraphCSR<TestBackend, uint32_t, uint64_t, TypeParam>(
                context, source, adjacency);
            EXPECT_EQ(read(result.graph.rowOffsets()), (std::vector<uint64_t>{0, 1, 2, 3}));
            if (adjacency == CSRAdjacency::OUTGOING) {
                EXPECT_EQ(read(result.graph.columnIndices()), (std::vector<uint32_t>{2, 0, 1}));
                EXPECT_EQ(read(result.graph.weights()),
                          (std::vector<TypeParam>{TypeParam{17}, TypeParam{0}, maxWeight}));
            } else {
                EXPECT_EQ(read(result.graph.columnIndices()), (std::vector<uint32_t>{1, 2, 0}));
                EXPECT_EQ(read(result.graph.weights()),
                          (std::vector<TypeParam>{TypeParam{0}, maxWeight, TypeParam{17}}));
            }
        }
    }
}

template <class Node, class Offset>
struct IndexTypes {
    using DeviceNodeT = Node;
    using DeviceOffsetT = Offset;
};

template <class Configuration>
class DeviceGraphCSRBuilderCapacityGTest : public ::testing::Test {};

using DeviceIndexTypes =
    ::testing::Types<IndexTypes<uint32_t, uint32_t>, IndexTypes<uint32_t, uint64_t>,
                     IndexTypes<uint64_t, uint32_t>, IndexTypes<uint64_t, uint64_t>>;
TYPED_TEST_SUITE(DeviceGraphCSRBuilderCapacityGTest, DeviceIndexTypes);

TYPED_TEST(DeviceGraphCSRBuilderCapacityGTest, testNodeCountLimitsWithoutGraphAllocation) {
    using DeviceNodeT = typename TypeParam::DeviceNodeT;
    EXPECT_EQ(CSRBuilderDetail::checkedNodeCount<DeviceNodeT>(0, uint64_t{0}), 0u);
    EXPECT_EQ(CSRBuilderDetail::checkedNodeCount<DeviceNodeT>(3, uint64_t{10}), 3u);
    EXPECT_THROW((CSRBuilderDetail::checkedNodeCount<DeviceNodeT>(
                     std::numeric_limits<std::size_t>::max(), uint64_t{0})),
                 std::length_error);
    // An invalid signed host ID bound must not wrap to an enormous allocation.
    EXPECT_THROW((CSRBuilderDetail::checkedNodeCount<DeviceNodeT>(0, int64_t{-1})),
                 std::length_error);
    if constexpr (sizeof(DeviceNodeT) == sizeof(uint32_t)) {
        const auto maxNodes = uint64_t{std::numeric_limits<uint32_t>::max()} + 1;
        EXPECT_EQ(CSRBuilderDetail::checkedNodeCount<DeviceNodeT>(maxNodes, maxNodes), maxNodes);
        EXPECT_THROW((CSRBuilderDetail::checkedNodeCount<DeviceNodeT>(maxNodes + 1, maxNodes + 1)),
                     std::length_error);
    } else {
        const auto maxNodes = std::numeric_limits<std::size_t>::max() - 1;
        EXPECT_EQ(CSRBuilderDetail::checkedNodeCount<DeviceNodeT>(maxNodes, maxNodes), maxNodes);
    }
}

TYPED_TEST(DeviceGraphCSRBuilderCapacityGTest, testEntryCountLimitsWithoutGraphAllocation) {
    using DeviceOffsetT = typename TypeParam::DeviceOffsetT;
    const auto limit = static_cast<std::size_t>(std::numeric_limits<DeviceOffsetT>::max());
    EXPECT_EQ(CSRBuilderDetail::checkedEntryCount<DeviceOffsetT>(0, 0), 0u);
    EXPECT_EQ(CSRBuilderDetail::checkedEntryCount<DeviceOffsetT>(2, 3), 5u);
    EXPECT_EQ(CSRBuilderDetail::checkedEntryCount<DeviceOffsetT>(limit - 1, 1), limit);
    EXPECT_THROW((CSRBuilderDetail::checkedEntryCount<DeviceOffsetT>(limit, 1)), std::length_error);
    if constexpr (sizeof(DeviceOffsetT) == sizeof(uint32_t)) {
        EXPECT_THROW((CSRBuilderDetail::checkedEntryCount<DeviceOffsetT>(limit + 1, 0)),
                     std::length_error);
        EXPECT_THROW((CSRBuilderDetail::checkedEntryCount<DeviceOffsetT>(0, uint64_t{limit} + 1)),
                     std::length_error);
    }
}

} // namespace
} // namespace NetworKit::GPU
