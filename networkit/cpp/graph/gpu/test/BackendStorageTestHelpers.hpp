// networkit-format
#ifndef NETWORKIT_GRAPH_GPU_TEST_BACKEND_STORAGE_TEST_HELPERS_HPP_
#define NETWORKIT_GRAPH_GPU_TEST_BACKEND_STORAGE_TEST_HELPERS_HPP_

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <networkit/graph/Graph.hpp>
#include <networkit/graph/gpu/DeviceGraphCSRBuilder.hpp>

namespace NetworKit::GPU::Test {

template <class Backend, class ValueType>
void checkDeviceBuffer(typename Backend::Context &context) {
    using DeviceBuffer = DeviceBuffer<Backend, ValueType>;
    static_assert(DeviceBufferStorage<typename DeviceBuffer::Storage, ValueType>);
    static_assert(!std::is_copy_constructible_v<DeviceBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer>);
    static_assert(std::is_nothrow_move_assignable_v<DeviceBuffer>);

    DeviceBuffer source(context, 2);
    auto hostStagingBuffer = source.getWritableBuffer();
    ASSERT_EQ(hostStagingBuffer.size(), 2u);
    hostStagingBuffer[0] = static_cast<ValueType>(2);
    hostStagingBuffer[1] = static_cast<ValueType>(7);
    source.uploadToDevice();
    const auto handle = source.nativeHandle();
    ASSERT_NE(handle, nullptr);

    DeviceBuffer moved(std::move(source));
    EXPECT_EQ(source.size(), 0u);
    EXPECT_TRUE(source.getWritableBuffer().empty());
    EXPECT_EQ(source.nativeHandle(), nullptr);
    EXPECT_EQ(moved.nativeHandle(), handle);
    // Verify the destination
    EXPECT_EQ(moved.size(), 2u);
    EXPECT_EQ(moved.nativeHandle(), handle);

    DeviceBuffer target(context, 1);
    target = std::move(moved);
    EXPECT_EQ(moved.size(), 0u);
    EXPECT_EQ(moved.nativeHandle(), nullptr);
    EXPECT_EQ(target.nativeHandle(), handle);
    const auto &readOnly = target;
    std::array<ValueType, 2> result{};
    readOnly.readBack(result);
    EXPECT_EQ(result[0], static_cast<ValueType>(2));
    EXPECT_EQ(result[1], static_cast<ValueType>(7));

    // Exercise another write/upload cycle after ownership transfer.
    target.getWritableBuffer()[0] = static_cast<ValueType>(9);
    target.uploadToDevice();
    readOnly.readBack(result);
    EXPECT_EQ(result[0], static_cast<ValueType>(9));
    EXPECT_EQ(result[1], static_cast<ValueType>(7));

    std::array<ValueType, 1> wrongSize{};
    EXPECT_THROW(target.readBack(wrongSize), std::invalid_argument);

    DeviceBuffer empty(context, 0);
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_TRUE(empty.getWritableBuffer().empty());
    EXPECT_EQ(empty.nativeHandle(), nullptr);
    empty.uploadToDevice();
    EXPECT_NO_THROW(empty.readBack({}));
}

template <class Backend, class WeightType>
void checkDeviceGraphBuilder(typename Backend::Context &context) {
    Graph source(4, true, true);
    source.removeNode(1);
    source.addEdge(0, 2, 2.5);
    source.addEdge(2, 3, 7.);

    const auto result = buildDeviceGraphCSR<Backend, uint32_t, uint64_t, WeightType>(
        context, source, CSRAdjacency::INCOMING);
    EXPECT_EQ(result.originalNodeIds, (std::vector<node>{0, 2, 3}));
    EXPECT_TRUE(result.directed);
    EXPECT_EQ(result.adjacency, CSRAdjacency::INCOMING);

    std::array<uint64_t, 4> offsets{};
    std::array<uint32_t, 2> columns{};
    std::array<WeightType, 2> weights{};
    result.graph.rowOffsets().readBack(offsets);
    result.graph.columnIndices().readBack(columns);
    result.graph.weights().readBack(weights);
    EXPECT_EQ(offsets, (std::array<uint64_t, 4>{0, 0, 1, 2}));
    EXPECT_EQ(columns, (std::array<uint32_t, 2>{0, 1}));
    EXPECT_EQ(weights[0], static_cast<WeightType>(2.5));
    EXPECT_EQ(weights[1], static_cast<WeightType>(7.));
}

} // namespace NetworKit::GPU::Test

#endif // NETWORKIT_GRAPH_GPU_TEST_BACKEND_STORAGE_TEST_HELPERS_HPP_
