// networkit-format
#include <array>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <networkit/graph/gpu/MetalBackend.hpp>

#include "BackendStorageTestHelpers.hpp"

namespace NetworKit::GPU {
namespace {

TEST(MetalDeviceBufferGTest, testStorageAndGraphBuilder) {
    static_assert(!MetalBackend::supportsValueType<double>);
    if (!MetalBackend::isAvailable()) {
        GTEST_SKIP() << "No usable Metal device";
    }
    MetalBackend::Context context;
    ::NetworKit::GPU::Test::checkDeviceBuffer<MetalBackend, uint32_t>(context);
    ::NetworKit::GPU::Test::checkDeviceBuffer<MetalBackend, uint64_t>(context);
    ::NetworKit::GPU::Test::checkDeviceBuffer<MetalBackend, float>(context);
    ::NetworKit::GPU::Test::checkDeviceGraphBuilder<MetalBackend, float>(context);
}

TEST(MetalDeviceBufferGTest, testPreparationUsesSharedBufferDirectly) {
    if (!MetalBackend::isAvailable()) {
        GTEST_SKIP() << "No usable Metal device";
    }
    MetalBackend::Context context;
    DeviceBuffer<MetalBackend, float> buffer(context, 2);
    EXPECT_EQ(buffer.nativeHandle()->storageMode(), MTL::StorageModeShared);
    EXPECT_EQ(buffer.getWritableBuffer().data(), buffer.nativeHandle()->contents());
}

TEST(MetalDeviceBufferGTest, testAllocationOverflow) {
    if (!MetalBackend::isAvailable()) {
        GTEST_SKIP() << "No usable Metal device";
    }
    MetalBackend::Context context;
    EXPECT_THROW((DeviceBuffer<MetalBackend, float>(
                     context, std::numeric_limits<std::size_t>::max() / sizeof(float) + 1)),
                 std::length_error);
}

TEST(MetalDeviceBufferGTest, testNativeDeviceWrite) {
    if (!MetalBackend::isAvailable()) {
        GTEST_SKIP() << "No usable Metal device";
    }
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MetalBackend::Context context;
    DeviceBuffer<MetalBackend, uint32_t> buffer(context, 2);
    buffer.getWritableBuffer()[0] = 2;
    buffer.getWritableBuffer()[1] = 7;
    buffer.uploadToDevice();

    const auto queue = NS::TransferPtr(context.nativeDevice()->newCommandQueue());
    ASSERT_TRUE(queue);
    auto *command = queue->commandBuffer();
    ASSERT_NE(command, nullptr);
    auto *encoder = command->blitCommandEncoder();
    ASSERT_NE(encoder, nullptr);
    encoder->fillBuffer(buffer.nativeHandle(), NS::Range::Make(0, 2 * sizeof(uint32_t)), 0);
    encoder->endEncoding();
    command->commit();
    command->waitUntilCompleted();
    ASSERT_EQ(command->status(), MTL::CommandBufferStatusCompleted);

    std::array<uint32_t, 2> result{};
    buffer.readBack(result);
    EXPECT_EQ(result, (std::array<uint32_t, 2>{0, 0}));
    // Shared preparation memory reflects the completed GPU writes directly.
    EXPECT_EQ(buffer.getWritableBuffer()[0], 0u);
}

} // namespace
} // namespace NetworKit::GPU
