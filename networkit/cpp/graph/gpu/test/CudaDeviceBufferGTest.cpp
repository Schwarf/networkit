// networkit-format
#include <array>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <networkit/graph/gpu/CudaBackend.hpp>

#include "BackendStorageTestHelpers.hpp"

namespace NetworKit::GPU {
namespace {

TEST(CudaDeviceBufferGTest, testStorageAndGraphBuilder) {
    if (!CudaBackend::isAvailable()) {
        GTEST_SKIP() << "No usable CUDA device";
    }
    CudaBackend::Context context;
    ::NetworKit::GPU::Test::checkDeviceBuffer<CudaBackend, uint32_t>(context);
    ::NetworKit::GPU::Test::checkDeviceBuffer<CudaBackend, uint64_t>(context);
    ::NetworKit::GPU::Test::checkDeviceBuffer<CudaBackend, float>(context);
    ::NetworKit::GPU::Test::checkDeviceBuffer<CudaBackend, double>(context);
    ::NetworKit::GPU::Test::checkDeviceGraphBuilder<CudaBackend, double>(context);
}

TEST(CudaDeviceBufferGTest, testAllocationOverflow) {
    if (!CudaBackend::isAvailable()) {
        GTEST_SKIP() << "No usable CUDA device";
    }
    CudaBackend::Context context;
    EXPECT_THROW((DeviceBuffer<CudaBackend, double>(
                     context, std::numeric_limits<std::size_t>::max() / sizeof(double) + 1)),
                 std::length_error);
}

TEST(CudaDeviceBufferGTest, testNativeAllocationAndDeviceWrite) {
    if (!CudaBackend::isAvailable()) {
        GTEST_SKIP() << "No usable CUDA device";
    }
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    CudaBackend::Context context(device);
    DeviceBuffer<CudaBackend, float> buffer(context, 2);
    buffer.getWritableBuffer()[0] = 2.f;
    buffer.getWritableBuffer()[1] = 7.f;
    buffer.uploadToDevice();

    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, buffer.nativeHandle()), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
    ASSERT_EQ(cudaMemset(buffer.nativeHandle(), 0, 2 * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::array<float, 2> result{};
    buffer.readBack(result);
    EXPECT_EQ(result, (std::array<float, 2>{0.f, 0.f}));
    // Device writes do not modify the independent host preparation allocation.
    EXPECT_EQ(buffer.getWritableBuffer()[0], 2.f);
}

TEST(CudaDeviceBufferGTest, testDeviceSelectionIsRestored) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        GTEST_SKIP() << "Requires two CUDA devices";
    }
    int previous = 0;
    ASSERT_EQ(cudaGetDevice(&previous), cudaSuccess);
    CudaBackend::Context context((previous + 1) % devices);
    ::NetworKit::GPU::Test::checkDeviceBuffer<CudaBackend, float>(context);
    int current = 0;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, previous);
}

} // namespace
} // namespace NetworKit::GPU
