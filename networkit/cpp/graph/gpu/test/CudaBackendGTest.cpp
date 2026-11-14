// networkit-format
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include <networkit/graph/gpu/CudaBackend.hpp>
#include <networkit/graph/gpu/DeviceBuffer.hpp>

namespace NetworKit::GPU {
namespace {

TEST(CudaBackendDiagnosticsGTest, testErrorPreservesCodeAndOperation) {
    EXPECT_NO_THROW(CudaBackend::check(cudaSuccess, "successful operation"));
    // Inject a status rather than exhausting device memory to test recovery.
    try {
        CudaBackend::check(cudaErrorMemoryAllocation, "test allocation");
        FAIL() << "Expected a CUDA error";
    } catch (const CudaBackend::Error &error) {
        EXPECT_EQ(error.code(), cudaErrorMemoryAllocation);
        const std::string message = error.what();
        EXPECT_NE(message.find("test allocation"), std::string::npos);
        EXPECT_NE(message.find(cudaGetErrorName(cudaErrorMemoryAllocation)), std::string::npos);
    }
}

TEST(CudaBackendDiagnosticsGTest, testAvailabilityRetainsRuntimeStatus) {
    const auto availability = CudaBackend::probeAvailability();
    if (availability.error != cudaSuccess) {
        EXPECT_EQ(availability.deviceCount, 0);
        EXPECT_FALSE(availability.isAvailable());
    } else {
        EXPECT_GE(availability.deviceCount, 0);
        EXPECT_EQ(availability.isAvailable(), availability.deviceCount > 0);
    }
    // An error must not become "available" merely because a count was supplied.
    const CudaBackend::AvailabilityInfo failed{1, cudaErrorInsufficientDriver};
    EXPECT_FALSE(failed.isAvailable());
}

TEST(CudaBackendDiagnosticsGTest, testNegativeOrdinalIsRejected) {
    // Negative input is rejected before querying the runtime, even without a GPU.
    EXPECT_THROW((CudaBackend::Context(-1)), std::invalid_argument);
}

TEST(CudaBackendDiagnosticsGTest, testCapabilityVersionComparison) {
    CudaBackend::DeviceInfo device;
    device.computeMajor = 8;
    device.computeMinor = 6;
    EXPECT_TRUE(device.computeCapabilityAtLeast(7, 9));
    EXPECT_TRUE(device.computeCapabilityAtLeast(8, 0));
    EXPECT_TRUE(device.computeCapabilityAtLeast(8, 6));
    EXPECT_FALSE(device.computeCapabilityAtLeast(8, 7));
    EXPECT_FALSE(device.computeCapabilityAtLeast(9, 0));
}

class CudaBackendGTest : public ::testing::Test {
protected:
    void SetUp() override {
        availability = CudaBackend::probeAvailability();
        if (!availability.isAvailable()) {
            GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorName(availability.error)
                         << ", enumerated devices: " << availability.deviceCount;
        }
        ASSERT_EQ(cudaGetDevice(&originalDevice), cudaSuccess);
        context.emplace(originalDevice);
    }

    CudaBackend::AvailabilityInfo availability;
    int originalDevice = 0;
    std::optional<CudaBackend::Context> context;
};

TEST_F(CudaBackendGTest, testCachedPropertiesMatchSelectedDevice) {
    cudaDeviceProp properties{};
    ASSERT_EQ(cudaGetDeviceProperties(&properties, originalDevice), cudaSuccess);
    const auto &device = context->deviceInfo();
    EXPECT_EQ(device.ordinal, originalDevice);
    EXPECT_EQ(context->nativeDevice(), originalDevice);
    EXPECT_EQ(device.name, properties.name);
    EXPECT_EQ(device.computeMajor, properties.major);
    EXPECT_EQ(device.computeMinor, properties.minor);
    EXPECT_EQ(device.totalMemoryBytes, properties.totalGlobalMem);
    EXPECT_EQ(device.maxThreadsPerBlock, properties.maxThreadsPerBlock);
    EXPECT_EQ(device.sharedMemoryPerBlockBytes, properties.sharedMemPerBlock);
    EXPECT_EQ(device.warpSize, properties.warpSize);
    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
        EXPECT_EQ(device.maxBlockDimensions[dimension], properties.maxThreadsDim[dimension]);
        EXPECT_EQ(device.maxGridDimensions[dimension], properties.maxGridSize[dimension]);
    }
    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, originalDevice);
}

TEST_F(CudaBackendGTest, testLiveMemoryQuery) {
    const auto before = context->memoryInfo();
    EXPECT_GT(before.totalBytes, 0u);
    EXPECT_LE(before.freeBytes, before.totalBytes);
    DeviceBuffer<CudaBackend, uint32_t> buffer(*context, 16);
    const auto after = context->memoryInfo();
    EXPECT_EQ(after.totalBytes, before.totalBytes);
    EXPECT_LE(after.freeBytes, after.totalBytes);
    // Do not assert a free-byte delta: allocators and other processes can change it.
    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, originalDevice);
}

TEST_F(CudaBackendGTest, testScopedActivation) {
    static_assert(!std::is_copy_constructible_v<CudaBackend::DeviceGuard>);
    static_assert(!std::is_move_constructible_v<CudaBackend::DeviceGuard>);
    {
        const auto guard = context->activate();
        int current = -1;
        ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
        EXPECT_EQ(current, context->nativeDevice());
    }
    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, originalDevice);
}

TEST_F(CudaBackendGTest, testSynchronizeDeviceWrites) {
    DeviceBuffer<CudaBackend, uint32_t> buffer(*context, 2);
    buffer.getWritableBuffer()[0] = 2;
    buffer.getWritableBuffer()[1] = 7;
    buffer.uploadToDevice();
    {
        const auto guard = context->activate();
        CudaBackend::check(cudaMemset(buffer.nativeHandle(), 0, 2 * sizeof(uint32_t)),
                           "cudaMemset");
    }
    context->synchronize();
    std::array<uint32_t, 2> result{};
    buffer.readBack(result);
    EXPECT_EQ(result, (std::array<uint32_t, 2>{0, 0}));
}

TEST_F(CudaBackendGTest, testRejectOrdinalBeyondDeviceCount) {
    EXPECT_THROW((CudaBackend::Context(availability.deviceCount)), std::invalid_argument);
}

TEST_F(CudaBackendGTest, testOtherDeviceAndExceptionRestoreCaller) {
    if (availability.deviceCount < 2) {
        GTEST_SKIP() << "Requires two CUDA devices";
    }
    const int otherDevice = (originalDevice + 1) % availability.deviceCount;
    CudaBackend::Context other(otherDevice);
    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, originalDevice);
    EXPECT_EQ(other.deviceInfo().ordinal, otherDevice);

    const auto memory = other.memoryInfo();
    EXPECT_GT(memory.totalBytes, 0u);
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, originalDevice);

    struct TestException {};
    try {
        const auto guard = other.activate();
        ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
        EXPECT_EQ(current, otherDevice);
        throw TestException{};
    } catch (const TestException &) {
    }
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, originalDevice);
}

} // namespace
} // namespace NetworKit::GPU
