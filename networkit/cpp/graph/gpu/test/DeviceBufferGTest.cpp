// networkit-format
#include <array>
#include <concepts>
#include <span>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "DeviceStorageTestHelpers.hpp"

namespace NetworKit::GPU {
namespace {

using Test::read;
using Test::TestBackend;
using DeviceBuffer = DeviceBuffer<TestBackend, float>;
using Storage = TestBackend::Storage<float>;

struct WrongSpan : Storage {
    std::span<double> getWritableBuffer();
};
struct MissingUpload : Storage {
    void uploadToDevice() = delete;
};
struct NonConstReadBack : Storage {
    void readBack(std::span<float>);
};
struct NonConstHandle : Storage {
    float *nativeHandle();
};
struct WrongSize : Storage {
    int size() const;
};
struct ThrowingDestructor : Storage {
    ~ThrowingDestructor() noexcept(false);
};
struct CopyableStorage {
    std::size_t size() const;
    std::span<float> getWritableBuffer();
    void uploadToDevice();
    void readBack(std::span<float>) const;
    float *nativeHandle();
    const float *nativeHandle() const;
};

TEST(DeviceBufferGTest, testStorageContract) {
    static_assert(DeviceBufferStorage<Storage, float>);
    static_assert(std::constructible_from<Storage, TestBackend::Context &, std::size_t>);
    static_assert(!DeviceBufferStorage<WrongSpan, float>);
    static_assert(!DeviceBufferStorage<MissingUpload, float>);
    static_assert(!DeviceBufferStorage<NonConstReadBack, float>);
    static_assert(!DeviceBufferStorage<NonConstHandle, float>);
    static_assert(!DeviceBufferStorage<WrongSize, float>);
    static_assert(!DeviceBufferStorage<ThrowingDestructor, float>);
    static_assert(!DeviceBufferStorage<CopyableStorage, float>);
    static_assert(!std::is_copy_constructible_v<DeviceBuffer>);
    static_assert(!std::is_copy_assignable_v<DeviceBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer>);
    static_assert(std::is_nothrow_move_assignable_v<DeviceBuffer>);
    static_assert(
        std::is_same_v<decltype(std::declval<const DeviceBuffer &>().nativeHandle()), const float *>);
    SUCCEED();
}

TEST(DeviceBufferGTest, testUploadAndReadBack) {
    TestBackend::Context context;
    DeviceBuffer buffer(context, 2);
    auto preparation = buffer.getWritableBuffer();
    preparation[0] = 1.5f;
    preparation[1] = -2.f;
    EXPECT_EQ(read(buffer), (std::vector<float>{0.f, 0.f}));
    buffer.uploadToDevice();
    EXPECT_EQ(read(buffer), (std::vector<float>{1.5f, -2.f}));
    EXPECT_EQ(context.uploads, 1u);
}

TEST(DeviceBufferGTest, testRejectMismatchedReadBackBeforeCallingBackend) {
    TestBackend::Context context;
    DeviceBuffer buffer(context, 2);
    std::array<float, 1> tooSmall{};
    std::array<float, 3> tooLarge{};
    EXPECT_THROW(buffer.readBack(tooSmall), std::invalid_argument);
    EXPECT_THROW(buffer.readBack(tooLarge), std::invalid_argument);
    EXPECT_EQ(context.readBackCalls, 0u);
}

TEST(DeviceBufferGTest, testMovesTransferOwnershipAndReleaseReplacedAllocation) {
    TestBackend::Context context;
    {
        DeviceBuffer source(context, 1);
        source.getWritableBuffer()[0] = 7.f;
        source.uploadToDevice();
        const auto handle = source.nativeHandle();

        DeviceBuffer moved(std::move(source));
        EXPECT_EQ(source.size(), 0u);
        EXPECT_EQ(moved.nativeHandle(), handle);
        EXPECT_EQ(context.liveAllocations, 1u);

        DeviceBuffer target(context, 3);
        target = std::move(moved);
        EXPECT_EQ(moved.size(), 0u);
        EXPECT_EQ(target.size(), 1u);
        EXPECT_EQ(target.nativeHandle(), handle);
        EXPECT_EQ(read(target), (std::vector<float>{7.f}));
        EXPECT_EQ(context.liveAllocations, 1u);
        EXPECT_EQ(context.releases, 1u);
    }
    EXPECT_EQ(context.liveAllocations, 0u);
    EXPECT_EQ(context.releases, 2u);
}

TEST(DeviceBufferGTest, testEmptyBuffer) {
    TestBackend::Context context;
    DeviceBuffer buffer(context, 0);
    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_TRUE(buffer.getWritableBuffer().empty());
    buffer.uploadToDevice();
    EXPECT_TRUE(read(buffer).empty());
}

TEST(DeviceBufferGTest, testAllocationFailure) {
    TestBackend::Context context;
    context.failOnAllocation = 1;
    EXPECT_THROW((DeviceBuffer(context, 1)), std::bad_alloc);
    EXPECT_EQ(context.liveAllocations, 0u);
    EXPECT_EQ(context.releases, 0u);
}

} // namespace
} // namespace NetworKit::GPU
