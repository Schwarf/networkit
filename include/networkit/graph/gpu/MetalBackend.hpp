// networkit-format
#ifndef NETWORKIT_GRAPH_GPU_METAL_BACKEND_HPP_
#define NETWORKIT_GRAPH_GPU_METAL_BACKEND_HPP_

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

namespace NetworKit::GPU {

/**
 * Metal shared-buffer storage. CPU preparation writes directly into the GPU
 * buffer; uploadToDevice() needs no copy. The caller must complete GPU work
 * before reading or modifying that memory. Native handles are borrowed Metal
 * buffer objects, including for const storage; GPU access permissions are set
 * by kernel bindings, not by the C++ const qualifier.
 * Supports integer scalars up to 64 bits and float storage; kernels must also
 * support the chosen types (64-bit integers require an appropriate MSL version).
 * Double-precision floating-point storage is deliberately unsupported.
 */
struct MetalBackend {
    static bool isAvailable();

    class Context {
    public:
        Context();
        MTL::Device *nativeDevice() const noexcept { return device.get(); }

    private:
        NS::SharedPtr<MTL::Device> device;
    };

    template <class ValueType>
    static constexpr bool supportsValueType =
        (std::is_integral_v<ValueType> && !std::is_same_v<ValueType, bool>
         && sizeof(ValueType) <= 8)
        || std::is_same_v<ValueType, float>;

    template <class ValueType>
    class Storage {
        static_assert(supportsValueType<ValueType>, "Unsupported Metal storage element type");

    public:
        Storage(Context &context, std::size_t count) : numberOfElements(count) {
            if (count > std::numeric_limits<std::size_t>::max() / sizeof(ValueType)) {
                throw std::length_error("Metal storage: allocation size overflow");
            }
            if (count == 0) {
                return;
            }
            const auto bytes = count * sizeof(ValueType);
            if (bytes > context.nativeDevice()->maxBufferLength()) {
                throw std::length_error("Metal storage: allocation exceeds maximum buffer length");
            }
            buffer = NS::TransferPtr(
                context.nativeDevice()->newBuffer(bytes, MTL::ResourceStorageModeShared));
            if (!buffer) {
                throw std::runtime_error("Metal storage: could not allocate shared buffer");
            }
        }

        Storage(const Storage &) = delete;
        Storage &operator=(const Storage &) = delete;

        Storage(Storage &&other) noexcept
            : numberOfElements(std::exchange(other.numberOfElements, 0)),
              buffer(std::move(other.buffer)) {}

        Storage &operator=(Storage &&other) noexcept {
            if (this != &other) {
                buffer = std::move(other.buffer);
                numberOfElements = std::exchange(other.numberOfElements, 0);
            }
            return *this;
        }

        std::size_t size() const noexcept { return numberOfElements; }

        std::span<ValueType> getWritableBuffer() {
            return {buffer ? static_cast<ValueType *>(buffer->contents()) : nullptr,
                    numberOfElements};
        }

        // Shared storage already contains the preparation data. This does not
        // wait for outstanding commands; synchronization belongs to the caller.
        void uploadToDevice() noexcept {}

        void readBack(std::span<ValueType> destination) const {
            if (destination.size() != numberOfElements) {
                throw std::invalid_argument("Metal storage: destination size mismatch");
            }
            if (numberOfElements != 0) {
                const auto *values = static_cast<const ValueType *>(buffer->contents());
                std::copy_n(values, numberOfElements, destination.begin());
            }
        }

        MTL::Buffer *nativeHandle() noexcept { return buffer.get(); }

        MTL::Buffer *nativeHandle() const noexcept { return buffer.get(); }

    private:
        std::size_t numberOfElements;
        NS::SharedPtr<MTL::Buffer> buffer;
    };
};

} // namespace NetworKit::GPU

#endif // NETWORKIT_GRAPH_GPU_METAL_BACKEND_HPP_
