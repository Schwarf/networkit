// networkit-format
#ifndef NETWORKIT_GRAPH_GPU_DEVICE_BUFFER_HPP_
#define NETWORKIT_GRAPH_GPU_DEVICE_BUFFER_HPP_

#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace NetworKit::GPU {

/**
 * Storage API required by DeviceBuffer. Storage owns its allocation exclusively and
 * must support moving and non-throwing destruction. Synchronization, lifetime,
 * and moved-from size guarantees are behavioral requirements documented below;
 * this concept only checks types and available operations.
 */
template <class S, class ValueType>
concept DeviceBufferStorage =
    std::movable<S> && !std::is_copy_constructible_v<S> && !std::is_copy_assignable_v<S>
    && requires(S &storage, const S &readOnly, std::span<ValueType> destination) {
           { readOnly.size() } -> std::same_as<std::size_t>;
           { storage.getWritableBuffer() } -> std::same_as<std::span<ValueType>>;
           { storage.uploadToDevice() } -> std::same_as<void>;
           { readOnly.readBack(destination) } -> std::same_as<void>;
           storage.nativeHandle();
           readOnly.nativeHandle();
       };

/**
 * Owns a typed allocation through a backend-specific storage policy.
 *
 * Backend must provide Context and Storage<ValueType>. Storage<ValueType> must be movable and
 * non-copyable, release its resources without throwing, and implement:
 *
 * - Storage(Context &, std::size_t numberOfElements)
 * - size() const: number of elements, zero after a move
 * - getWritableBuffer(): std::span<ValueType> over CPU-accessible preparation memory
 * - uploadToDevice(): make preparation writes available for GPU execution
 * - readBack(std::span<ValueType>) const: copy completed results into host memory
 * - nativeHandle() / nativeHandle() const: backend-specific kernel binding
 *
 * For CUDA, preparation memory can be host staging storage; for Metal it can
 * be shared buffer storage. Neither getWritableBuffer() nor nativeHandle() transfers
 * ownership. Their results must not outlive this allocation.
 *
 * Transfers are synchronous. The caller must ensure GPU execution has completed
 * before accessing results or modifying storage. Context must outlive the buffer.
 */
template <class Backend, class ValueType>
class DeviceBuffer {
    static_assert(std::is_arithmetic_v<ValueType> && !std::is_same_v<ValueType, bool>
                      && !std::is_const_v<ValueType> && !std::is_volatile_v<ValueType>,
                  "GPU buffers require an unqualified numeric element type other than bool");

public:
    using value_type = ValueType;
    using Storage = Backend::template Storage<ValueType>;

    static_assert(DeviceBufferStorage<Storage, ValueType>,
                  "Backend storage must implement the move-only DeviceBuffer storage API");
    static_assert(std::constructible_from<Storage, typename Backend::Context &, std::size_t>,
                  "Backend storage must be constructible from Context & and an element count");

    DeviceBuffer(Backend::Context &context, std::size_t numberOfElements)
        : storage(context, numberOfElements) {}

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&) = default;
    DeviceBuffer &operator=(DeviceBuffer &&) = default;

    std::size_t size() const { return storage.size(); }

    /**
     * Returns a non-owning, CPU-accessible view of preparation memory. This
     * neither copies data nor uploads it. Fill the view, then call
     * uploadToDevice(). The view must not outlive the underlying allocation.
     */
    std::span<ValueType> getWritableBuffer() { return storage.getWritableBuffer(); }

    /**
     * Makes host preparation writes available to GPU execution. Backends with
     * shared memory may skip the copy. The caller must finish previous GPU work
     * before modifying preparation memory or uploading it.
     */
    void uploadToDevice() { storage.uploadToDevice(); }

    void readBack(std::span<ValueType> destination) const {
        if (destination.size() != size()) {
            throw std::invalid_argument(
                "DeviceBuffer::readBack: destination size must match buffer size");
        }
        storage.readBack(destination);
    }

    /**
     * Returns a borrowed backend-native handle to GPU-accessible storage:
     * ValueType * for CUDA, or MTL::Buffer * for Metal. Backend execution code
     * uses it to pass the allocation to kernels. A CUDA device pointer must not
     * be dereferenced on the CPU.
     *
     * Ownership stays with this buffer; do not free the handle or use it after
     * its allocation is released. This call performs no upload or synchronization.
     * The caller must ensure GPU work completes before CPU access or destruction.
     */
    decltype(auto) nativeHandle() { return storage.nativeHandle(); }
    decltype(auto) nativeHandle() const { return storage.nativeHandle(); }

private:
    Storage storage;
};

} // namespace NetworKit::GPU

#endif // NETWORKIT_GRAPH_GPU_DEVICE_BUFFER_HPP_
