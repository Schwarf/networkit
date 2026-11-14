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
template <class S, class T>
concept DeviceBufferStorage =
    std::movable<S> && !std::is_copy_constructible_v<S> && !std::is_copy_assignable_v<S>
    && requires(S &storage, const S &readOnly, std::span<T> destination) {
           { readOnly.size() } -> std::same_as<std::size_t>;
           { storage.hostWrite() } -> std::same_as<std::span<T>>;
           { storage.commitHostWrites() } -> std::same_as<void>;
           { readOnly.readBack(destination) } -> std::same_as<void>;
           storage.nativeHandle();
           readOnly.nativeHandle();
       };

/**
 * Owns a typed allocation through a backend-specific storage policy.
 *
 * Backend must provide Context and Storage<T>. Storage<T> must be movable and
 * non-copyable, release its resources without throwing, and implement:
 *
 * - Storage(Context &, std::size_t numberOfElements)
 * - size() const: number of elements, zero after a move
 * - hostWrite(): std::span<T> over CPU-accessible preparation memory
 * - commitHostWrites(): make preparation writes available for GPU execution
 * - readBack(std::span<T>) const: copy completed results into host memory
 * - nativeHandle() / nativeHandle() const: backend-specific kernel binding
 *
 * For CUDA, preparation memory can be host staging storage; for Metal it can
 * be shared buffer storage. Neither hostWrite() nor nativeHandle() transfers
 * ownership. Their results must not outlive this allocation.
 *
 * Transfers are synchronous. The caller must ensure GPU execution has completed
 * before accessing results or modifying storage. Context must outlive the buffer.
 */
template <class Backend, class T>
class DeviceBuffer {
    static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool> && !std::is_const_v<T>
                      && !std::is_volatile_v<T>,
                  "GPU buffers require an unqualified numeric element type other than bool");

public:
    using value_type = T;
    using Storage = Backend::template Storage<T>;

    static_assert(DeviceBufferStorage<Storage, T>,
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

    std::span<T> hostWrite() { return storage.hostWrite(); }

    void commitHostWrites() { storage.commitHostWrites(); }

    void readBack(std::span<T> destination) const {
        if (destination.size() != size()) {
            throw std::invalid_argument(
                "DeviceBuffer::readBack: destination size must match buffer size");
        }
        storage.readBack(destination);
    }

    decltype(auto) nativeHandle() { return storage.nativeHandle(); }
    decltype(auto) nativeHandle() const { return storage.nativeHandle(); }

private:
    Storage storage;
};

} // namespace NetworKit::GPU

#endif // NETWORKIT_GRAPH_GPU_DEVICE_BUFFER_HPP_
