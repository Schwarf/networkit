// networkit-format
#ifndef NETWORKIT_GRAPH_GPU_CUDA_BACKEND_HPP_
#define NETWORKIT_GRAPH_GPU_CUDA_BACKEND_HPP_

#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>

namespace NetworKit::GPU {
namespace CudaDetail {

inline void check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA storage: ") + cudaGetErrorString(error));
    }
}

// CUDA's current device is thread-local. Restore the caller's selection on exit.
class DeviceGuard {
public:
    explicit DeviceGuard(int device) {
        check(cudaGetDevice(&previous));
        if (device != previous) {
            check(cudaSetDevice(device));
            changed = true;
        }
    }
    ~DeviceGuard() {
        if (changed) {
            cudaSetDevice(previous);
        }
    }
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;

private:
    int previous = 0;
    bool changed = false;
};

struct AllocationDeleter {
    int device;

    void operator()(void *pointer) const noexcept {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) {
            return;
        }
        if (previous != device && cudaSetDevice(device) != cudaSuccess) {
            return;
        }
        cudaFree(pointer);
        if (previous != device) {
            cudaSetDevice(previous);
        }
    }
};

} // namespace CudaDetail

/**
 * CUDA storage with CPU staging memory and a separate device allocation.
 * Include only when CUDA is enabled; link against CUDA::cudart.
 * Kernel submission and completion remain the caller's responsibility.
 * Context selects a device ordinal, defaulting to zero. Allocation and transfer
 * operations temporarily select that device, then restore the thread's previous
 * selection. Native kernel launches must target the same device explicitly.
 */
struct CudaBackend {
    static bool isAvailable() noexcept {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    class Context {
    public:
        explicit Context(int device = 0) : device(device) {
            int count = 0;
            CudaDetail::check(cudaGetDeviceCount(&count));
            if (device < 0 || device >= count) {
                throw std::invalid_argument("CUDA storage: invalid device ordinal");
            }
        }
        int nativeDevice() const noexcept { return device; }

    private:
        int device;
    };

    template <class ValueType>
    static constexpr bool supportsValueType =
        (std::is_integral_v<ValueType> && !std::is_same_v<ValueType, bool>
         && sizeof(ValueType) <= 8)
        || std::is_same_v<ValueType, float> || std::is_same_v<ValueType, double>;

    template <class ValueType>
    class Storage {
        static_assert(supportsValueType<ValueType>, "Unsupported CUDA storage element type");

    public:
        Storage(Context &context, std::size_t count)
            : device(context.nativeDevice()), hostStagingBuffer(checkedCount(count)),
              allocation(nullptr, CudaDetail::AllocationDeleter{device}) {
            if (count != 0) {
                CudaDetail::DeviceGuard guard(device);
                void *pointer = nullptr;
                CudaDetail::check(cudaMalloc(&pointer, count * sizeof(ValueType)));
                allocation.reset(pointer);
            }
        }

        Storage(const Storage &) = delete;
        Storage &operator=(const Storage &) = delete;
        // Moving the allocation and staging owner leaves size() exactly zero.
        Storage(Storage &&) noexcept = default;
        Storage &operator=(Storage &&) noexcept = default;

        std::size_t size() const noexcept { return allocation ? hostStagingBuffer.size() : 0; }

        std::span<ValueType> getWritableBuffer() { return {hostStagingBuffer.data(), size()}; }

        void uploadToDevice() {
            if (size() != 0) {
                CudaDetail::DeviceGuard guard(device);
                CudaDetail::check(cudaMemcpy(allocation.get(), hostStagingBuffer.data(),
                                             size() * sizeof(ValueType), cudaMemcpyHostToDevice));
                // Pageable host-to-device copies may return before DMA has
                // finished. Fulfill the buffer's synchronous upload contract.
                CudaDetail::check(cudaStreamSynchronize(nullptr));
            }
        }

        void readBack(std::span<ValueType> destination) const {
            if (destination.size() != size()) {
                throw std::invalid_argument("CUDA storage: destination size mismatch");
            }
            if (size() != 0) {
                CudaDetail::DeviceGuard guard(device);
                CudaDetail::check(cudaMemcpy(destination.data(), allocation.get(),
                                             size() * sizeof(ValueType), cudaMemcpyDeviceToHost));
            }
        }

        ValueType *nativeHandle() noexcept { return static_cast<ValueType *>(allocation.get()); }

        const ValueType *nativeHandle() const noexcept {
            return static_cast<const ValueType *>(allocation.get());
        }

    private:
        static std::size_t checkedCount(std::size_t count) {
            if (count > std::numeric_limits<std::size_t>::max() / sizeof(ValueType)) {
                throw std::length_error("CUDA storage: allocation size overflow");
            }
            return count;
        }

        int device;
        std::vector<ValueType> hostStagingBuffer;
        std::unique_ptr<void, CudaDetail::AllocationDeleter> allocation;
    };
};

} // namespace NetworKit::GPU

#endif // NETWORKIT_GRAPH_GPU_CUDA_BACKEND_HPP_
