// networkit-format
#ifndef NETWORKIT_GRAPH_GPU_CUDA_BACKEND_HPP_
#define NETWORKIT_GRAPH_GPU_CUDA_BACKEND_HPP_

#include <array>
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

/** A CUDA failure preserving its runtime error code for deliberate recovery. */
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t error, const std::string &operation)
        : std::runtime_error("CUDA: " + operation + " failed [" + cudaGetErrorName(error)
                             + "]: " + cudaGetErrorString(error)),
          error(error) {}

    cudaError_t code() const noexcept { return error; }

private:
    cudaError_t error;
};

namespace CudaDetail {

inline void check(cudaError_t error, const char *operation = "runtime operation") {
    if (error != cudaSuccess) {
        throw CudaError(error, operation);
    }
}

// CUDA's current device is thread-local. Restore the caller's selection on exit.
class DeviceGuard {
public:
    explicit DeviceGuard(int device) {
        check(cudaGetDevice(&previous), "cudaGetDevice");
        if (device != previous) {
            check(cudaSetDevice(device), "cudaSetDevice");
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
    using Error = CudaError;
    using DeviceGuard = CudaDetail::DeviceGuard;

    struct AvailabilityInfo {
        int deviceCount = 0;
        cudaError_t error = cudaSuccess;

        bool isAvailable() const noexcept { return error == cudaSuccess && deviceCount > 0; }
    };

    /** Cached properties; each runner decides which capabilities it requires. */
    struct DeviceInfo {
        int ordinal = 0;
        std::string name;
        int computeMajor = 0;
        int computeMinor = 0;
        std::size_t totalMemoryBytes = 0;
        int maxThreadsPerBlock = 0;
        std::array<int, 3> maxBlockDimensions{};
        std::array<int, 3> maxGridDimensions{};
        std::size_t sharedMemoryPerBlockBytes = 0;
        int warpSize = 0;

        bool computeCapabilityAtLeast(int major, int minor) const noexcept {
            return computeMajor > major || (computeMajor == major && computeMinor >= minor);
        }
    };

    struct MemoryInfo {
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
    };

    /** Enumeration probe, not a guarantee of successful context creation or execution. */
    static AvailabilityInfo probeAvailability() noexcept {
        AvailabilityInfo result;
        result.error = cudaGetDeviceCount(&result.deviceCount);
        if (result.error != cudaSuccess) {
            result.deviceCount = 0;
        }
        return result;
    }

    static bool isAvailable() noexcept { return probeAvailability().isAvailable(); }

    /** Use for kernel-launch and other runner-side CUDA calls as well as storage operations. */
    static void check(cudaError_t error, const char *operation) {
        CudaDetail::check(error, operation);
    }

    /**
     * Selects a device, caches its properties, and verifies a runtime memory query
     * during construction. It neither reserves workspace nor guarantees kernel
     * compatibility. Construction and queries preserve the calling thread's device.
     */
    class Context {
    public:
        explicit Context(int device = 0) : device(device) {
            if (device < 0) {
                throw std::invalid_argument("CUDA context: negative device ordinal "
                                            + std::to_string(device));
            }
            const auto availability = probeAvailability();
            check(availability.error, "cudaGetDeviceCount");
            if (availability.deviceCount == 0) {
                throw Error(cudaErrorNoDevice, "create CUDA context");
            }
            if (device >= availability.deviceCount) {
                throw std::invalid_argument("CUDA context: device ordinal " + std::to_string(device)
                                            + " is outside [0, "
                                            + std::to_string(availability.deviceCount) + ")");
            }

            const auto guard = activate();
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
            info = {
                device,
                properties.name,
                properties.major,
                properties.minor,
                properties.totalGlobalMem,
                properties.maxThreadsPerBlock,
                {properties.maxThreadsDim[0], properties.maxThreadsDim[1],
                 properties.maxThreadsDim[2]},
                {properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]},
                properties.sharedMemPerBlock,
                properties.warpSize};
            // Verify access to the selected device's runtime context without
            // allocating workspace. Free memory is deliberately not cached.
            (void)memoryInfo();
        }

        const DeviceInfo &deviceInfo() const noexcept { return info; }

        /** A live snapshot only; later allocation can still fail. */
        MemoryInfo memoryInfo() const {
            const auto guard = activate();
            MemoryInfo memory;
            check(cudaMemGetInfo(&memory.freeBytes, &memory.totalBytes), "cudaMemGetInfo");
            return memory;
        }

        /** Keep this scope guard alive while issuing native CUDA calls. */
        [[nodiscard]] DeviceGuard activate() const { return DeviceGuard(device); }

        /** Wait for all work on this device, including work issued outside this context. */
        void synchronize() const {
            const auto guard = activate();
            check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }

        int nativeDevice() const noexcept { return device; }

    private:
        int device;
        DeviceInfo info;
    };

    // These are supported storage representations, not kernel capability checks.
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
                check(cudaMalloc(&pointer, count * sizeof(ValueType)), "cudaMalloc");
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
                check(cudaMemcpy(allocation.get(), hostStagingBuffer.data(),
                                 size() * sizeof(ValueType), cudaMemcpyHostToDevice),
                      "cudaMemcpy host-to-device");
                // Pageable host-to-device copies may return before DMA has
                // finished. Fulfill the buffer's synchronous upload contract.
                check(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize after upload");
            }
        }

        void readBack(std::span<ValueType> destination) const {
            if (destination.size() != size()) {
                throw std::invalid_argument("CUDA storage: destination size mismatch");
            }
            if (size() != 0) {
                CudaDetail::DeviceGuard guard(device);
                check(cudaMemcpy(destination.data(), allocation.get(), size() * sizeof(ValueType),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy device-to-host");
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
