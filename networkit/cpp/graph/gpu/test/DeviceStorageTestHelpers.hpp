// networkit-format
#ifndef NETWORKIT_GRAPH_GPU_TEST_DEVICE_STORAGE_TEST_HELPERS_HPP_
#define NETWORKIT_GRAPH_GPU_TEST_DEVICE_STORAGE_TEST_HELPERS_HPP_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
#include <span>
#include <vector>

#include <networkit/graph/gpu/DeviceBuffer.hpp>

namespace NetworKit::GPU::Test {

// Separate staging and committed memory, with ownership and failure tracking.
struct TestBackend {
    struct Context {
        std::size_t commits = 0;
        std::size_t readBackCalls = 0;
        std::size_t allocationAttempts = 0;
        std::size_t liveAllocations = 0;
        std::size_t releases = 0;
        // One-based allocation attempt on which to fail; zero disables failure.
        std::size_t failOnAllocation = 0;
    };

    template <class T>
    class Storage {
        struct Allocation {
            Context &context;
            std::vector<T> preparation;
            std::vector<T> committed;

            Allocation(Context &context, std::size_t n)
                : context(context), preparation(n), committed(n) {
                ++context.liveAllocations;
            }
            ~Allocation() {
                --context.liveAllocations;
                ++context.releases;
            }
        };

        std::unique_ptr<Allocation> allocation;

    public:
        Storage(Context &context, std::size_t n) {
            ++context.allocationAttempts;
            if (context.failOnAllocation == context.allocationAttempts) {
                throw std::bad_alloc();
            }
            allocation = std::make_unique<Allocation>(context, n);
        }
        Storage(const Storage &) = delete;
        Storage &operator=(const Storage &) = delete;
        Storage(Storage &&) noexcept = default;
        Storage &operator=(Storage &&) noexcept = default;

        std::size_t size() const { return allocation ? allocation->preparation.size() : 0; }
        std::span<T> hostWrite() { return allocation->preparation; }
        void commitHostWrites() {
            allocation->committed = allocation->preparation;
            ++allocation->context.commits;
        }
        void readBack(std::span<T> destination) const {
            std::copy(allocation->committed.begin(), allocation->committed.end(),
                      destination.begin());
            ++allocation->context.readBackCalls;
        }
        T *nativeHandle() { return allocation ? allocation->committed.data() : nullptr; }
        const T *nativeHandle() const {
            return allocation ? allocation->committed.data() : nullptr;
        }
    };
};

template <class T>
std::vector<T> read(const DeviceBuffer<TestBackend, T> &buffer) {
    std::vector<T> result(buffer.size());
    buffer.readBack(result);
    return result;
}

} // namespace NetworKit::GPU::Test

#endif // NETWORKIT_GRAPH_GPU_TEST_DEVICE_STORAGE_TEST_HELPERS_HPP_
