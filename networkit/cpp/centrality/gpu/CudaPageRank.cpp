// networkit-format
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/graph/gpu/CudaBackend.hpp>
#include <networkit/graph/gpu/DeviceBuffer.hpp>
#include <networkit/graph/gpu/DeviceGraphCSR.hpp>

#include "CudaPageRankKernels.hpp"
#include "PageRankBackends.hpp"
#include "PageRankMatrix.hpp"

namespace NetworKit {
namespace GPU {

namespace {

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
    }
}

bool hasDevice() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess) {
        // Clears the error, so that it does not show up in later calls.
        cudaGetLastError();
        return false;
    }
    return devices > 0;
}

} // namespace

std::optional<PageRankRun> cudaPageRank(const Graph &G, const std::vector<uint32_t> &rows,
                                        double damp, double tol, count maxIterations) {
    Aux::SignalHandler handler;
    const count n = G.numberOfNodes();

    if (!hasDevice()) {
        return std::nullopt;
    }

    // Preserve the caller's current device for allocations, memory queries,
    // native CUDA operations, and kernel launches throughout this run.
    int device = 0;
    check(cudaGetDevice(&device));
    CudaBackend::Context context(device);
    CudaDetail::DeviceGuard deviceGuard(context.nativeDevice());

    // Count actual adjacency entries, including each undirected self-loop once,
    // before allocating the CSR buffers so the memory preflight stays intact.
    count numEntries = 0;
    G.forNodes([&](node u) { numEntries += G.degreeIn(u); });

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    check(cudaMemGetInfo(&freeBytes, &totalBytes));
    const count bytes = (n + 1) * sizeof(uint64_t)
                        + numEntries * (sizeof(uint32_t) + sizeof(double)) + 2 * n * sizeof(double)
                        + sizeof(double);
    if (bytes > freeBytes) {
        return std::nullopt;
    }

    DeviceGraphCSR<CudaBackend, uint32_t, uint64_t, double> matrix(context, n, numEntries);
    auto rowBegins = matrix.rowOffsets().getWritableBuffer();
    auto columns = matrix.columnIndices().getWritableBuffer();
    auto coefficients = matrix.weights().getWritableBuffer();
    fillRowBegins(G, rows, rowBegins.data());
    // Keep PageRank's incoming adjacency and damp * weight / source-degree
    // coefficients; the generic graph builder would upload raw edge weights.
    fillEntries(G, damp, rows, rowBegins.data(), columns.data(), coefficients.data());

    std::array deviceScores{DeviceBuffer<CudaBackend, double>(context, n),
                            DeviceBuffer<CudaBackend, double>(context, n)};
    DeviceBuffer<CudaBackend, double> deviceSquaredChange(context, 1);

    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device));
    INFO("GPUPageRank: running on CUDA, device ", properties.name);

    matrix.uploadToDevice();
    auto initialScores = deviceScores[0].getWritableBuffer();
    std::fill(initialScores.begin(), initialScores.end(), 1. / static_cast<double>(n));
    deviceScores[0].uploadToDevice();

    const double teleport = (1. - damp) / static_cast<double>(n);
    count iterations = 0;
    bool isConverged = false;
    do {
        handler.assureRunning();
        check(cudaMemset(deviceSquaredChange.nativeHandle(), 0, sizeof(double)));
        CudaPageRankKernels::iterate(
            matrix.rowOffsets().nativeHandle(), matrix.columnIndices().nativeHandle(),
            matrix.weights().nativeHandle(), teleport, deviceScores[iterations % 2].nativeHandle(),
            deviceScores[(iterations + 1) % 2].nativeHandle(), static_cast<uint32_t>(n),
            deviceSquaredChange.nativeHandle());
        check(cudaGetLastError());

        // Waits for the kernel to finish.
        double squaredChange = 0.;
        deviceSquaredChange.readBack(std::span<double>{&squaredChange, 1});

        ++iterations;
        isConverged = iterations >= maxIterations || std::sqrt(squaredChange) <= tol;
    } while (!isConverged);

    PageRankRun result;
    result.scores.resize(n);
    deviceScores[iterations % 2].readBack(result.scores);
    result.iterations = iterations;
    return result;
}

} // namespace GPU
} // namespace NetworKit
