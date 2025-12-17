/*  SSSPRunner.cu
 *
 *  Created on: 17.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */
#include <networkit/gpu/CudaHelpers.hpp>
#include <networkit/gpu/SSSPKernels.cuh>
#include <networkit/gpu/SSSPRunner.hpp>

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace NetworKit::GPU {

template <>
SsspRunner<float>::SsspRunner(const DeviceGraphCSR<float> &deviceGraph) {
    view = deviceGraph.view();
    numberOfNodes = view.numberOfNodes;

    if (!view.weights) {
        throw std::runtime_error("SsspRunner<float>: view.weights is null (no device weights).");
    }
    if (!view.rowPointer || !view.columnIndices) {
        throw std::runtime_error("SsspRunner<float>: CSR pointers are null.");
    }

    cudaCheck(cudaMalloc(&deviceDistances, static_cast<std::size_t>(numberOfNodes) * sizeof(float)),
              "cudaMalloc deviceDistances");

    cudaCheck(cudaMalloc(&frontierPing, static_cast<std::size_t>(numberOfNodes) * sizeof(node_t)),
              "cudaMalloc frontierPing");

    cudaCheck(cudaMalloc(&frontierPong, static_cast<std::size_t>(numberOfNodes) * sizeof(node_t)),
              "cudaMalloc frontierPong");

    cudaCheck(cudaMalloc(&deviceFrontierCount, sizeof(std::uint32_t)),
              "cudaMalloc deviceFrontierCount");

    cudaCheck(cudaMalloc(&deviceQueued, static_cast<std::size_t>(numberOfNodes) * sizeof(int)),
              "cudaMalloc deviceQueued");
}

template <>
SsspRunner<float>::~SsspRunner() {
    if (deviceDistances)
        cudaCheck(cudaFree(deviceDistances), "cudaFree deviceDistances");
    if (frontierPing)
        cudaCheck(cudaFree(frontierPing), "cudaFree frontierPing");
    if (frontierPong)
        cudaCheck(cudaFree(frontierPong), "cudaFree frontierPong");
    if (deviceFrontierCount)
        cudaCheck(cudaFree(deviceFrontierCount), "cudaFree deviceFrontierCount");
    if (deviceQueued)
        cudaCheck(cudaFree(deviceQueued), "cudaFree deviceQueued");

}

template <>
SsspRunner<float> &SsspRunner<float>::operator=(SsspRunner &&other) noexcept {
    if (this == &other)
        return *this;

    deviceDistances = other.deviceDistances;
    other.deviceDistances = nullptr;
    frontierPing = other.frontierPing;
    other.frontierPing = nullptr;
    frontierPong = other.frontierPong;
    other.frontierPong = nullptr;
    deviceFrontierCount = other.deviceFrontierCount;
    other.deviceFrontierCount = nullptr;

    view = other.view;
    numberOfNodes = other.numberOfNodes;

    other.view = {};
    other.numberOfNodes = 0;

    return *this;
}

template <>
SsspRunner<float>::SsspRunner(SsspRunner &&other) noexcept {
    *this = std::move(other);
}

template <>
std::vector<float> SsspRunner<float>::run(node_t source) {
    if (source >= numberOfNodes) {
        throw std::runtime_error("SsspRunner<float>::run: src out of range.");
    }

    constexpr int BLOCK = 256;
    const int initGrid = (static_cast<int>(numberOfNodes) + BLOCK - 1) / BLOCK;

    // Device-side initialization
    initDistancesAndFrontierKernel<node_t, float><<<initGrid, BLOCK>>>(
        deviceDistances, numberOfNodes, source, frontierPing, deviceFrontierCount, deviceQueued);
    cudaCheck(cudaGetLastError(), "init kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "init kernel sync");

    std::uint32_t currentCount = 0;
    cudaCheck(cudaMemcpy(&currentCount, deviceFrontierCount, sizeof(std::uint32_t),
                         cudaMemcpyDeviceToHost),
              "read initial frontierCount");

    node_t *currentFrontier = frontierPing;
    node_t *nextFrontier = frontierPong;

    while (currentCount != 0) {
        std::uint32_t zero = 0;
        cudaCheck(
            cudaMemcpy(deviceFrontierCount, &zero, sizeof(std::uint32_t), cudaMemcpyHostToDevice),
            "reset deviceFrontierCount");

        const int grid = (static_cast<int>(currentCount) + BLOCK - 1) / BLOCK;

        relaxFromFrontierKernel<index_t, node_t>
            <<<grid, BLOCK>>>(view.rowPointer, view.columnIndices, view.weights, deviceDistances,
                              currentFrontier, currentCount, nextFrontier, deviceFrontierCount, deviceQueued);
        cudaCheck(cudaGetLastError(), "kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "kernel sync");

        cudaCheck(cudaMemcpy(&currentCount, deviceFrontierCount, sizeof(std::uint32_t),
                             cudaMemcpyDeviceToHost),
                  "read deviceFrontierCount");

        std::swap(currentFrontier, nextFrontier);
    }
    std::vector<float> hostDistances(static_cast<std::size_t>(numberOfNodes));
    cudaCheck(cudaMemcpy(hostDistances.data(), deviceDistances,
                         hostDistances.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy distances back");

    return hostDistances;
}

template class SsspRunner<float>;

} // namespace NetworKit::GPU
