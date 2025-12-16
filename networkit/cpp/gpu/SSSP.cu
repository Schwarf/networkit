/*  SSSP.cu
 *
 *  Created on: 16.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <vector>

#include <networkit/gpu/SSSP.hpp>

namespace NetworKit::GPU {

static inline void cudaCheck(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    }
}

// atomicMin for float (portable via CAS)
__device__ inline float atomicMinFloat(float *address, float value) {
    int *addressAsInt = reinterpret_cast<int *>(address);
    int oldValue = *addressAsInt;
    while (true) {
        float oldFloatValue = __int_as_float(oldValue);
        if (oldFloatValue <= value)
            return oldFloatValue;
        int assumed = oldValue;
        int desired = __float_as_int(value);
        oldValue = atomicCAS(addressAsInt, assumed, desired);
        if (oldValue == assumed)
            return oldFloatValue;
    }
}

template <typename IndexT, typename NodeT>
__global__ void relaxFromFrontierKernel(const IndexT *__restrict__ rowPointer,
                                        const NodeT *__restrict__ columnIndices,
                                        const float *__restrict__ weights, float *__restrict__ distances,
                                        const NodeT *__restrict__ currentlyActiveNodes,
                                        std::uint32_t hostCurrentlyActiveNodeCount, NodeT *__restrict__ nextActiveNodes,
                                        std::uint32_t *__restrict__ frontierOutSize) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hostCurrentlyActiveNodeCount)
        return;

    const NodeT u = currentlyActiveNodes[tid];
    const float du = distances[u];

    const IndexT begin = rowPointer[u];
    const IndexT end = rowPointer[u + 1];

    for (IndexT index = begin; index < end; ++index) {
        const NodeT v = columnIndices[index];
        const float newDistance = du + weights[index];

        float oldDistance = atomicMinFloat(&distances[v], newDistance);
        if (newDistance < oldDistance) {
            // push v to next frontier
            const std::uint32_t pos = atomicAdd(frontierOutSize, 1u);
            nextActiveNodes[pos] = v;
        }
    }
}

std::vector<float> ssspWorklistCuda(const DeviceGraph<float> &g, DeviceGraph<float>::node_t src) {
    if (!g.hasWeights) {
        throw std::runtime_error(
            "ssspWorklistCuda: DeviceGraph has no weights. Provide defaultWeight when building.");
    }

    using index = DeviceGraph<float>::index_t;
    using node = DeviceGraph<float>::node_t;

    const std::uint64_t n = static_cast<std::uint64_t>(g.rowPointer.size() - 1);
    const std::uint64_t m = static_cast<std::uint64_t>(g.columnIndices.size());

    if (src >= n)
        throw std::runtime_error("ssspWorklistCuda: src out of range.");

    // ---- allocate device arrays ----
    index *deviceRowPointer = nullptr;
    node *deviceColumnIndex = nullptr;
    float *deviceWeights = nullptr;
    float *deviceDistances = nullptr;

    node *currentlyActiveNodes = nullptr;
    node *nextActiveNodes = nullptr;
    std::uint32_t *currentlyActiveNodeCount = nullptr;

    cudaCheck(cudaMalloc(&deviceRowPointer, g.rowPointer.size() * sizeof(index)),
              "cudaMalloc deviceRowPointer");
    cudaCheck(cudaMalloc(&deviceColumnIndex, m * sizeof(node)), "cudaMalloc deviceColumnIndex");
    cudaCheck(cudaMalloc(&deviceWeights, m * sizeof(float)), "cudaMalloc deviceWeights");
    cudaCheck(cudaMalloc(&deviceDistances, n * sizeof(float)), "cudaMalloc deviceDistances");

    // Worst-case frontier can be large; allocate up to n (safe baseline).
    cudaCheck(cudaMalloc(&currentlyActiveNodes, n * sizeof(node)), "cudaMalloc deviceFrontierA");
    cudaCheck(cudaMalloc(&nextActiveNodes, n * sizeof(node)), "cudaMalloc deviceFrontierB");
    cudaCheck(cudaMalloc(&currentlyActiveNodeCount, sizeof(std::uint32_t)),
              "cudaMalloc deviceFrontierSize");

    cudaCheck(cudaMemcpy(deviceRowPointer, g.rowPointer.data(), g.rowPointer.size() * sizeof(index),
                         cudaMemcpyHostToDevice),
              "memcpy rowPointer");
    cudaCheck(cudaMemcpy(deviceColumnIndex, g.columnIndices.data(), m * sizeof(node),
                         cudaMemcpyHostToDevice),
              "memcpy columnIndex");
    cudaCheck(
        cudaMemcpy(deviceWeights, g.weights.data(), m * sizeof(float), cudaMemcpyHostToDevice),
        "memcpy weights");

    // init dist to INF, dist[src]=0
    std::vector<float> hostDistances(n, std::numeric_limits<float>::infinity());
    hostDistances[src] = 0.0f;
    cudaCheck(cudaMemcpy(deviceDistances, hostDistances.data(), n * sizeof(float),
                         cudaMemcpyHostToDevice),
              "memcpy distances init");

    // frontier = {src}
    cudaCheck(cudaMemcpy(currentlyActiveNodes, &src, sizeof(node), cudaMemcpyHostToDevice),
              "memcpy frontierA init");
    std::uint32_t hostCurrentlyActiveNodeCount = 1;
    cudaCheck(cudaMemcpy(currentlyActiveNodeCount, &hostCurrentlyActiveNodeCount, sizeof(std::uint32_t),
                         cudaMemcpyHostToDevice),
              "memcpy frontierSizeA init");

    node *swapNodesBufferA = currentlyActiveNodes;
    node *swapNodesBufferB = nextActiveNodes;

    // ---- main loop ----
    constexpr int BLOCK = 256;

    while (true) {
        // get current frontier size
        cudaCheck(cudaMemcpy(&hostCurrentlyActiveNodeCount, currentlyActiveNodeCount, sizeof(std::uint32_t),
                             cudaMemcpyDeviceToHost),
                  "memcpy frontierSize D2H");
        if (hostCurrentlyActiveNodeCount == 0)
            break;

        // reset out frontier size to 0
        std::uint32_t zero = 0;
        cudaCheck(
            cudaMemcpy(currentlyActiveNodeCount, &zero, sizeof(std::uint32_t), cudaMemcpyHostToDevice),
            "memcpy frontierSize reset");

        const int grid = (static_cast<int>(hostCurrentlyActiveNodeCount) + BLOCK - 1) / BLOCK;
        relaxFromFrontierKernel<index, node>
            <<<grid, BLOCK>>>(deviceRowPointer, deviceColumnIndex, deviceWeights, deviceDistances,
                              swapNodesBufferA, hostCurrentlyActiveNodeCount, swapNodesBufferB, currentlyActiveNodeCount);
        cudaCheck(cudaGetLastError(), "kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "kernel sync");

        // swap buffers
        std::swap(swapNodesBufferA, swapNodesBufferB);
    }

    // copy distances back
    cudaCheck(cudaMemcpy(hostDistances.data(), deviceDistances, n * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "memcpy dist back");

    // cleanup
    cudaFree(deviceRowPointer);
    cudaFree(deviceColumnIndex);
    cudaFree(deviceWeights);
    cudaFree(deviceDistances);
    cudaFree(currentlyActiveNodes);
    cudaFree(nextActiveNodes);
    cudaFree(currentlyActiveNodeCount);

    return hostDistances;
}

} // namespace NetworKit::GPU
