/*  SSSPKernels.cuh
 *
 *  Created on: 17.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */
#ifndef NETWORKIT_GPU_SSSP_KERNELS_CUH_
#define NETWORKIT_GPU_SSSP_KERNELS_CUH_

#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace NetworKit::GPU {

template <typename node_t, typename WeightT>
__global__ void initDistancesAndFrontierKernel(WeightT *distances, node_t numberOfNodes,
                                               node_t source, node_t *frontier,
                                               std::uint32_t *frontierCount) {

    const node_t tid = static_cast<node_t>(blockIdx.x) * static_cast<node_t>(blockDim.x)
                       + static_cast<node_t>(threadIdx.x);

    if (tid < numberOfNodes) {
        distances[tid] = (tid == source)?  0.0f : CUDART_INF_F;
    }

    // initialize source
    if (tid == 0) {
        frontier[0] = source;
        *frontierCount = 1;
    }
}

__device__ inline float atomicMinFloat(float *address, float value) {
    int *addressAsInt = reinterpret_cast<int *>(address);
    int oldValue = *addressAsInt;

    while (true) {
        const float oldFloatValue = __int_as_float(oldValue);
        if (oldFloatValue <= value)
            return oldFloatValue;
        const int assumed = oldValue;
        oldValue = atomicCAS(addressAsInt, assumed, /*desired*/ __float_as_int(value));
        if (oldValue == assumed)
            return oldFloatValue;
    }
}

template <typename IndexT, typename NodeT>
__global__ void
relaxFromFrontierKernel(const IndexT *__restrict__ rowPointer,
                        const NodeT *__restrict__ columnIndices, const float *__restrict__ weights,
                        float *__restrict__ dist, const NodeT *__restrict__ currentFrontier,
                        std::uint32_t currentCount, NodeT *__restrict__ nextFrontier,
                        std::uint32_t *__restrict__ nextCount) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= currentCount)
        return;

    const NodeT u = currentFrontier[tid];
    const float du = dist[u];

    const IndexT begin = rowPointer[u];
    const IndexT end = rowPointer[u + 1];

    for (IndexT e = begin; e < end; ++e) {
        const NodeT v = columnIndices[e];
        const float nd = du + weights[e];

        const float old = atomicMinFloat(&dist[v], nd);
        if (nd < old) {
            const std::uint32_t pos = atomicAdd(nextCount, 1u);
            nextFrontier[pos] = v;
        }
    }
}

} // namespace NetworKit::GPU

#endif // NETWORKIT_GPU_SSSP_KERNELS_CUH_
