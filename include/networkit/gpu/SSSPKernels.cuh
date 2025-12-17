/*  SSSPKernels.cuh
*
 *  Created on: 17.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */
#ifndef NETWORKIT_GPU_SSSP_KERNELS_CUH_
#define NETWORKIT_GPU_SSSP_KERNELS_CUH_

#include <cuda_runtime.h>
#include <cstdint>


namespace NetworKit::GPU {

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
__global__ void relaxFromFrontierKernel(const IndexT* __restrict__ rowPointer,
                                        const NodeT*  __restrict__ columnIndices,
                                        const float*  __restrict__ weights,
                                        float* __restrict__ dist,
                                        const NodeT* __restrict__ currentFrontier,
                                        std::uint32_t currentCount,
                                        NodeT* __restrict__ nextFrontier,
                                        std::uint32_t* __restrict__ nextCount) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= currentCount) return;

    const NodeT u = currentFrontier[tid];
    const float du = dist[u];

    const IndexT begin = rowPointer[u];
    const IndexT end   = rowPointer[u + 1];

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
