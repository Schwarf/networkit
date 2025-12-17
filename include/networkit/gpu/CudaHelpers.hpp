/*  CudaHelpers.hpp
 *
 *  Created on: 17.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#ifndef NETWORKIT_GPU_CUDA_HELPERS_HPP_
#define NETWORKIT_GPU_CUDA_HELPERS_HPP_

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace NetworKit::GPU {
inline void cudaCheck(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": "
                                 + cudaGetErrorString(error));
    }
}

} // namespace NetworKit::GPU
#endif // NETWORKIT_GPU_CUDA_HELPERS_HPP_
