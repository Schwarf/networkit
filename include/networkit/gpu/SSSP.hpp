/*  DeviceGraph.hpp
*
 *  Created on: 16.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#ifndef NETWORKIT_GPU_SSSP_HPP_
#define NETWORKIT_GPU_SSSP_HPP_
#include "DeviceGraph.hpp"
#include <vector>
namespace NetworKit::GPU {
std::vector<float> ssspBatchCuda(const DeviceGraph<float> &graph,
                                 const std::vector<DeviceGraph<float>::node_t> &sources);
}
#endif // NETWORKIT_GPU_SSSP_HPP_
