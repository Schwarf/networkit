/*  SSSP.hpp
 *
 *  Created on: 16.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#ifndef NETWORKIT_GPU_SSSP_HPP_
#define NETWORKIT_GPU_SSSP_HPP_

#include "HostGraphCSR.hpp"
#include <vector>

namespace NetworKit::GPU {
std::vector<float> ssspWorklistCuda(const HostGraphCSR<float> &graph, HostGraphCSR<float>::node_t source);
}
#endif // NETWORKIT_GPU_SSSP_HPP_
