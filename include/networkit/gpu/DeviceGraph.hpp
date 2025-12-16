/*  CSRGraph.hpp
 *
 *  Created on: 16.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#ifndef NETWORKIT_CSRGRAPH_H
#define NETWORKIT_CSRGRAPH_H

#include <networkit/graph/Graph.hpp>

#include <type_traits>
#include <vector>
namespace NetworKit {

template <typename WeightT, std::size_t N = 0>
struct DeviceGraph {
    static_assert(std::is_floating_point_v<WeightT>,
                  "DeviceGraph<WeightT>: WeightT should be float or double.");

    static constexpr bool HasStaticN = (N != 0);
    static constexpr bool Prefer32   = HasStaticN && (N <= std::numeric_limits<std::uint32_t>::max());

    using node_t  = std::conditional_t<Prefer32, std::uint32_t, std::uint64_t>;
    using index_t = std::conditional_t<Prefer32, std::uint32_t, std::uint64_t>;

    std::vector<index_t> rowPointer;
    std::vector<node_t>  columnIndices;
    std::vector<WeightT> weights;

    bool hasWeights = false;
    bool directed   = false;
};


} // namespace NetworKit
#endif // NETWORKIT_CSRGRAPH_H
