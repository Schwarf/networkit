/*  HostGraphCSR.hpp
 *
 *  Created on: 16.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#ifndef NETWORKIT_GPU_HOST_GRAPH_CSR_HPP_
#define NETWORKIT_GPU_HOST_GRAPH_CSR_HPP_

#include <networkit/graph/Graph.hpp>

#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace NetworKit::GPU {

template <typename WeightT, count N = 0>
struct HostGraphCSR {
    static_assert(std::is_floating_point_v<WeightT>,
                  "DeviceGraph<WeightT>: WeightT should be float or double.");

    static constexpr bool HasStaticN = (N != 0);
    static constexpr bool Prefer32 =
        HasStaticN && (N <= static_cast<count>(std::numeric_limits<std::uint32_t>::max()));

    using node_t = std::conditional_t<Prefer32, std::uint32_t, node>;
    using index_t = std::conditional_t<Prefer32, std::uint32_t, index>;

    std::vector<index_t> rowPointer;
    std::vector<node_t> columnIndices;
    std::vector<WeightT> weights;

    bool hasWeights = false;
    bool directed = false;
};

template <typename WeightT, count N = 0>
HostGraphCSR<WeightT, N> buildHostGraphCSR(const Graph &G, bool requireContinuousNodeIds = true,
                                         std::optional<WeightT> defaultWeight = std::nullopt) {
    using DG = HostGraphCSR<WeightT, N>;

    DG deviceGraph;
    deviceGraph.directed = G.isDirected();

    const count numberOfNodes = G.numberOfNodes();
    const bool graphHasWeights = G.isWeighted();

    if constexpr (DG::HasStaticN) {
        if (numberOfNodes != static_cast<count>(N)) {
            throw std::runtime_error(
                "buildDeviceGraph: Graph.numberOfNodes() does not match template N.");
        }
    }

    if (requireContinuousNodeIds) {
        if (G.upperNodeIdBound() != G.numberOfNodes()) {
            throw std::runtime_error("buildDeviceGraph: Graph node IDs are not continuous "
                                     "(upperNodeIdBound != numberOfNodes). "
                                     "Compact/reindex the graph, or implement a remap build.");
        }
    }

    // node_t range check (note: node_t may be uint32_t or NetworKit::node)
    if (G.upperNodeIdBound()
        > static_cast<index>(std::numeric_limits<typename DG::node_t>::max())) {
        throw std::runtime_error("buildDeviceGraph: node IDs do not fit into chosen node_t.");
    }

    deviceGraph.rowPointer.assign(static_cast<std::size_t>(numberOfNodes) + 1, 0);

    // Weight policy:
    // - weighted graph: store weights
    // - unweighted graph: store weights iff defaultWeight provided
    deviceGraph.hasWeights = graphHasWeights || defaultWeight.has_value();

    // ---- pass 1: degree count ----
    for (node u = 0; u < numberOfNodes; ++u) {
        count degree = 0;
        G.forNeighborsOf(u, [&](node /*v*/) { ++degree; });

        if (degree > static_cast<count>(std::numeric_limits<typename DG::index_t>::max())) {
            throw std::runtime_error("buildDeviceGraph: a node degree exceeds index_t capacity.");
        }
        deviceGraph.rowPointer[static_cast<std::size_t>(u) + 1] = static_cast<DG::index_t>(degree);
    }

    // prefix sum -> rowPointer offsets
    for (std::size_t i = 1; i < deviceGraph.rowPointer.size(); ++i) {
        const std::uint64_t sum = static_cast<std::uint64_t>(deviceGraph.rowPointer[i])
                                  + static_cast<std::uint64_t>(deviceGraph.rowPointer[i - 1]);

        if (sum > static_cast<std::uint64_t>(std::numeric_limits<typename DG::index_t>::max())) {
            throw std::runtime_error(
                "buildDeviceGraph: total stored edge count exceeds index_t capacity. "
                "Use 64-bit indices (e.g., buildDeviceGraph<WeightT, 0>), or change index_t.");
        }
        deviceGraph.rowPointer[i] = static_cast<DG::index_t>(sum);
    }

    // Total number of CSR edge entries (directed edges; undirected graphs typically store both
    // directions).
    const std::uint64_t numberOfStoredEdges =
        static_cast<std::uint64_t>(deviceGraph.rowPointer.back());

    deviceGraph.columnIndices.resize(static_cast<std::size_t>(numberOfStoredEdges));
    if (deviceGraph.hasWeights)
        deviceGraph.weights.resize(static_cast<std::size_t>(numberOfStoredEdges));
    else
        deviceGraph.weights.clear();

    // Per-node write offsets into columnIndices/weights.
    std::vector<typename DG::index_t> writeOffset(deviceGraph.rowPointer.size());
    for (std::size_t i = 0; i < deviceGraph.rowPointer.size(); ++i) {
        writeOffset[i] = deviceGraph.rowPointer[i];
    }

    // ---- pass 2: fill ----
    if (graphHasWeights) {
        for (node u = 0; u < numberOfNodes; ++u) {
            G.forNeighborsOf(u, [&](node v, edgeweight w) {
                const auto pos = writeOffset[static_cast<std::size_t>(u)]++;
                deviceGraph.columnIndices[static_cast<std::size_t>(pos)] =
                    static_cast<DG::node_t>(v);
                if (deviceGraph.hasWeights) {
                    deviceGraph.weights[static_cast<std::size_t>(pos)] = static_cast<WeightT>(w);
                }
            });
        }
    } else {
        // unweighted
        for (node u = 0; u < numberOfNodes; ++u) {
            G.forNeighborsOf(u, [&](node v) {
                const auto pos = writeOffset[static_cast<std::size_t>(u)]++;
                deviceGraph.columnIndices[static_cast<std::size_t>(pos)] =
                    static_cast<DG::node_t>(v);
                if (deviceGraph.hasWeights) {
                    deviceGraph.weights[static_cast<std::size_t>(pos)] = *defaultWeight;
                }
            });
        }
    }

    return deviceGraph;
}

} // namespace NetworKit::GPU

#endif // NETWORKIT_GPU_HOST_GRAPH_CSR_HPP_
