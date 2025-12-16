/*  DeviceGraph.hpp
 *
 *  Created on: 16.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#ifndef NETWORKIT_GPU_DEVICEGRAPH_HPP
#define NETWORKIT_GPU_DEVICEGRAPH_HPP

#include <networkit/graph/Graph.hpp>

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace NetworKit {

template <typename WeightT, count N = 0>
struct DeviceGraph {
    static_assert(std::is_floating_point_v<WeightT>, "DeviceGraph<WeightT>: WeightT should be float or double.");

    static constexpr bool HasStaticN = (N != 0);
    static constexpr bool Prefer32 =
        HasStaticN && (N <= static_cast<count>(std::numeric_limits<std::uint32_t>::max()));

    using node_t = std::conditional_t<Prefer32, std::uint32_t, node>;
    using index_t = std::conditional_t<Prefer32, std::uint32_t, index>;

    std::vector<index_t> rowPointer;   // size n+1
    std::vector<node_t> columnIndices; // size m'
    std::vector<WeightT> weights;      // size m' if hasWeights==true, else empty

    bool hasWeights = false;
    bool directed = false;
};

template <typename WeightT, count N = 0>
DeviceGraph<WeightT, N>
buildDeviceGraph(const Graph &G, bool storeWeights = true, bool requireContinuousNodeIds = true,
                 bool forceDefaultWeights = false, WeightT defaultWeight = WeightT{1}) {
    using DG = DeviceGraph<WeightT, N>;

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

    // For CUDA kernels you typically want compact node IDs [0..n-1]
    if (requireContinuousNodeIds) {
        if (G.upperNodeIdBound() != G.numberOfNodes()) {
            throw std::runtime_error("buildDeviceGraph: Graph node IDs are not continuous "
                                     "(upperNodeIdBound != numberOfNodes). "
                                     "Compact/reindex the graph, or implement a remap build.");
        }
    }

    // Check node ID range fits chosen DG::node_t
    if ((G.upperNodeIdBound())
        > static_cast<index>(std::numeric_limits<typename DG::node_t>::max())) {
        throw std::runtime_error("buildDeviceGraph: node IDs do not fit into chosen node_t.");
    }

    deviceGraph.rowPointer.assign(numberOfNodes + 1, 0);

    deviceGraph.hasWeights = (storeWeights && graphHasWeights) || forceDefaultWeights;

    // ---- pass 1: degree count ----
    for (node u = 0; u < G.numberOfNodes(); ++u) {
        count degree = 0;
        G.forNeighborsOf(u, [&](node /*v*/) { ++degree; });

        if (degree > static_cast<count>(std::numeric_limits<typename DG::index_t>::max())) {
            throw std::runtime_error("buildDeviceGraph: a node degree exceeds index_t capacity.");
        }
        deviceGraph.rowPointer[u + 1] = static_cast<DG::index_t>(degree);
    }

    // prefix sum -> rowPointer offsets
    for (std::size_t i = 1; i < deviceGraph.rowPointer.size(); ++i) {
        const std::uint64_t sum = static_cast<std::uint64_t>(deviceGraph.rowPointer[i])
                                  + static_cast<std::uint64_t>(deviceGraph.rowPointer[i - 1]);

        if (sum > static_cast<count>(std::numeric_limits<typename DG::index_t>::max())) {
            throw std::runtime_error(
                "buildDeviceGraph: total stored edge count (m') exceeds index_t capacity. "
                "Use 64-bit indices (e.g., buildDeviceGraph<WeightT, 0>), or change index_t.");
        }
        deviceGraph.rowPointer[i] = static_cast<DG::index_t>(sum);
    }

    // Total number of CSR edge entries (directed edges; undirected graphs typically store both directions).
    const std::uint64_t numberOfStoredEdges = static_cast<std::uint64_t>(deviceGraph.rowPointer.back());

    // allocate
    deviceGraph.columnIndices.resize(static_cast<count>(numberOfStoredEdges));
    if (deviceGraph.hasWeights)
        deviceGraph.weights.resize(static_cast<count>(numberOfStoredEdges));
    else
        deviceGraph.weights.clear();

    // Per-node write offsets into columnIndices/weights.
    // Initialized to rowPointer[u] and incremented as we fill edges for node u.
    std::vector<typename DG::index_t> writeOffset(deviceGraph.rowPointer.size());
    for (std::size_t i = 0; i < deviceGraph.rowPointer.size(); ++i) {
        writeOffset[i] = static_cast<DG::index_t>(deviceGraph.rowPointer[i]);
    }

    // ---- pass 2: fill ----
    if (deviceGraph.hasWeights && graphHasWeights) {
        for (node u = 0; u < G.numberOfNodes(); ++u) {
            G.forNeighborsOf(u, [&](node v, edgeweight w) {
                const auto pos = writeOffset[u]++;
                deviceGraph.columnIndices[static_cast<std::size_t>(pos)] =
                    static_cast<typename DG::node_t>(v);
                deviceGraph.weights[static_cast<std::size_t>(pos)] = static_cast<WeightT>(w);
            });
        }
    } else if (deviceGraph.hasWeights && !graphHasWeights) {
        for (node u = 0; u < G.numberOfNodes(); ++u) {
            G.forNeighborsOf(u, [&](node v) {
                const auto pos = writeOffset[static_cast<std::size_t>(u)]++;
                deviceGraph.columnIndices[static_cast<std::size_t>(pos)] =
                    static_cast<typename DG::node_t>(v);
                deviceGraph.weights[static_cast<std::size_t>(pos)] = defaultWeight;
            });
        }
    } else {
        for (node u = 0; u < G.numberOfNodes(); ++u) {
            G.forNeighborsOf(u, [&](node v) {
                const auto pos = writeOffset[static_cast<std::size_t>(u)]++;
                deviceGraph.columnIndices[static_cast<std::size_t>(pos)] =
                    static_cast<typename DG::node_t>(v);
            });
        }
    }

    return deviceGraph;
}

} // namespace NetworKit

#endif // NETWORKIT_GPU_DEVICEGRAPH_HPP
