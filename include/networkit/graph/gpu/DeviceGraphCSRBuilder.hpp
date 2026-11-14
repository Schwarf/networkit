// networkit-format
#ifndef NETWORKIT_GRAPH_GPU_DEVICE_GRAPH_CSR_BUILDER_HPP_
#define NETWORKIT_GRAPH_GPU_DEVICE_GRAPH_CSR_BUILDER_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <networkit/graph/AdjListGraph.hpp>
#include <networkit/graph/gpu/DeviceGraphCSR.hpp>

namespace NetworKit::GPU {

enum class CSRAdjacency { OUTGOING, INCOMING };

/** CSR storage and the metadata required to interpret its rows and columns. */
template <class Backend, class NodeT, class OffsetT, class WeightT, class SourceNodeT>
struct DeviceGraphCSRBuildResult {
    DeviceGraphCSR<Backend, NodeT, OffsetT, WeightT> graph;
    // originalNodeIds[row] is the source graph's node ID for this compact index.
    std::vector<SourceNodeT> originalNodeIds;
    CSRAdjacency adjacency;
    bool directed;
};

/**
 * Converts an adjacency-list graph directly into device preparation buffers.
 *
 * Existing nodes receive compact indices in graph iteration order. Row u lists
 * outgoing neighbors by default, or incoming neighbors when requested. Columns
 * use the same compact indices; originalNodeIds maps them back to source IDs.
 * Undirected adjacency is identical in either orientation, with self-loops
 * stored once. Unweighted edges receive weight 1.
 *
 * The backend supplies CPU-accessible spans through getWritableBuffer(); no separate
 * host CSR allocation is made. uploadToDevice() is called before returning.
 * Graph conversion makes no CUDA or Metal API calls. The graph must not be
 * modified during construction, and context must outlive the returned storage.
 *
 * Weights must be finite and representable in WeightT. Floating-point rounding
 * is allowed; fractional weights cannot be converted to an integer WeightT.
 * No algorithm-specific normalization or coefficient transformation is applied.
 */
template <class Backend, class NodeT, class OffsetT, class WeightT, GraphNode SourceNodeT,
          GraphEdgeWeight SourceWeightT>
auto buildDeviceGraphCSR(typename Backend::Context &context,
                         const AdjListGraph<SourceNodeT, SourceWeightT> &source,
                         CSRAdjacency adjacency = CSRAdjacency::OUTGOING) {
    using Result = DeviceGraphCSRBuildResult<Backend, NodeT, OffsetT, WeightT, SourceNodeT>;

    if (adjacency != CSRAdjacency::OUTGOING && adjacency != CSRAdjacency::INCOMING) {
        throw std::invalid_argument("buildDeviceGraphCSR: invalid adjacency orientation");
    }
    const auto nodeCount = source.numberOfNodes();
    if (!std::in_range<std::size_t>(nodeCount)
        || nodeCount == std::numeric_limits<std::size_t>::max()
        || (nodeCount > 0 && nodeCount - 1 > std::numeric_limits<NodeT>::max())
        || !std::in_range<std::size_t>(source.upperNodeIdBound())) {
        throw std::length_error("buildDeviceGraphCSR: node count or ID range exceeds capacity");
    }

    const auto nodes = static_cast<std::size_t>(nodeCount);
    const auto entryLimit = std::min<std::uintmax_t>(std::numeric_limits<std::size_t>::max(),
                                                     std::numeric_limits<OffsetT>::max());
    const auto degree = [&](SourceNodeT u) {
        return adjacency == CSRAdjacency::OUTGOING ? source.degree(u) : source.degreeIn(u);
    };
    std::size_t entries = 0;
    source.forNodes([&](SourceNodeT u) {
        const auto d = degree(u);
        if (std::cmp_greater(d, entryLimit - entries)) {
            throw std::length_error("buildDeviceGraphCSR: adjacency count exceeds offset capacity");
        }
        entries += static_cast<std::size_t>(d);
    });

    Result result{DeviceGraphCSR<Backend, NodeT, OffsetT, WeightT>(context, nodes, entries),
                  {},
                  adjacency,
                  source.isDirected()};
    result.originalNodeIds.reserve(nodes);
    std::vector<NodeT> nodeToRow(static_cast<std::size_t>(source.upperNodeIdBound()));
    source.forNodes([&](SourceNodeT u) {
        nodeToRow[static_cast<std::size_t>(u)] = static_cast<NodeT>(result.originalNodeIds.size());
        result.originalNodeIds.push_back(u);
    });

    auto offsets = result.graph.rowOffsets().getWritableBuffer();
    auto columns = result.graph.columnIndices().getWritableBuffer();
    auto weights = result.graph.weights().getWritableBuffer();
    offsets[0] = 0;
    std::size_t entry = 0;
    for (std::size_t row = 0; row < nodes; ++row) {
        const auto fillNeighbor = [&](SourceNodeT v, SourceWeightT weight) {
            const auto value = static_cast<long double>(weight);
            if (!std::isfinite(value)
                || value < static_cast<long double>(std::numeric_limits<WeightT>::lowest())
                || value > static_cast<long double>(std::numeric_limits<WeightT>::max())
                || (std::is_integral_v<WeightT> && std::trunc(value) != value)) {
                throw std::invalid_argument("buildDeviceGraphCSR: weight cannot be represented");
            }
            columns[entry] = nodeToRow[static_cast<std::size_t>(v)];
            weights[entry] = static_cast<WeightT>(weight);
            ++entry;
        };
        const auto u = result.originalNodeIds[row];
        if (adjacency == CSRAdjacency::OUTGOING) {
            source.forNeighborsOf(u, fillNeighbor);
        } else {
            source.forInNeighborsOf(u, fillNeighbor);
        }
        offsets[row + 1] = static_cast<OffsetT>(entry);
    }
    result.graph.uploadToDevice();
    return result;
}

} // namespace NetworKit::GPU

#endif // NETWORKIT_GRAPH_GPU_DEVICE_GRAPH_CSR_BUILDER_HPP_
