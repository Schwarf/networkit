// networkit-format
#ifndef NETWORKIT_GRAPH_GPU_DEVICE_GRAPH_CSR_BUILDER_HPP_
#define NETWORKIT_GRAPH_GPU_DEVICE_GRAPH_CSR_BUILDER_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <networkit/graph/AdjListGraph.hpp>
#include <networkit/graph/gpu/DeviceGraphCSR.hpp>

namespace NetworKit::GPU {

enum class CSRAdjacency { OUTGOING, INCOMING };

namespace CSRBuilderDetail {

// Format only on error paths. Preserve full integer values and floating-point
// round-trip precision, including values very close to conversion boundaries.
template <class Number>
std::string numberString(Number value) {
    std::ostringstream text;
    text.imbue(std::locale::classic());
    if constexpr (std::is_floating_point_v<Number>) {
        text.precision(std::numeric_limits<Number>::max_digits10);
    }
    text << +value;
    return text.str();
}

// Kept independent of graph allocation so limits can be checked with large counts.
template <class DeviceNodeT, GraphNode HostNodeT>
std::size_t checkedNodeCount(count nodes, HostNodeT upperNodeIdBound) {
    if (!std::in_range<std::size_t>(nodes) || nodes == std::numeric_limits<std::size_t>::max()
        || (nodes > 0 && nodes - 1 > std::numeric_limits<DeviceNodeT>::max())
        || !std::in_range<std::size_t>(+upperNodeIdBound)) {
        throw std::length_error("buildDeviceGraphCSR: node count or ID range exceeds capacity");
    }
    return static_cast<std::size_t>(nodes);
}

template <class DeviceOffsetT>
std::size_t checkedEntryCount(std::size_t entries, count degree) {
    const auto limit = std::min<std::uintmax_t>(std::numeric_limits<std::size_t>::max(),
                                                std::numeric_limits<DeviceOffsetT>::max());
    if (entries > limit || std::cmp_greater(degree, limit - entries)) {
        throw std::length_error("buildDeviceGraphCSR: adjacency count exceeds offset capacity");
    }
    return entries + static_cast<std::size_t>(degree);
}

template <class DeviceEdgeWeightT, GraphEdgeWeight HostEdgeWeightT>
DeviceEdgeWeightT checkedEdgeWeight(HostEdgeWeightT weight) {
    if constexpr (std::is_integral_v<HostEdgeWeightT> && std::is_integral_v<DeviceEdgeWeightT>) {
        // Unary + promotes char and bool to integer types accepted by cmp_*.
        if (std::cmp_less(+weight, +std::numeric_limits<DeviceEdgeWeightT>::lowest())) {
            throw std::invalid_argument(
                "buildDeviceGraphCSR: integer weight " + numberString(weight)
                + " is below DeviceEdgeWeightT minimum "
                + numberString(std::numeric_limits<DeviceEdgeWeightT>::lowest()));
        }
        if (std::cmp_greater(+weight, +std::numeric_limits<DeviceEdgeWeightT>::max())) {
            throw std::invalid_argument(
                "buildDeviceGraphCSR: integer weight " + numberString(weight)
                + " exceeds DeviceEdgeWeightT maximum "
                + numberString(std::numeric_limits<DeviceEdgeWeightT>::max()));
        }
    } else {
        const auto value = static_cast<long double>(weight);
        if (!std::isfinite(value)) {
            throw std::invalid_argument("buildDeviceGraphCSR: non-finite weight "
                                        + numberString(weight) + "; a finite weight is required");
        }
        if constexpr (std::is_integral_v<DeviceEdgeWeightT>) {
            // The upper bound is exclusive. Unlike integer max(), powers of two
            // remain exact even when long double has only double precision.
            const auto upper = std::ldexp(1.L, std::numeric_limits<DeviceEdgeWeightT>::digits);
            const auto lower = std::is_signed_v<DeviceEdgeWeightT> ? -upper : 0.L;
            if (value < lower) {
                throw std::invalid_argument(
                    "buildDeviceGraphCSR: floating-point weight " + numberString(weight)
                    + " is below integer DeviceEdgeWeightT minimum " + numberString(lower));
            }
            if (value >= upper) {
                throw std::invalid_argument(
                    "buildDeviceGraphCSR: floating-point weight " + numberString(weight)
                    + " reaches or exceeds integer DeviceEdgeWeightT exclusive upper bound "
                    + numberString(upper));
            }
            if (std::trunc(value) != value) {
                throw std::invalid_argument(
                    "buildDeviceGraphCSR: fractional weight " + numberString(weight)
                    + " cannot be converted to an integer DeviceEdgeWeightT without truncation");
            }
        } else {
            if (value
                < static_cast<long double>(std::numeric_limits<DeviceEdgeWeightT>::lowest())) {
                throw std::invalid_argument(
                    "buildDeviceGraphCSR: floating-point weight " + numberString(weight)
                    + " is below floating-point DeviceEdgeWeightT minimum "
                    + numberString(std::numeric_limits<DeviceEdgeWeightT>::lowest()));
            }
            if (value > static_cast<long double>(std::numeric_limits<DeviceEdgeWeightT>::max())) {
                throw std::invalid_argument(
                    "buildDeviceGraphCSR: floating-point weight " + numberString(weight)
                    + " exceeds floating-point DeviceEdgeWeightT maximum "
                    + numberString(std::numeric_limits<DeviceEdgeWeightT>::max()));
            }
        }
    }
    return static_cast<DeviceEdgeWeightT>(weight);
}

} // namespace CSRBuilderDetail

/** CSR storage and the metadata required to interpret its rows and columns. */
template <class Backend, class DeviceNodeT, class DeviceOffsetT, class DeviceEdgeWeightT,
          class HostNodeT>
struct DeviceGraphCSRBuildResult {
    DeviceGraphCSR<Backend, DeviceNodeT, DeviceOffsetT, DeviceEdgeWeightT> graph;
    // originalNodeIds[row] is the source graph's node ID for this compact index.
    std::vector<HostNodeT> originalNodeIds;
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
 * Weights must be finite and representable in DeviceEdgeWeightT. Floating-point rounding
 * is allowed; fractional weights cannot be converted to an integer DeviceEdgeWeightT.
 * No algorithm-specific normalization or coefficient transformation is applied.
 */
template <class Backend, class DeviceNodeT, class DeviceOffsetT, class DeviceEdgeWeightT,
          GraphNode HostNodeT, GraphEdgeWeight HostEdgeWeightT>
auto buildDeviceGraphCSR(typename Backend::Context &context,
                         const AdjListGraph<HostNodeT, HostEdgeWeightT> &source,
                         CSRAdjacency adjacency = CSRAdjacency::OUTGOING) {
    using Result = DeviceGraphCSRBuildResult<Backend, DeviceNodeT, DeviceOffsetT, DeviceEdgeWeightT,
                                             HostNodeT>;

    if (adjacency != CSRAdjacency::OUTGOING && adjacency != CSRAdjacency::INCOMING) {
        throw std::invalid_argument("buildDeviceGraphCSR: invalid adjacency orientation");
    }
    const auto nodes = CSRBuilderDetail::checkedNodeCount<DeviceNodeT>(source.numberOfNodes(),
                                                                       source.upperNodeIdBound());
    const auto degree = [&](HostNodeT u) {
        return adjacency == CSRAdjacency::OUTGOING ? source.degree(u) : source.degreeIn(u);
    };
    std::size_t entries = 0;
    source.forNodes([&](HostNodeT u) {
        entries = CSRBuilderDetail::checkedEntryCount<DeviceOffsetT>(entries, degree(u));
    });

    Result result{DeviceGraphCSR<Backend, DeviceNodeT, DeviceOffsetT, DeviceEdgeWeightT>(
                      context, nodes, entries),
                  {},
                  adjacency,
                  source.isDirected()};
    result.originalNodeIds.reserve(nodes);
    std::vector<DeviceNodeT> nodeToRow(static_cast<std::size_t>(source.upperNodeIdBound()));
    source.forNodes([&](HostNodeT u) {
        nodeToRow[static_cast<std::size_t>(u)] =
            static_cast<DeviceNodeT>(result.originalNodeIds.size());
        result.originalNodeIds.push_back(u);
    });

    auto offsets = result.graph.rowOffsets().getWritableBuffer();
    auto columns = result.graph.columnIndices().getWritableBuffer();
    auto weights = result.graph.weights().getWritableBuffer();
    offsets[0] = 0;
    std::size_t entry = 0;
    for (std::size_t row = 0; row < nodes; ++row) {
        const auto u = result.originalNodeIds[row];
        // Use the full edge callback: (node, weight) is ambiguous when node
        // IDs and weights have the same type, and three arguments can overlap
        // with (node, node, edgeid) when weights have the edgeid type.
        const auto fillNeighbor = [&](HostNodeT, HostNodeT v, HostEdgeWeightT weight, edgeid) {
            columns[entry] = nodeToRow[static_cast<std::size_t>(v)];
            try {
                weights[entry] = CSRBuilderDetail::checkedEdgeWeight<DeviceEdgeWeightT>(weight);
            } catch (const std::invalid_argument &error) {
                const bool outgoing = adjacency == CSRAdjacency::OUTGOING;
                throw std::invalid_argument(
                    std::string(error.what()) + "; source edge "
                    + CSRBuilderDetail::numberString(outgoing ? u : v) + " -> "
                    + CSRBuilderDetail::numberString(outgoing ? v : u) + "; CSR row "
                    + std::to_string(row) + (outgoing ? " (outgoing)" : " (incoming)"));
            }
            ++entry;
        };
        if (adjacency == CSRAdjacency::OUTGOING) {
            source.forEdgesOf(u, fillNeighbor);
        } else {
            source.forInEdgesOf(u, fillNeighbor);
        }
        offsets[row + 1] = static_cast<DeviceOffsetT>(entry);
    }
    result.graph.uploadToDevice();
    return result;
}

} // namespace NetworKit::GPU

#endif // NETWORKIT_GRAPH_GPU_DEVICE_GRAPH_CSR_BUILDER_HPP_
