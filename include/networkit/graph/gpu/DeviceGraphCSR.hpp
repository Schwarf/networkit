// networkit-format
#ifndef NETWORKIT_GRAPH_GPU_DEVICE_GRAPH_CSR_HPP_
#define NETWORKIT_GRAPH_GPU_DEVICE_GRAPH_CSR_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <networkit/graph/gpu/DeviceBuffer.hpp>

namespace NetworKit::GPU {

/**
 * Owns backend-managed graph storage in compressed sparse row (CSR) format.
 *
 * NodeT indexes nodes, OffsetT indexes adjacency entries, and WeightT stores
 * edge weights. Node and offset widths are independent. Supported element types
 * and operations may be further restricted by the backend.
 *
 * Construction allocates storage only. The caller fills rowOffsets(),
 * columnIndices(), and weights() through getWritableBuffer(), then calls
 * uploadToDevice(). Offsets must start at zero, be nondecreasing, and end at
 * numberOfEntries(); columns must be less than numberOfNodes().
 *
 * Row u stores the neighbors of node u and their corresponding edge weights.
 * For directed graphs, the builder determines whether rows store incoming or
 * outgoing adjacency and must communicate that convention to consumers.
 *
 * An undirected edge generally contributes two entries. Buffers and graph
 * storage can be reused. Updating or rebuilding it when the graph changes is
 * the caller's responsibility. No NetworKit Graph is owned.
 */
template <class Backend, class NodeT, class OffsetT, class WeightT>
class DeviceGraphCSR {
    static_assert(std::is_same_v<NodeT, uint32_t> || std::is_same_v<NodeT, uint64_t>,
                  "CSR node indices must be uint32_t or uint64_t");
    static_assert(std::is_same_v<OffsetT, uint32_t> || std::is_same_v<OffsetT, uint64_t>,
                  "CSR offsets must be uint32_t or uint64_t");
    static_assert(std::is_floating_point_v<WeightT> || std::is_integral_v<WeightT>,
                  "CSR weights must be numeric");

public:
    using node_type = NodeT;
    using offset_type = OffsetT;
    using weight_type = WeightT;

    DeviceGraphCSR(typename Backend::Context &context, std::size_t nodes, std::size_t entries)
        : DeviceGraphCSR(context, checkedShape(nodes, entries)) {}

    DeviceGraphCSR(const DeviceGraphCSR &) = delete;
    DeviceGraphCSR &operator=(const DeviceGraphCSR &) = delete;
    DeviceGraphCSR(DeviceGraphCSR &&) = default;
    DeviceGraphCSR &operator=(DeviceGraphCSR &&) = default;

    std::size_t numberOfNodes() const { return offsets.size() == 0 ? 0 : offsets.size() - 1; }
    std::size_t numberOfEntries() const { return columns.size(); }

    DeviceBuffer<Backend, OffsetT> &rowOffsets() { return offsets; }
    const DeviceBuffer<Backend, OffsetT> &rowOffsets() const { return offsets; }

    DeviceBuffer<Backend, NodeT> &columnIndices() { return columns; }
    const DeviceBuffer<Backend, NodeT> &columnIndices() const { return columns; }

    DeviceBuffer<Backend, WeightT> &weights() { return values; }
    const DeviceBuffer<Backend, WeightT> &weights() const { return values; }

    void uploadToDevice() {
        offsets.uploadToDevice();
        columns.uploadToDevice();
        values.uploadToDevice();
    }

private:
    struct Shape {
        std::size_t nodes;
        std::size_t entries;
    };

    static Shape checkedShape(std::size_t nodes, std::size_t entries) {
        if (nodes == std::numeric_limits<std::size_t>::max()
            || (nodes > 0 && nodes - 1 > std::numeric_limits<NodeT>::max())) {
            throw std::length_error("DeviceGraphCSR: node count exceeds index capacity");
        }
        if (entries > std::numeric_limits<OffsetT>::max()) {
            throw std::length_error("DeviceGraphCSR: entry count exceeds offset capacity");
        }
        return {nodes, entries};
    }

    DeviceGraphCSR(typename Backend::Context &context, Shape shape)
        : offsets(context, shape.nodes + 1), columns(context, shape.entries),
          values(context, shape.entries) {}

    DeviceBuffer<Backend, OffsetT> offsets;
    DeviceBuffer<Backend, NodeT> columns;
    DeviceBuffer<Backend, WeightT> values;
};

} // namespace NetworKit::GPU

#endif // NETWORKIT_GRAPH_GPU_DEVICE_GRAPH_CSR_HPP_
