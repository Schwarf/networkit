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
 * DeviceNodeT indexes nodes, DeviceOffsetT indexes adjacency entries, and DeviceEdgeWeightT stores
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
template <class Backend, class DeviceNodeT, class DeviceOffsetT, class DeviceEdgeWeightT>
class DeviceGraphCSR {
    static_assert(std::is_same_v<DeviceNodeT, uint32_t> || std::is_same_v<DeviceNodeT, uint64_t>,
                  "CSR node indices must be uint32_t or uint64_t");
    static_assert(std::is_same_v<DeviceOffsetT, uint32_t>
                      || std::is_same_v<DeviceOffsetT, uint64_t>,
                  "CSR offsets must be uint32_t or uint64_t");
    static_assert(std::is_floating_point_v<DeviceEdgeWeightT>
                      || std::is_integral_v<DeviceEdgeWeightT>,
                  "CSR weights must be numeric");

public:
    using node_type = DeviceNodeT;
    using offset_type = DeviceOffsetT;
    using weight_type = DeviceEdgeWeightT;

    DeviceGraphCSR(typename Backend::Context &context, std::size_t nodes, std::size_t entries)
        : offsets(context, checkedRowOffsetCount(nodes, entries)), columns(context, entries),
          values(context, entries) {}

    DeviceGraphCSR(const DeviceGraphCSR &) = delete;
    DeviceGraphCSR &operator=(const DeviceGraphCSR &) = delete;
    DeviceGraphCSR(DeviceGraphCSR &&) = default;
    DeviceGraphCSR &operator=(DeviceGraphCSR &&) = default;

    std::size_t numberOfNodes() const { return offsets.size() == 0 ? 0 : offsets.size() - 1; }
    std::size_t numberOfEntries() const { return columns.size(); }

    DeviceBuffer<Backend, DeviceOffsetT> &rowOffsets() { return offsets; }
    const DeviceBuffer<Backend, DeviceOffsetT> &rowOffsets() const { return offsets; }

    DeviceBuffer<Backend, DeviceNodeT> &columnIndices() { return columns; }
    const DeviceBuffer<Backend, DeviceNodeT> &columnIndices() const { return columns; }

    DeviceBuffer<Backend, DeviceEdgeWeightT> &weights() { return values; }
    const DeviceBuffer<Backend, DeviceEdgeWeightT> &weights() const { return values; }

    void uploadToDevice() {
        offsets.uploadToDevice();
        columns.uploadToDevice();
        values.uploadToDevice();
    }

private:
    // Validate both dimensions before constructing offsets, the first buffer.
    static std::size_t checkedRowOffsetCount(std::size_t nodes, std::size_t entries) {
        if (nodes == std::numeric_limits<std::size_t>::max()
            || (nodes > 0 && nodes - 1 > std::numeric_limits<DeviceNodeT>::max())) {
            throw std::length_error("DeviceGraphCSR: node count exceeds index capacity");
        }
        if (entries > std::numeric_limits<DeviceOffsetT>::max()) {
            throw std::length_error("DeviceGraphCSR: entry count exceeds offset capacity");
        }
        return nodes + 1;
    }

    // Declaration order ensures validation precedes every buffer allocation.
    DeviceBuffer<Backend, DeviceOffsetT> offsets;
    DeviceBuffer<Backend, DeviceNodeT> columns;
    DeviceBuffer<Backend, DeviceEdgeWeightT> values;
};

} // namespace NetworKit::GPU

#endif // NETWORKIT_GRAPH_GPU_DEVICE_GRAPH_CSR_HPP_
