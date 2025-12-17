/*  DeviceGraphCSR.hpp
 *
 *  Created on: 17.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#ifndef NETWORKIT_GPU_DEVICE_GRAPH_CSR_HPP
#define NETWORKIT_GPU_DEVICE_GRAPH_CSR_HPP

#include <networkit/gpu/HostGraphCSR.hpp>

#include <cstdint>

namespace NetworKit::GPU {

template <typename WeightT, typename IndexT, typename NodeT>
struct DeviceCSRView {
    const IndexT*  rowPointer{};
    const NodeT*   columnIndices{};
    const WeightT* weights{};
    NodeT          numberOfNodes{};
    std::uint64_t  numberOfEdgeEntries{};
};

template <typename WeightT>
class DeviceGraphCSR {
public:
    using HG      = HostGraphCSR<WeightT>;
    using index_t = typename HG::index_t;
    using node_t  = typename HG::node_t;

    explicit DeviceGraphCSR(const HG& hostGraph);
    ~DeviceGraphCSR();

    DeviceGraphCSR(const DeviceGraphCSR&) = delete;
    DeviceGraphCSR& operator=(const DeviceGraphCSR&) = delete;

    DeviceGraphCSR(DeviceGraphCSR&&) noexcept;
    DeviceGraphCSR& operator=(DeviceGraphCSR&&) noexcept;

    DeviceCSRView<WeightT, index_t, node_t> view() const noexcept;

    node_t n() const noexcept {
        return numberOfNodes;
    }

private:
    index_t* rowPointer{};
    node_t*  columnIndices{};
    WeightT* weights{};
    node_t numberOfNodes{};
    std::uint64_t numberOfEdgeEntries{};
};

} // namespace NetworKit::GPU

#endif // NETWORKIT_GPU_DEVICE_GRAPH_CSR_HPP_
