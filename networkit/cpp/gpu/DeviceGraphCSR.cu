/*  DeviceGraphCSR.cpp
 *
 *  Created on: 17.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#include <networkit/gpu/DeviceGraphCSR.hpp>
#include <networkit/gpu/CudaHelpers.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace NetworKit::GPU {

template <typename WeightT>
DeviceGraphCSR<WeightT>::DeviceGraphCSR(const HG &hostGraph) {
    if (hostGraph.rowPointer.empty()) {
        numberOfNodes = 0;
        numberOfEdgeEntries = 0;
        rowPointer = nullptr;
        columnIndices = nullptr;
        weights = nullptr;
        return;
    }

    const std::size_t hostNumberOfNodes = hostGraph.rowPointer.size() - 1;
    const std::size_t hostNumberOfEdges = hostGraph.columnIndices.size();

    // Basic consistency checks
    if (static_cast<std::uint64_t>(hostGraph.rowPointer.back())
        != static_cast<std::uint64_t>(hostNumberOfEdges)) {
        throw std::runtime_error(
            "DeviceGraphCSR: host CSR inconsistent: rowPointer.back() != columnIndices.size().");
    }
    if (hostGraph.hasWeights) {
        if (hostGraph.weights.size() != hostNumberOfEdges) {
            throw std::runtime_error(
                "DeviceGraphCSR: host CSR inconsistent: weights.size() != columnIndices.size().");
        }
    }

    numberOfNodes = static_cast<node_t>(hostNumberOfNodes);
    numberOfEdgeEntries = static_cast<std::uint64_t>(hostNumberOfEdges);

    // Allocate device arrays
    cudaCheck(cudaMalloc(&rowPointer, hostGraph.rowPointer.size() * sizeof(index_t)),
              "cudaMalloc rowPointer");
    cudaCheck(cudaMalloc(&columnIndices, hostNumberOfEdges * sizeof(node_t)),
              "cudaMalloc columnIndices");

    if (hostGraph.hasWeights) {
        cudaCheck(cudaMalloc(&weights, hostNumberOfEdges * sizeof(WeightT)), "cudaMalloc weights");
    } else {
        weights = nullptr;
    }

    // Copy host -> device
    cudaCheck(cudaMemcpy(rowPointer, hostGraph.rowPointer.data(),
                         hostGraph.rowPointer.size() * sizeof(index_t), cudaMemcpyHostToDevice),
              "cudaMemcpy rowPointer Host2Device");

    cudaCheck(cudaMemcpy(columnIndices, hostGraph.columnIndices.data(),
                         hostNumberOfEdges * sizeof(node_t), cudaMemcpyHostToDevice),
              "cudaMemcpy columnIndices Host2Device");

    if (hostGraph.hasWeights) {
        cudaCheck(cudaMemcpy(weights, hostGraph.weights.data(), hostNumberOfEdges * sizeof(WeightT),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy weights Host2Device");
    }
}

template <typename WeightT>
DeviceGraphCSR<WeightT>::~DeviceGraphCSR() {
    if (rowPointer)
        cudaFree(rowPointer);
    if (columnIndices)
        cudaFree(columnIndices);
    if (weights)
        cudaFree(weights);

    rowPointer = nullptr;
    columnIndices = nullptr;
    weights = nullptr;
    numberOfNodes = 0;
    numberOfEdgeEntries = 0;
}

template <typename WeightT>
DeviceGraphCSR<WeightT>::DeviceGraphCSR(DeviceGraphCSR &&other) noexcept {
    rowPointer = other.rowPointer;
    columnIndices = other.columnIndices;
    weights = other.weights;
    numberOfNodes = other.numberOfNodes;
    numberOfEdgeEntries = other.numberOfEdgeEntries;

    other.rowPointer = nullptr;
    other.columnIndices = nullptr;
    other.weights = nullptr;
    other.numberOfNodes = 0;
    other.numberOfEdgeEntries = 0;
}

template <typename WeightT>
DeviceGraphCSR<WeightT> &DeviceGraphCSR<WeightT>::operator=(DeviceGraphCSR &&other) noexcept {
    if (this == &other)
        return *this;

    if (rowPointer)
        cudaFree(rowPointer);
    if (columnIndices)
        cudaFree(columnIndices);
    if (weights)
        cudaFree(weights);

    rowPointer = other.rowPointer;
    columnIndices = other.columnIndices;
    weights = other.weights;
    numberOfNodes = other.numberOfNodes;
    numberOfEdgeEntries = other.numberOfEdgeEntries;

    other.rowPointer = nullptr;
    other.columnIndices = nullptr;
    other.weights = nullptr;
    other.numberOfNodes = 0;
    other.numberOfEdgeEntries = 0;

    return *this;
}

template <typename WeightT>
DeviceCSRView<WeightT, typename DeviceGraphCSR<WeightT>::index_t,
              typename DeviceGraphCSR<WeightT>::node_t>
DeviceGraphCSR<WeightT>::view() const noexcept {
    return DeviceCSRView<WeightT, index_t, node_t>{rowPointer, columnIndices, weights,
                                                   numberOfNodes, numberOfEdgeEntries};
}

template class DeviceGraphCSR<float>;
// template class DeviceGraphCSR<double>;

} // namespace NetworKit::GPU
