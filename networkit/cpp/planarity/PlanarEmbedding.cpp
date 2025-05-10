//
// Created by andreas on 06.05.25.
//
#include <set>
#include <networkit/planarity/PlanarEmbedding.hpp>

namespace NetworKit {

PlanarEmbedding::PlanarEmbedding(const Graph &G) : graph(G) {
    clockwiseOrder.resize(graph.numberOfNodes());
    for (auto &neighbors : clockwiseOrder) {
        neighbors.reserve(G.numberOfNodes());
    }
}

void PlanarEmbedding::addHalfEdge(node source, node target, bool clockwiseInsert, node ref) {
    // Add directed edge and maintain CW order
    if (!graph.hasEdge(source, target)) {
        throw std::runtime_error("addHalfEdge: graph does not contain edge "
                                 + std::to_string(source) + "->" + std::to_string(target));
    }
    auto &neighbors = clockwiseOrder[source];
    if (neighbors.empty()) {
        neighbors.push_back(target);
    } else {
        auto it = std::ranges::find(neighbors, ref);
        if (it == neighbors.end()) {
            throw std::runtime_error("addHalfEdge: reference node not found");
        }
        if (clockwiseInsert)
            ++it;
        neighbors.insert(it, target);
    }
}

const std::vector<std::vector<node>> &PlanarEmbedding::getClockwiseOrder() const {
    return clockwiseOrder;
}

void PlanarEmbedding::checkStructure() const {}

} // namespace NetworKit
