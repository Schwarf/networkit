//
// Created by andreas on 06.05.25.
//
#include <networkit/planarity/PlanarEmbedding.hpp>
namespace NetworKit {

void PlanarEmbedding::addHalfEdge(node source, node target, bool is_counter_clock_wise,
                                  node reference_node) {
    // add the directed edge
    graph.addEdge(source, target);

    // splice `target` into source’s clockwise neighbor list
    if (auto &sourceNeighbors = clockWiseNeighborOrder[source]; sourceNeighbors.empty()) {
        // first half‐edge out of `source`
        sourceNeighbors.push_back(target);
    } else {
        auto it = std::find(sourceNeighbors.begin(), sourceNeighbors.end(), reference_node);
        if (it == sourceNeighbors.end())
            throw std::invalid_argument("addHalfEdge: reference_node not found");
        if (is_counter_clock_wise) {
            // counterclockwise insertion = place before reference_node in clockwise list
            sourceNeighbors.insert(it, target);
        } else {
            // clockwise insertion = place after reference_node
            sourceNeighbors.insert(std::next(it), target);
        }
    }
}

} // namespace NetworKit
