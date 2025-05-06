//
// Created by andreas on 06.05.25.
//

#ifndef PLANAREMBEDDING_HPP
#define PLANAREMBEDDING_HPP
#include <networkit/graph/Graph.hpp>
namespace NetworKit {
class PlanarEmbedding {
public:
    explicit PlanarEmbedding::PlanarEmbedding(count n) {
        graph = Graph(n, /*weighted=*/false, /*directed=*/true, /*fast=*/false);
        clockWiseNeighborOrder.resize(n);
        for (auto &L : clockWiseNeighborOrder) {
            L.reserve(n);
        }
    }
    /**
     * Add the half‐edge source→target into the embedding.
     *
     * @param source                 the tail of the half‐edge
     * @param target                 the head of the half‐edge
     * @param is_counter_clock_wise  if true, insert target immediately
     *                               before reference_node in the CW list
     *                               (i.e. CCW relative to ref);
     *                               if false, insert immediately after.
     * @param reference_node         must already be a neighbor of source
     *                               when neighborOrder[source] is nonempty.
     */
    void addHalfEdge(node source, node target, bool is_counter_clock_wise, node reference_node);

private:
    Graph graph;
    std::vector<std::vector<node>> clockWiseNeighborOrder;
};
} // namespace NetworKit
#endif // PLANAREMBEDDING_HPP
