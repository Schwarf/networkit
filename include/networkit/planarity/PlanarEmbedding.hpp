//
// Created by andreas on 06.05.25.
//

#ifndef PLANAREMBEDDING_HPP
#define PLANAREMBEDDING_HPP

#include <networkit/graph/Graph.hpp>
namespace NetworKit {
class PlanarEmbedding {
public:
    PlanarEmbedding(){}
    explicit PlanarEmbedding(count n)
        : graph(n, /*weighted=*/false, /*directed=*/true, /*fast=*/false),
          clockWiseNeighborOrder(n) {
        for (auto &neighbors : clockWiseNeighborOrder) {
            neighbors.reserve(n);
        }
    }
    /**
     * Add the half‐edge source→target into the embedding.
     *
     * @param source                 the tail of the half‐edge
     * @param target                 the head of the half‐edge
     * @param is_counter_clock_wise  if true, insert target immediately
     *                               before reference_node in the clockwise list
     *                               (i.e. counterclockwise relative to the reference_node);
     *                               if false, insert immediately after.
     * @param reference_node         must already be a neighbor of source
     *                               when clockWiseNeighborOrder[source] is nonempty.
     */
    void addHalfEdge(node source, node target, bool is_counter_clock_wise, node reference_node);

    std::vector<std::vector<node>> getEmbedding() const;
    std::vector<node> getClockWiseOrderedNeighbors(node u) const;
    friend bool operator==(const PlanarEmbedding &embedding1, const PlanarEmbedding &embedding2);
private:
    Graph graph;
    std::vector<std::vector<node>> clockWiseNeighborOrder;
};

} // namespace NetworKit
#endif // PLANAREMBEDDING_HPP
