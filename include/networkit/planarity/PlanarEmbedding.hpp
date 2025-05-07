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

    std::vector<std::vector<node>> getClockwiseNeighborOrder() const;
    std::vector<node> getClockWiseOrderedNeighborOf(node u) const;
    Graph getGraph() const;
    friend bool operator==(const PlanarEmbedding &embedding1, const PlanarEmbedding &embedding2);
private:
    Graph graph{};
    std::vector<std::vector<node>> clockWiseNeighborOrder{};
};

inline bool operator==(const PlanarEmbedding &embedding1, const PlanarEmbedding &embedding2) {
    // 1) quick check: same number of nodes, same number of edges?
    if (embedding1.graph.numberOfNodes() != embedding2.graph.numberOfNodes())
        return false;
    if (embedding1.graph.numberOfEdges() != embedding2.graph.numberOfEdges())
        return false;

    // 2) check each vertex’s cyclic neighbor order (up to rotation+mirroring)
    auto areCyclesEquivalent = [](const std::vector<node> &A, const std::vector<node> &B) {
        // 1) Quick size check
        if (A.size() != B.size())
            return false;
        const count n = A.size();
        if (n == 0)
            return true;

        // 2) Inner lambda to test “is A a cyclic shift of Y?”
        auto const isShift = [&](const std::vector<node> &X, const std::vector<node> &Y) {
            for (size_t shift = 0; shift < n; ++shift) {
                if (Y[shift] != X[0])
                    continue;
                bool ok = true;
                for (size_t i = 1; i < n; ++i) {
                    if (X[i] != Y[(shift + i) % n]) {
                        ok = false;
                        break;
                    }
                }
                if (ok)
                    return true;
            }
            return false;
        };
        return isShift(A, B);


        // 4) Test mirror (reverse B then rotation)
        // TODO: DO we allow mirrors or not?
        // auto mirror = B;
        // std::reverse(mirror.begin(), mirror.end());
        // return isShift(A, mirror);
    };

    // 5) Test that graphs are identical
    bool foundMismatch{};
    for (node currentNode = 0; currentNode < embedding1.graph.numberOfNodes(); ++currentNode) {
        embedding1.graph.forNeighborsOf(currentNode, [&](node neighbor) {
            if (!embedding2.graph.hasEdge(currentNode, neighbor)) {
                foundMismatch = true;
            }
        });
        if (foundMismatch)
            return false;
    }

    for (auto u = 0u; u < embedding1.graph.numberOfNodes(); ++u) {
        auto const &neighbors1 = embedding1.clockWiseNeighborOrder[u];
        auto const &neighbors2 = embedding2.clockWiseNeighborOrder[u];
        if (!areCyclesEquivalent(neighbors1, neighbors2))
            return false;
    }
    return true;
}

} // namespace NetworKit
#endif // PLANAREMBEDDING_HPP
