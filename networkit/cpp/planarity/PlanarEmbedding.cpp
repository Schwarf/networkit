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
        const auto it = std::ranges::find(sourceNeighbors, reference_node);
        if (it == sourceNeighbors.end())
            throw std::invalid_argument("addHalfEdge: The reference_node has not been found!");
        if (is_counter_clock_wise) {
            // counterclockwise insertion = place before reference_node in clockwise list
            sourceNeighbors.insert(it, target);
        } else {
            // clockwise insertion = place after reference_node
            sourceNeighbors.insert(std::next(it), target);
        }
    }
}

std::vector<std::vector<node>> PlanarEmbedding::getEmbedding() const {
    return clockWiseNeighborOrder;
}

std::vector<node> PlanarEmbedding::getClockWiseOrderedNeighbors(node u) const {
    if (u < graph.numberOfNodes())
        return clockWiseNeighborOrder[u];
    throw std::runtime_error("getClockWiseOrderedNeighbors: Node u is not in Embedding!");
}

inline bool operator==(PlanarEmbedding const &embedding1, PlanarEmbedding const &embedding2) {
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
        if (isShift(A, B))
            return true;

        // 4) Test mirror (reverse B then rotation)
        auto mirror = B;
        std::reverse(mirror.begin(), mirror.end());
        return isShift(A, mirror);
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
