//
// Created by andreas on 06.05.25.
//

#ifndef PLANAREMBEDDING_HPP
#define PLANAREMBEDDING_HPP

#include <networkit/graph/Graph.hpp>
namespace NetworKit {

class PlanarEmbedding {
public:
    /**
     * Construct an embedding for the given graph.
     * The graph must outlive this embedding.
     */
    explicit PlanarEmbedding(const Graph &G);

    /**
     * Add a directed half-edge u->v into the clockwise cycle at u.
     * If cwInsert==true, insert v after ref in the CW list;
     * else insert before ref (i.e. CCW of ref).
     */
    void addHalfEdge(node source, node target, bool clockwiseInsert, node ref);

    /** Access underlying graph */
    const Graph &getGraph() const { return graph; }

    /** Access per-vertex clockwise neighbor order */
    const std::vector<std::vector<node>> &getClockwiseOrder() const;

    /** Validate embedding invariants:
     *  - half-edges come in opposite pairs,
     *  - each cwOrder[u] forms a simple cycle,
     *  - faces counted via CW-walks satisfy Euler's formula.
     */
    void checkStructure() const;

private:
    const Graph &graph;
    std::vector<std::vector<node>> clockwiseOrder;
};

} // namespace NetworKit
#endif // PLANAREMBEDDING_HPP
