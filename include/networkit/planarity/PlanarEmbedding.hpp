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
     * Construct an empty embedding on n nodes.
     */
    explicit PlanarEmbedding(count n);

    /**
     * Add a directed half-edge u->v into the clockwise cycle at u.
     * If cwInsert==true, insert v after ref in the CW list;
     * else insert before ref (i.e. CCW of ref).
     */
    void addHalfEdge(node u, node v, bool cwInsert, node ref);

    /** Access underlying graph */
    const Graph& getGraph() const;

    /** Access per-vertex clockwise neighbor order */
    const std::vector<std::vector<node>>& getClockwiseOrder() const;

    /** Validate embedding invariants:
     *  - half-edges come in opposite pairs,
     *  - each cwOrder[u] forms a simple cycle,
     *  - faces counted via CW-walks satisfy Euler's formula.
     */
    void checkStructure() const;

private:
    Graph graph;
    std::vector<std::vector<node>> cwOrder;
};


} // namespace NetworKit
#endif // PLANAREMBEDDING_HPP
