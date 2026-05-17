/*
 * APSP.hpp
 *
 *  Created on: 07.07.2015
 *      Author: Arie Slobbe
 */

#ifndef NETWORKIT_DISTANCE_APSP_HPP_
#define NETWORKIT_DISTANCE_APSP_HPP_

#include <omp.h>

#include <memory>

#include <networkit/base/Algorithm.hpp>
#include <networkit/distance/BFS.hpp>
#include <networkit/distance/Dijkstra.hpp>
#include <networkit/distance/SSSP.hpp>
#include <networkit/graph/Graph.hpp>

namespace NetworKit {

/**
 * @ingroup distance
 * Class for all-pair shortest path algorithm.
 */
template <typename GraphType>
class APSPBase : public Algorithm {
    using NodeType = typename GraphType::node_type;
    using EdgeWeightType = typename GraphType::edge_weight_type;

public:
    /**
     * Creates the APSP class for @a G.
     *
     * @param G The graph.
     */
    APSPBase(const GraphType &G) : Algorithm(), G(G) {}

    ~APSPBase() override = default;

    /**
     * Computes the shortest paths from each node to all other nodes.
     * The algorithm is parallel.
     */
    void run() override;

    /**
     * Returns a vector of weighted distances between node pairs.
     *
     * @return The shortest-path distances from each node to any other node in
     * the graph.
     */
    const std::vector<std::vector<EdgeWeightType>> &getDistances() const {
        assureFinished();
        return distances;
    }

    /**
     * Returns the distance from u to v or infinity if u and v are not
     * connected.
     *
     */
    EdgeWeightType getDistance(NodeType u, NodeType v) const {
        assureFinished();
        return distances[u][v];
    }

protected:
    const GraphType &G;
    std::vector<std::vector<EdgeWeightType>> distances;
    std::vector<std::unique_ptr<SSSPBase<GraphType>>> sssps;
};

template <typename GraphType>
void APSPBase<GraphType>::run() {
    const count n = G.upperNodeIdBound();
    distances.assign(n, std::vector<EdgeWeightType>(n));

    sssps.resize(omp_get_max_threads());
#pragma omp parallel
    {
        omp_index i = omp_get_thread_num();
        if (G.isWeighted())
            sssps[i] =
                std::make_unique<DijkstraBase<GraphType>>(G, static_cast<NodeType>(0), false);
        else
            sssps[i] =
                std::make_unique<BFSBase<GraphType>>(G, static_cast<NodeType>(0), false);
    }

    G.parallelForNodes([&](NodeType source) {
        auto sssp = sssps[omp_get_thread_num()].get();
        sssp->setSource(source);
        sssp->run();
        distances[source] = sssp->getDistances();
    });

    hasRun = true;
}

using APSP = APSPBase<Graph>;

} /* namespace NetworKit */

#endif // NETWORKIT_DISTANCE_APSP_HPP_
