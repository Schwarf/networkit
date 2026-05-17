/*
 * Dijkstra.hpp
 *
 *  Created on: Jul 23, 2013
 *      Author: Henning, Christian Staudt
 */

#ifndef NETWORKIT_DISTANCE_DIJKSTRA_HPP_
#define NETWORKIT_DISTANCE_DIJKSTRA_HPP_

#include <limits>

#include <tlx/container/d_ary_addressable_int_heap.hpp>

#include <networkit/auxiliary/VectorComparator.hpp>
#include <networkit/distance/SSSP.hpp>

namespace NetworKit {

/**
 * @ingroup distance
 * Dijkstra's SSSP algorithm.
 */
template <typename GraphType>
class DijkstraBase final : public SSSPBase<GraphType> {
    using NodeType = typename GraphType::node_type;
    using EdgeWeightType = typename GraphType::edge_weight_type;

public:
    /**
     * Creates the Dijkstra class for @a G and the source node @a source.
     *
     * @param G The graph.
     * @param source The source node.
     * @param storePaths Paths are reconstructable and the number of paths is
     *        stored.
     * @param storeNodesSortedByDistance Store a vector of nodes ordered in
     *        increasing distance from the source.
     * @param target The target node.
     */
    DijkstraBase(const GraphType &G, NodeType source, bool storePaths = true,
                 bool storeNodesSortedByDistance = false,
                 NodeType target = std::numeric_limits<NodeType>::max());

    /**
     * Performs the Dijkstra SSSP algorithm on the graph given in the
     * constructor.
     */
    void run() override;

private:
    tlx::d_ary_addressable_int_heap<NodeType, 2, Aux::LessInVector<EdgeWeightType>> heap;
};

template <typename GraphType>
DijkstraBase<GraphType>::DijkstraBase(const GraphType &G, NodeType source, bool storePaths,
                                      bool storeNodesSortedByDistance, NodeType target)
    : SSSPBase<GraphType>(G, source, storePaths, storeNodesSortedByDistance, target),
      heap(Aux::LessInVector<EdgeWeightType>{this->distances}) {}

template <typename GraphType>
void DijkstraBase<GraphType>::run() {

    TRACE("initializing Dijkstra data structures");

    const auto infDist = std::numeric_limits<EdgeWeightType>::max();
    std::fill(this->distances.begin(), this->distances.end(), infDist);

    if (this->distances.size() < this->G->upperNodeIdBound())
        this->distances.resize(this->G->upperNodeIdBound(), infDist);

    this->sumDist = 0.;
    this->reachedNodes = 1;

    if (this->storePaths) {
        this->previous.clear();
        this->previous.resize(this->G->upperNodeIdBound());
        this->npaths.clear();
        this->npaths.resize(this->G->upperNodeIdBound(), 0);
        this->npaths[this->source] = 1;
    }

    if (this->storeNodesSortedByDistance) {
        this->nodesSortedByDistance.clear();
        this->nodesSortedByDistance.reserve(this->G->upperNodeIdBound());
    }

    this->distances[this->source] = 0.;
    heap.clear();
    heap.push(this->source);

    auto initPath = [&](NodeType u, NodeType v) {
        if (this->storePaths) {
            this->previous[v] = {u};
            this->npaths[v] = this->npaths[u];
        }
    };
    const bool breakWhenFound = (this->target != std::numeric_limits<NodeType>::max());
    TRACE("traversing graph");
    do {
        TRACE("pq size: ", heap.size());
        NodeType u = heap.extract_top();
        this->sumDist += this->distances[u];
        TRACE("current node in Dijkstra: ", u);
        TRACE("pq size: ", heap.size());
        if ((breakWhenFound && this->target == u) || this->distances[u] == infDist)
            break;

        if (this->storeNodesSortedByDistance)
            this->nodesSortedByDistance.push_back(u);

        this->G->forNeighborsOf(u, [&](NodeType v, EdgeWeightType w) {
            const EdgeWeightType newDist = this->distances[u] + w;
            if (this->distances[v] == infDist) {
                this->distances[v] = newDist;
                heap.push(v);
                ++this->reachedNodes;
                if (this->storePaths)
                    initPath(u, v);
            } else if (this->distances[v] > newDist) {
                if (this->storePaths)
                    initPath(u, v);
                this->distances[v] = newDist;
                heap.update(v);
            } else if (this->storePaths && this->distances[v] == newDist) {
                this->previous[v].push_back(u);
                this->npaths[v] += this->npaths[u];
            }
        });
    } while (!heap.empty());

    this->hasRun = true;
}

using Dijkstra = DijkstraBase<Graph>;

} /* namespace NetworKit */
#endif // NETWORKIT_DISTANCE_DIJKSTRA_HPP_
