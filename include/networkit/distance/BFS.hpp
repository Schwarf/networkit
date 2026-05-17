/*
 * BFS.hpp
 *
 *  Created on: Jul 23, 2013
 *      Author: Henning
 */

#ifndef NETWORKIT_DISTANCE_BFS_HPP_
#define NETWORKIT_DISTANCE_BFS_HPP_

#include <limits>
#include <queue>

#include <networkit/distance/SSSP.hpp>

namespace NetworKit {

/**
 * @ingroup distance
 * The BFS class is used to do a breadth-first search on a Graph from a given
 * source node.
 */
template <typename GraphType>
class BFSBase final : public SSSPBase<GraphType> {
    using NodeType = typename GraphType::node_type;
    using EdgeWeightType = typename GraphType::edge_weight_type;

public:
    /**
     * Constructs the BFS class for @a G and source node @a source.
     *
     * @param G The graph
     * @param source The source node of the breadth-first search
     * @param storePaths Paths are reconstructable and the number of paths is
     * stored.
     * @param storeNodesSortedByDistance Store a vector of nodes ordered in
     * increasing distance from the source.
     * @param target The target node.
     */
    BFSBase(const GraphType &G, NodeType source, bool storePaths = true,
            bool storeNodesSortedByDistance = false,
            NodeType target = std::numeric_limits<NodeType>::max());

    /**
     * Breadth-first search from @a source.
     */
    void run() override;
};

template <typename GraphType>
BFSBase<GraphType>::BFSBase(const GraphType &G, NodeType source, bool storePaths,
                            bool storeNodesSortedByDistance, NodeType target)
    : SSSPBase<GraphType>(G, source, storePaths, storeNodesSortedByDistance, target) {}

template <typename GraphType>
void BFSBase<GraphType>::run() {
    const count z = this->G->upperNodeIdBound();
    this->reachedNodes = 1;
    this->sumDist = 0.;

    const auto infDist = std::numeric_limits<EdgeWeightType>::max();
    std::fill(this->distances.begin(), this->distances.end(), infDist);

    if (this->distances.size() < z)
        this->distances.resize(z, infDist);

    if (this->storePaths) {
        this->previous.clear();
        this->previous.resize(z);
        this->npaths.clear();
        this->npaths.resize(z, 0);
        this->npaths[this->source] = 1;
    }

    if (this->storeNodesSortedByDistance) {
        std::vector<NodeType> empty;
        std::swap(this->nodesSortedByDistance, empty);
    }

    std::queue<NodeType> q;
    q.push(this->source);
    this->distances[this->source] = 0.;

    const bool breakWhenFound = (this->target != std::numeric_limits<NodeType>::max());
    while (!q.empty()) {
        NodeType u = q.front();
        q.pop();

        if (this->storeNodesSortedByDistance) {
            this->nodesSortedByDistance.push_back(u);
        }
        if (breakWhenFound && this->target == u) {
            break;
        }

        // Insert untouched neighbors into queue.
        this->G->forNeighborsOf(u, [&](NodeType v) {
            if (this->distances[v] == infDist) {
                q.push(v);
                this->distances[v] = this->distances[u] + 1.;
                this->sumDist += this->distances[v];
                ++this->reachedNodes;
                if (this->storePaths) {
                    this->previous[v] = {u};
                    this->npaths[v] = this->npaths[u];
                }
            } else if (this->storePaths && (this->distances[v] == this->distances[u] + 1.)) {
                this->previous[v].push_back(u);
                this->npaths[v] += this->npaths[u];
            }
        });
    }

    this->hasRun = true;
}

using BFS = BFSBase<Graph>;
} /* namespace NetworKit */
#endif // NETWORKIT_DISTANCE_BFS_HPP_
