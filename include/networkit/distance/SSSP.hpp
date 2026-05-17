/*
 * SSSP.hpp
 *
 *  Created on: 15.04.2014
 *      Author: cls
 */

#ifndef NETWORKIT_DISTANCE_SSSP_HPP_
#define NETWORKIT_DISTANCE_SSSP_HPP_

#include <algorithm>
#include <cassert>
#include <limits>
#include <set>
#include <stack>

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Multiprecision.hpp>
#include <networkit/base/Algorithm.hpp>
#include <networkit/graph/Graph.hpp>

namespace NetworKit {

/**
 * @ingroup distance
 * Abstract base class for single-source shortest path algorithms.
 */
template <typename GraphType>
class SSSPBase : public Algorithm {
    using NodeType = typename GraphType::node_type;
    using EdgeWeightType = typename GraphType::edge_weight_type;

public:
    /**
     * Creates the SSSP class for @a G and source @a s.
     *
     * @param G The graph.
     * @param source The source node.
     * @param storePaths Paths are reconstructable and the number of paths is
     * stored.
     * @param storeNodesSortedByDistance Store a vector of nodes ordered in
     * increasing distance from the source.
     * @param target The target node.
     */
    SSSPBase(const GraphType &G, NodeType source, bool storePaths = true,
             bool storeNodesSortedByDistance = false,
             NodeType target = std::numeric_limits<NodeType>::max());

    ~SSSPBase() override = default;

    /** Computes the shortest paths from the source to all other nodes. */
    void run() override = 0;

    /**
     * Returns a vector of weighted distances from the source node, i.e. the
     * length of the shortest path from the source node to any other node.
     *
     * @return The weighted distances from the source node to any other node in
     * the graph.
     */
    const std::vector<EdgeWeightType> &getDistances();

    /**
     * Returns the distance from the source node to @a t.
     * @param  t Target node.
     * @return The distance from source to target node @a t.
     */
    EdgeWeightType distance(NodeType t) const;

    /**
     * Returns the number of shortest paths between the source node and @a t.
     * @param  t Target node.
     * @return The number of shortest paths between source and @a t.
     */
    bigfloat numberOfPaths(NodeType t) const;

    /**
     * Returns the number of shortest paths between the source node and @a t
     * as a double value. Workaround for Cython
     * @param  t Target node.
     * @return The number of shortest paths between source and @a t.
     */
    double _numberOfPaths(NodeType t) const;

    /**
     * Returns the predecessor nodes of @a t on all shortest paths from source
     * to @a t.
     * @param t Target node.
     * @return The predecessors of @a t on all shortest paths from source to @a
     * t.
     */
    const std::vector<NodeType> &getPredecessors(NodeType t) const;

    /**
     * Returns a shortest path from source to @a t and an empty path if source
     * and @a t are not connected.
     *
     * @param t Target node.
     * @param forward If @c true (default) the path is directed from source to
     * @a t, otherwise the path is reversed.
     * @return A shortest path from source to @a t or an empty path.
     */
    std::vector<NodeType> getPath(NodeType t, bool forward = true) const;

    /**
     * Returns all shortest paths from source to @a t and an empty set if source
     * and @a t are not connected.
     *
     * @param t Target node.
     * @param forward If @c true (default) the path is directed from source to
     * @a t, otherwise the path is reversed.
     * @return All shortest paths from source node to target node @a t.
     */
    std::set<std::vector<NodeType>> getPaths(NodeType t, bool forward = true) const;

    /* Returns the number of shortest paths to node t.*/
    bigfloat getNumberOfPaths(NodeType t) const;

    /**
     * Returns a vector of nodes ordered in increasing distance from the source.
     *
     * For this functionality to be available, storeNodesSortedByDistance has
     * to be set to true in the constructor. There are no guarantees regarding
     * the ordering of two nodes with the same distance to the source.
     *
     * @return vector of nodes ordered in increasing distance from the source
     */
    const std::vector<NodeType> &getNodesSortedByDistance() const;

    /**
     * Returns the number of nodes reached by the source.
     *
     * @return Number of nodes reached by the source.
     */
    count getReachableNodes() const {
        assureFinished();
        return reachedNodes;
    }

    /**
     * Sets a new source.
     *
     * @param newSource The new source node.
     */
    void setSource(NodeType newSource) {
        if (!G->hasNode(newSource))
            throw std::runtime_error("Error: node not in the graph.");
        source = newSource;
    }

    /**
     * Sets a new target.
     */
    void setTarget(NodeType newTarget) {
        if (!G->hasNode(newTarget))
            throw std::runtime_error("Error: node not in the graph.");
        target = newTarget;
    }

    /**
     * Returns the sum of distances from the source node node to the reached
     * nodes.
     */
    double getSumOfDistances() const {
        assureFinished();
        return sumDist;
    }

protected:
    const GraphType *G;
    NodeType source;
    NodeType target;
    double sumDist;
    count reachedNodes;
    std::vector<EdgeWeightType> distances;
    std::vector<std::vector<NodeType>> previous; // predecessors on shortest path
    std::vector<bigfloat> npaths;

    std::vector<NodeType> nodesSortedByDistance;

    bool storePaths;                 //!< if true, paths are reconstructable and the number of
                                     //!< paths is stored
    bool storeNodesSortedByDistance; //!< if true, store a vector of nodes
                                     //!< ordered in increasing distance from
                                     //!< the source
};

template <typename GraphType>
SSSPBase<GraphType>::SSSPBase(const GraphType &G, NodeType source, bool storePaths,
                              bool storeNodesSortedByDistance, NodeType target)
    : Algorithm(), G(&G), source(source), target(target), storePaths(storePaths),
      storeNodesSortedByDistance(storeNodesSortedByDistance) {}

template <typename GraphType>
const std::vector<typename GraphType::edge_weight_type> &SSSPBase<GraphType>::getDistances() {
    return distances;
}

template <typename GraphType>
typename GraphType::edge_weight_type SSSPBase<GraphType>::distance(NodeType t) const {
    return distances[t];
}

template <typename GraphType>
bigfloat SSSPBase<GraphType>::numberOfPaths(NodeType t) const {
    if (!storePaths) {
        throw std::runtime_error("number of paths have not been stored");
    }
    return npaths[t];
}

template <typename GraphType>
double SSSPBase<GraphType>::_numberOfPaths(NodeType t) const {
    if (!storePaths) {
        throw std::runtime_error("number of paths have not been stored");
    }
    bigfloat limit = std::numeric_limits<double>::max();
    if (npaths[t] > limit) {
        throw std::overflow_error("number of paths do not fit into a double");
    }
    double res;
    npaths[t].ToDouble(res);
    return res;
}

template <typename GraphType>
const std::vector<typename GraphType::node_type> &SSSPBase<GraphType>::getPredecessors(
    NodeType t) const {
    if (!storePaths) {
        throw std::runtime_error("predecessors have not been stored");
    }
    return previous[t];
}

template <typename GraphType>
std::vector<typename GraphType::node_type> SSSPBase<GraphType>::getPath(NodeType t,
                                                                        bool forward) const {
    if (!storePaths) {
        throw std::runtime_error("paths have not been stored");
    }
    std::vector<NodeType> path;
    if (previous[t].empty()) { // t is not reachable from source
        WARN("there is no path from ", source, " to ", t);
        return path;
    }
    NodeType v = t;
    while (v != source) {
        path.push_back(v);
        v = previous[v].front();
    }
    path.push_back(source);

    if (forward) {
        std::reverse(path.begin(), path.end());
    }
    return path;
}

template <typename GraphType>
std::set<std::vector<typename GraphType::node_type>> SSSPBase<GraphType>::getPaths(
    NodeType t, bool forward) const {

    std::set<std::vector<NodeType>> paths;
    if (previous[t].empty()) { // t is not reachable from source
        WARN("there is no path from ", source, " to ", t);
        return paths;
    }

    std::vector<NodeType> targetPredecessors = previous[t];

#pragma omp parallel for schedule(dynamic)
    for (omp_index i = 0; i < static_cast<omp_index>(targetPredecessors.size()); ++i) {

        std::stack<std::vector<NodeType>> stack;
        std::vector<std::vector<NodeType>> currPaths;

        NodeType pred = targetPredecessors[i];
        if (pred == source) {
            currPaths.push_back({t, pred});
        } else {
            stack.push({t, pred});
        }

        while (!stack.empty()) {

            NodeType topPathLastNode = stack.top().back();

            if (topPathLastNode == source) {
                currPaths.push_back(stack.top());
                stack.pop();
                continue;
            }

            std::vector<NodeType> topPath = stack.top();
            stack.pop();

            std::vector<NodeType> currPredecessors = previous[topPath.back()];

            for (NodeType currPredecessor : currPredecessors) {
                std::vector<NodeType> suffix(topPath);
                suffix.push_back(currPredecessor);
                stack.push(suffix);
            }
        }

#pragma omp critical
        paths.insert(currPaths.begin(), currPaths.end());
    }

    if (forward) {
        std::set<std::vector<NodeType>> reversedPaths;
        for (std::vector<NodeType> path : paths) {
            std::reverse(std::begin(path), std::end(path));
            reversedPaths.insert(path);
        }
        paths = reversedPaths;
    }

    return paths;
}

template <typename GraphType>
const std::vector<typename GraphType::node_type> &SSSPBase<GraphType>::getNodesSortedByDistance()
    const {
    if (!storeNodesSortedByDistance) {
        throw std::runtime_error("Nodes sorted by distance have not been stored. Set "
                                 "storeNodesSortedByDistance in the constructor to true to enable "
                                 "this behaviour.");
    }

    assert(!nodesSortedByDistance.empty());
    return nodesSortedByDistance;
}

template <typename GraphType>
bigfloat SSSPBase<GraphType>::getNumberOfPaths(NodeType t) const {
    return npaths[t];
}

using SSSP = SSSPBase<Graph>;

} /* namespace NetworKit */

#endif // NETWORKIT_DISTANCE_SSSP_HPP_
