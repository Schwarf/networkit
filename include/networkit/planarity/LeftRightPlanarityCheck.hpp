/*  LeftRightPlanarityCheck.hpp
 *
 *  Created on: 03.01.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#ifndef NETWORKIT_PLANARITY_LEFT_RIGHT_PLANARITY_CHECK_HPP_
#define NETWORKIT_PLANARITY_LEFT_RIGHT_PLANARITY_CHECK_HPP_

#include <limits>
#include <stack>
#include <vector>

#include <networkit/base/Algorithm.hpp>
#include <networkit/graph/Graph.hpp>

namespace NetworKit {

template <class GraphType>
class LeftRightPlanarityCheck final : public Algorithm {
    using NodeType = GraphType::node_type;
    using EdgeType = GraphType::edge_weight_type;
    static constexpr NodeType NullNode = std::numeric_limits<NodeType>::max();
public:
    /**
     * Implements the left-right planarity test as described in
     * [citation](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7963e9feffe1c9362eb1a69010a5139d1da3661e).
     * This algorithm determines whether a graph is planar, i.e., whether it can be drawn on a plane
     * without any edges crossing. For an overview of planar graphs, refer to:
     * https://en.wikipedia.org/wiki/Planar_graph
     *
     * The algorithm achieves (almost) linear runtime complexity. The only non-linear component
     * arises from sorting the nodes of the depth-first search tree.
     *
     * @param G The input graph to test for planarity. The graph should be undirected.
     * @throws std::runtime_error if graph is not an undirected graph
     */
    LeftRightPlanarityCheck(const GraphType &G): graph(&G){
        if (G.isDirected()) {
            throw std::runtime_error("The graph is not an undirected graph.");
        }
        if (!G.hasEdgeIds()) {
            throw std::runtime_error("The graph has no edge IDs.");
        }

        numberOfEdges = graph->numberOfEdges();
        nestingDepth.resize(numberOfEdges, none);
        lowestPointEdge.resize(numberOfEdges, noneEdgeId);
        secondLowestPoint.resize(numberOfEdges, none);
        ref.resize(numberOfEdges, noneEdgeId);
        stackBottom.resize(numberOfEdges, {});

        lowestPoint.resize(numberOfEdges, none);

        // dfsGraph: directed view of DFS tree + back edges
        dfsGraph = GraphType(graph->numberOfNodes(), false, true, false);
    }

    void run() override;

    bool isPlanar() const {
        assureFinished();
        return isGraphPlanar;
    }

private:
    count numberOfEdges;
    static constexpr count noneHeight{std::numeric_limits<count>::max()};
    static constexpr edgeid noneEdgeId{std::numeric_limits<edgeid>::max()};

    struct Interval {
        edgeid low{noneEdgeId};
        edgeid high{noneEdgeId};

        Interval() = default;
        Interval(edgeid lowId, edgeid highId) : low(lowId), high(highId) {}

        bool isEmpty() const { return low == noneEdgeId && high == noneEdgeId; }

        friend bool operator==(const Interval &lhs, const Interval &rhs) {
            return lhs.low == rhs.low && lhs.high == rhs.high;
        }
    };

    struct ConflictPair {
        Interval left{};
        Interval right{};

        ConflictPair() = default;
        ConflictPair(const Interval &left, const Interval &right) : left(left), right(right) {}

        void swap() { std::swap(left, right); }

        friend bool operator==(const ConflictPair &lhs, const ConflictPair &rhs) {
            return lhs.left == rhs.left && lhs.right == rhs.right;
        }
    };

    const ConflictPair NoneConflictPair{Interval{}, Interval{}};

    const GraphType *graph;
    bool isGraphPlanar{false};

    void dfsOrientation(NodeType startNode);
    bool dfsTesting(NodeType startNode);

    bool applyConstraints(edgeid edgeId, edgeid parentEdgeId);
    void removeBackEdges(edgeid edgeId, NodeType parentNode);
    void sortAdjacencyListByNestingDepth();
    bool conflicting(const Interval &interval, edgeid edgeId);
    count getLowestLowPoint(const ConflictPair &conflictPair);

    // DFS / lowpoint state
    std::vector<count> heights; // per node
    std::vector<NodeType> roots;    // roots of DFS forest

    std::vector<count> lowestPoint;
    std::vector<count> secondLowestPoint;
    std::vector<edgeid> ref;
    std::vector<edgeid> lowestPointEdge;
    std::vector<count> nestingDepth;
    std::vector<ConflictPair> stackBottom;

    // Per-node parent edge + parent node in DFS tree
    std::vector<edgeid> parentEdgeIds; // parent edge for each node (noneEdgeId if root)
    std::vector<NodeType> parentNodes;     // parent node for each node (none if root)

    // For each *underlying* edge ID, remember the DFS orientation's head (v)
    std::vector<NodeType> edgeEndpoints; // edgeEndpoints[eid] = v in oriented DFS edge (u -> v)

    // Conflict stack
    std::stack<ConflictPair> stack;
    // DFS graph with tree/back orientation
    GraphType dfsGraph;
};

template <class GraphType>
void LeftRightPlanarityCheck<GraphType>::run() {
    // Euler-criterion: non-planar if m > 3n - 6 for n > 2
    if (graph->numberOfNodes() > 2 && graph->numberOfEdges() > 3 * graph->numberOfNodes() - 6) {
        hasRun = true;
        isGraphPlanar = false;
        return;
    }

    heights.assign(graph->upperNodeIdBound(), noneHeight);
    parentEdgeIds.assign(graph->upperNodeIdBound(), noneEdgeId);
    parentNodes.assign(graph->upperNodeIdBound(), NullNode);
    edgeEndpoints.assign(graph->upperEdgeIdBound(), NullNode);

    // DFS orientation
    graph->forNodes([&](NodeType currentNode) {
        if (heights[currentNode] == noneHeight) {
            heights[currentNode] = 0;
            roots.push_back(currentNode);
            this->dfsOrientation(currentNode);
        }
    });

    // Sort DFS adjacency by nesting depth
    sortAdjacencyListByNestingDepth();

    // Planarity testing DFS
    isGraphPlanar =
        std::ranges::all_of(roots, [this](node rootNode) { return dfsTesting(rootNode); });

    hasRun = true;
}


template <class GraphType>
void LeftRightPlanarityCheck<GraphType>::sortAdjacencyListByNestingDepth() {
    dfsGraph.forNodes([&](NodeType currentNode) {
        dfsGraph.sortNeighbors(currentNode, [&](NodeType neighbor1, NodeType neighbor2) {
            const edgeid e1 = graph->edgeId(currentNode, neighbor1);
            const edgeid e2 = graph->edgeId(currentNode, neighbor2);
            return nestingDepth[e1] < nestingDepth[e2];
        });
    });
}

template <class GraphType>
bool LeftRightPlanarityCheck<GraphType>::conflicting(const Interval &interval, edgeid edgeId) {
    if (interval.isEmpty()) {
        return false;
    }
    auto iteratorHigh = lowestPoint[interval.high];
    auto iteratorEdge = lowestPoint[edgeId];

    return !interval.isEmpty() && iteratorHigh != none && iteratorEdge != none
           && iteratorHigh > iteratorEdge;
}

template <class GraphType>
bool LeftRightPlanarityCheck<GraphType>::applyConstraints(const edgeid edgeId, const edgeid parentEdgeId) {
    ConflictPair tmpConflictPair{};

    // First phase: pop until stackBottom[edgeId], merging intervals on the right side.
    do {
        ConflictPair currentConflictPair = stack.top();
        stack.pop();

        if (!currentConflictPair.left.isEmpty()) {
            currentConflictPair.swap();
        }
        if (!currentConflictPair.left.isEmpty()) {
            return false;
        }

        auto rightLowIterator = lowestPoint[currentConflictPair.right.low];
        auto parentEdgeIterator = lowestPoint[parentEdgeId];
        ;

        if (rightLowIterator != none && parentEdgeIterator != none
            && rightLowIterator > parentEdgeIterator) {

            if (tmpConflictPair.right.isEmpty()) {
                tmpConflictPair.right = currentConflictPair.right;
            } else {
                ref[tmpConflictPair.right.low] = currentConflictPair.right.high;
            }

            tmpConflictPair.right.low = currentConflictPair.right.low;
        } else {
            ref[currentConflictPair.right.low] = lowestPointEdge[parentEdgeId];
        }
    } while (!stack.empty() && (stack.top() != stackBottom[edgeId]));

    // Second phase: pop while there are conflicts with edgeId, merging left/right intervals.
    while (!stack.empty()
           && (conflicting(stack.top().left, edgeId) || conflicting(stack.top().right, edgeId))) {

        auto currentConflictPair = stack.top();
        stack.pop();

        if (conflicting(currentConflictPair.right, edgeId)) {
            currentConflictPair.swap();
        }
        if (conflicting(currentConflictPair.right, edgeId)) {
            // Still conflicting after swap -> non-planar
            return false;
        }

        ref[tmpConflictPair.right.low] = currentConflictPair.right.high;

        if (currentConflictPair.right.low != noneEdgeId) {
            tmpConflictPair.right = currentConflictPair.right;
        }

        if (tmpConflictPair.left.isEmpty()) {
            tmpConflictPair.left = currentConflictPair.left;
        } else {
            ref[tmpConflictPair.left.low] = currentConflictPair.left.high;
        }
        tmpConflictPair.left.low = currentConflictPair.left.low;
    }

    if (!tmpConflictPair.left.isEmpty() || !tmpConflictPair.right.isEmpty()) {
        stack.push(tmpConflictPair);
    }

    return true;
}

template <class GraphType>
count LeftRightPlanarityCheck<GraphType>::getLowestLowPoint(const ConflictPair &conflictPair) {
    if (conflictPair.left.isEmpty()) {
        return lowestPoint[conflictPair.right.low];
    }
    if (conflictPair.right.isEmpty()) {
        return lowestPoint[conflictPair.left.low];
    }
    return std::min(lowestPoint[conflictPair.right.low], lowestPoint[conflictPair.left.low]);
}

template <class GraphType>
void LeftRightPlanarityCheck<GraphType>::removeBackEdges(const edgeid edgeId, const NodeType parentNode) {
    while (!stack.empty() && getLowestLowPoint(stack.top()) == heights[parentNode]) {
        stack.pop();
    }

    if (!stack.empty()) {
        auto conflictPair = stack.top();
        stack.pop();

        // Reduce left interval
        while (conflictPair.left.high != noneEdgeId
               && edgeEndpoints[conflictPair.left.high] == parentNode) {
            auto tmpEdgeId = ref[conflictPair.left.high];
            conflictPair.left.high = (tmpEdgeId != noneEdgeId) ? tmpEdgeId : noneEdgeId;
        }
        if (conflictPair.left.high == noneEdgeId && conflictPair.left.low != noneEdgeId) {
            ref[conflictPair.left.low] = conflictPair.right.low;
            conflictPair.left.low = noneEdgeId;
        }

        // Reduce right interval
        while (conflictPair.right.high != noneEdgeId
               && edgeEndpoints[conflictPair.right.high] == parentNode) {
            auto tmpEdgeId = ref[conflictPair.right.high];
            conflictPair.right.high = (tmpEdgeId != noneEdgeId) ? tmpEdgeId : noneEdgeId;
        }
        if (conflictPair.right.high == noneEdgeId && conflictPair.right.low != noneEdgeId) {
            ref[conflictPair.right.low] = conflictPair.left.low;
            conflictPair.right.low = noneEdgeId;
        }

        stack.push(conflictPair);
    }

    if (!stack.empty() && lowestPoint[edgeId] < heights[parentNode]) {
        const edgeid highestReturnEdgeLeft = stack.top().left.high;
        const edgeid highestReturnEdgeRight = stack.top().right.high;

        if (highestReturnEdgeLeft != noneEdgeId
            && (highestReturnEdgeRight == noneEdgeId
                || lowestPoint[highestReturnEdgeLeft] > lowestPoint[highestReturnEdgeRight])) {
            ref[edgeId] = highestReturnEdgeLeft;
        } else {
            ref[edgeId] = highestReturnEdgeRight;
        }
    }
}


template <class GraphType>
void LeftRightPlanarityCheck<GraphType>::dfsOrientation(NodeType startNode) {
    std::stack<NodeType> dfsStack;
    dfsStack.push(startNode);

    std::vector<edgeid> preprocessedEdges(numberOfEdges, noneEdgeId);
    do {
        const NodeType currentNode = dfsStack.top();
        dfsStack.pop();

        const edgeid parentEdgeId = parentEdgeIds[currentNode];

        for (NodeType neighbor : graph->neighborRange(currentNode)) {
            const edgeid edgeId = graph->edgeId(currentNode, neighbor);

            if (preprocessedEdges[edgeId] == noneEdgeId) {
                if (dfsGraph.hasEdge(currentNode, neighbor)
                    || dfsGraph.hasEdge(neighbor, currentNode)) {
                    continue;
                }

                dfsGraph.addEdge(currentNode, neighbor);
                edgeEndpoints[edgeId] = neighbor;

                lowestPoint[edgeId] = heights[currentNode];
                secondLowestPoint[edgeId] = heights[currentNode];

                if (heights[neighbor] == noneHeight) {
                    // Tree edge
                    parentEdgeIds[neighbor] = edgeId;
                    parentNodes[neighbor] = currentNode;

                    heights[neighbor] = heights[currentNode] + 1;

                    dfsStack.push(currentNode);
                    dfsStack.push(neighbor);

                    preprocessedEdges[edgeId] = edgeId;
                    break;
                }

                // Back edge: lowpoint is ancestor's height
                lowestPoint[edgeId] = heights[neighbor];
            }

            nestingDepth[edgeId] = 2 * lowestPoint[edgeId];
            if (secondLowestPoint[edgeId] < heights[currentNode]) {
                nestingDepth[edgeId] += 1;
            }

            if (parentEdgeId != noneEdgeId) {
                if (lowestPoint[edgeId] < lowestPoint[parentEdgeId]) {
                    secondLowestPoint[parentEdgeId] =
                        std::min(lowestPoint[parentEdgeId], secondLowestPoint[edgeId]);
                    lowestPoint[parentEdgeId] = lowestPoint[edgeId];
                } else if (lowestPoint[edgeId] > lowestPoint[parentEdgeId]) {
                    secondLowestPoint[parentEdgeId] =
                        std::min(secondLowestPoint[parentEdgeId], lowestPoint[edgeId]);
                } else {
                    secondLowestPoint[parentEdgeId] =
                        std::min(secondLowestPoint[parentEdgeId], secondLowestPoint[edgeId]);
                }
            }
        }
    } while (!dfsStack.empty());
}


template <class GraphType>
bool LeftRightPlanarityCheck<GraphType>::dfsTesting(NodeType startNode) {
    std::stack<NodeType> dfsStack;
    dfsStack.push(startNode);

    // Per-node neighbor iterators in DFS graph
    using NeighborIterator = decltype(dfsGraph.neighborRange(static_cast<NodeType>(0)).begin());
    std::vector<NeighborIterator> neighborIterators(dfsGraph.upperNodeIdBound());
    std::vector<bool> neighborInitialized(dfsGraph.upperNodeIdBound(), false);
    std::vector<edgeid> preprocessedEdges(numberOfEdges, noneEdgeId);

    auto processNeighborEdges = [&](NodeType currentNode, bool &callRemoveBackEdges) -> bool {
        auto range = dfsGraph.neighborRange(currentNode);

        auto &neighborIterator = neighborIterators[currentNode];
        while (neighborIterator != range.end()) {
            const NodeType neighbor = *neighborIterator;
            const edgeid currentEdgeId = graph->edgeId(currentNode, neighbor);
            assert(currentEdgeId != noneEdgeId);
            if (preprocessedEdges[currentEdgeId] == noneEdgeId) {
                stackBottom[currentEdgeId] = stack.empty() ? NoneConflictPair : stack.top();

                if (currentEdgeId == parentEdgeIds[neighbor]) {
                    // Tree edge: go deeper
                    dfsStack.push(currentNode);
                    dfsStack.push(neighbor);
                    preprocessedEdges[currentEdgeId] = currentEdgeId;
                    callRemoveBackEdges = false;
                    return true; // more work later
                }

                lowestPointEdge[currentEdgeId] = currentEdgeId;
                stack.emplace(Interval{}, Interval(currentEdgeId, currentEdgeId));
            }

            auto currentEdgeIterator = lowestPoint[currentEdgeId];
            if (currentEdgeIterator != none && currentEdgeIterator < heights[currentNode]) {

                if (neighbor == *dfsGraph.neighborRange(currentNode).begin()) {
                    // First neighbor: propagate lowestPointEdge to parent edge
                    lowestPointEdge[parentEdgeIds[currentNode]] = lowestPointEdge[currentEdgeId];
                } else if (!applyConstraints(currentEdgeId, parentEdgeIds[currentNode])) {
                    return false;
                }
            }

            ++neighborIterator;
        }

        return true;
    };

    // Main DFS loop
    do {
        const NodeType currentNode = dfsStack.top();
        dfsStack.pop();

        const edgeid parentEid = parentEdgeIds[currentNode];
        bool callRemoveBackEdges{true};

        if (!neighborInitialized[currentNode]) {
            neighborIterators[currentNode] = dfsGraph.neighborRange(currentNode).begin();
            neighborInitialized[currentNode] = true;
        }

        if (!processNeighborEdges(currentNode, callRemoveBackEdges)) {
            return false;
        }

        if (callRemoveBackEdges && parentEid != noneEdgeId) {
            removeBackEdges(parentEid, parentNodes[currentNode]);
        }

    } while (!dfsStack.empty());

    return true;
}



} // namespace NetworKit

#endif // NETWORKIT_PLANARITY_LEFT_RIGHT_PLANARITY_CHECK_HPP_
