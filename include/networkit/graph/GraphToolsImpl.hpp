//
// Created by andreas on 14.11.26.
//

#ifndef NETWORKIT_GRAPHTOOLSIMPL_HPP
#define NETWORKIT_GRAPHTOOLSIMPL_HPP

#include <networkit/Globals.hpp>

#include <algorithm>

namespace NetworKit::GraphTools {

template <typename GraphT>
count computeMaxDegree(const GraphT &G, bool inDegree) {
    count result = 0;
#ifndef NETWORKIT_OMP2
#pragma omp parallel for reduction(max : result)
    for (omp_index u = 0; u < static_cast<omp_index>(G.upperNodeIdBound()); ++u) {
        result = std::max(result, inDegree ? G.degreeIn(u) : G.degreeOut(u));
    }
#else
    G.forNodes([&](const GraphT::NodeT u) {
        result = std::max(result, inDegree ? G.degreeIn(u) : G.degreeOut(u));
    });
#endif
    return result;
}

template <typename GraphT>
GraphT::EdgeWeightT computeMaxWeightedDegree(const GraphT &G, bool inDegree) {
    typename GraphT::EdgeWeightT result = 0;
#ifndef NETWORKIT_OMP2
#pragma omp parallel for reduction(max : result)
    for (omp_index u = 0; u < static_cast<omp_index>(G.upperNodeIdBound()); ++u) {
        result = std::max(result, inDegree ? G.weightedDegreeIn(u) : G.weightedDegree(u));
    }
#else
    G.forNodes([&](const GraphT::NodeT u) {
        result = std::max(result, inDegree ? G.weightedDegreeIn(u) : G.weightedDegree(u));
    });
#endif
    return result;
}

template <typename GraphT>
GraphT::NodeT randomNode(const GraphT &G) {
    using NodeT = typename GraphT::NodeT;
    if (!G.numberOfNodes())
        return  NullNodeId<NodeT>;

    auto &gen = Aux::Random::getURNG();
    std::uniform_int_distribution<NodeT> distr{NodeT{0},  static_cast<NodeT>(G.upperNodeIdBound() - 1)};
    NodeT v;

    do {
        // When there are many deleted nodes, we might call Aux::Random::integer
        // many times, and it is very expensive.
        v = distr(gen);
    } while (!G.hasNode(v));

    return v;
}

template <typename GraphT>
std::vector<typename GraphT::NodeT> randomNodes(const GraphT &G, count n) {
    using NodeT = typename GraphT::NodeT;
    assert(n <= G.numberOfNodes());
    std::vector<NodeT> selectedNodes;
    std::vector<bool> alreadySelected(G.upperNodeIdBound(), false);

    if (n == G.numberOfNodes()) {
        selectedNodes.insert(selectedNodes.begin(), G.nodeRange().begin(), G.nodeRange().end());
    } else if (n > G.numberOfNodes() / 2) { // in order to minimize the calls to randomNode
        // we randomize the ones that aren't pivot
        // if the are more to be selected than not-selected
        for (count i = 0; i < G.numberOfNodes() - n; ++i) { // we have to sample distinct nodes
            NodeT v = GraphTools::randomNode(G);
            while (alreadySelected[v]) {
                v = GraphTools::randomNode(G);
            }
            alreadySelected[v] = true;
        }

        for (const auto sample : G.nodeRange()) {
            if (!alreadySelected[sample]) { // recall that we selected the non-pivot nodes
                selectedNodes.push_back(sample);
                if (selectedNodes.size() == n)
                    break;
            }
        }
    } else {
        for (count i = 0; i < n; ++i) { // we have to selected distinct nodes
            NodeT v = GraphTools::randomNode(G);
            while (alreadySelected[v]) {
                v = GraphTools::randomNode(G);
            }
            selectedNodes.push_back(v);
            alreadySelected[v] = true;
        }
    }
    return selectedNodes;
}


} // namespace NetworKit::GraphTools
#endif // NETWORKIT_GRAPHTOOLSIMPL_HPP
