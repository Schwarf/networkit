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
        if (G.hasNode(u))
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
        if (G.hasNode(u))
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

template <typename GraphT>
std::pair<typename GraphT::NodeT, typename GraphT::NodeT> randomEdge(const GraphT &G,
                                                                     bool uniformDistribution) {
    using NodeT = typename GraphT::NodeT;
    if (!G.numberOfEdges()) {
        throw std::runtime_error("Error: the graph has no edges!");
    }

    if (uniformDistribution) {
        /*
         * The simple idea here is to interpret all neighborhoods next to each other, resulting
         * in a virtual vector of size m. Then we draw a random index and return the edge.
         * For undirected edges, the vector has size 2m; but the idea remains. There is one minor
         * complication for undirected edges with self-loops: each edge {u,v} with u != v is stored
         * twice (once in the neighborhood of u, once in v) but a loop (u, u) is only stored once.
         * To equalize the probabilities we reject edges {u,v} with u > v and try again. This leads
         * to less than two expected trails in and is only done for undirected graphs with
         * self-loops.
         */

        do {
            const auto upper =
                G.isDirected() ? G.numberOfEdges() : 2 * G.numberOfEdges() - G.numberOfSelfLoops();
            auto idx = Aux::Random::index(upper);

            NodeT u, v;

            if (idx > upper / 2) {
                // assuming degrees are somewhat distributed uniformly, it's better to start with
                // larger nodes for large indices. In this case we have to mirror the index:
                idx = (upper - 1) - idx;

                for (u = static_cast<NodeT>(G.upperNodeIdBound() - 1);
                     !G.hasNode(u) || idx >= G.degree(u); --u) {
                    if (G.hasNode(u))
                        idx -= G.degree(u);
                }

                v = G.getIthNeighbor(u, G.degree(u) - 1 - idx);

            } else {
                for (u = 0; !G.hasNode(u) || idx >= G.degree(u); ++u) {
                    assert(u < G.upperNodeIdBound());
                    if (G.hasNode(u))
                        idx -= G.degree(u);
                }

                v = G.getIthNeighbor(u, idx);
            }

            if (G.numberOfSelfLoops() && !G.isDirected() && u > v)
                // reject (see above)
                continue;

            return {u, v};
        } while (true);
    }

    NodeT u; // we will return edge (u, v)

    // fast way, but not a uniform random edge!
    do {
        u = GraphTools::randomNode(G);
    } while (!G.degree(u));

    const NodeT v = G.getIthNeighbor(u, Aux::Random::integer(G.degree(u) - 1));

    return {u, v};
}

template<typename GraphT>
std::vector<std::pair<typename GraphT::NodeT, typename GraphT::NodeT>> randomEdges(const GraphT &G, count nr) {
    using NodeT = typename GraphT::NodeT;
    if (!nr)
        return {};

    if (!G.numberOfEdges()) {
        throw std::runtime_error(
            "Graph has no edges to sample from. Add edges to the graph first.");
    }

    std::vector<std::pair<NodeT, NodeT>> edges;

    auto &gen = Aux::Random::getURNG();
    std::vector<count> outDeg(G.upperNodeIdBound());
    G.forNodes([&outDeg, &G](const NodeT u) { outDeg[u] = G.degree(u); });

    std::discrete_distribution<count> distribution(outDeg.begin(), outDeg.end());

    for (index i = 0; i < nr; i++) {
        NodeT u, v; // we will pick edge (u, v)
        if (G.isDirected()) {
            u = static_cast<NodeT>(distribution(gen));
            // should always be the case as  without
            // edges should have probability 0
            assert(G.degree(u));
            v = randomNeighbor(G, u);
        } else {
            // self-loops which appear only once in the outEdge arrays
            // easiest way it to ignore edges (u, v) with u > v
            do {
                u = distribution(gen);
                // should always be the case as  without
                // edges should have probability 0
                assert(G.degree(u));
                v = randomNeighbor(G, u);
            } while (u > v);
        }
        edges.emplace_back(u, v);
    }

    return edges;
}


} // namespace NetworKit::GraphTools
#endif // NETWORKIT_GRAPHTOOLSIMPL_HPP
