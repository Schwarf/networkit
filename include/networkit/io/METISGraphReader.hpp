/*
 * METISGraphReader.hpp
 *
 *  Created on: 17.01.2013
 *      Author: Christian Staudt
 */

#ifndef NETWORKIT_IO_METIS_GRAPH_READER_HPP_
#define NETWORKIT_IO_METIS_GRAPH_READER_HPP_

#include "METISParser.hpp"
#include <networkit/auxiliary/Enforce.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/StringTools.hpp>
#include <networkit/io/GraphReader.hpp>

namespace NetworKit {

/**
 * @ingroup io
 * Reader for the METIS file format documented in [1]
 *
 * [1] http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf
 */
template <class NodeType = node, class EdgeWeightType = edgeweight>
class METISGraphReader final  {
public:
    METISGraphReader() = default;

    /**
     * Takes a file path as parameter and returns a graph file.
     *
     * @param[in]  path  file path
     *
     * @param[out]  the graph read from file
     */
    DynamicGraph<NodeType, EdgeWeightType> read(std::string_view path);
};

template <class NodeType, class EdgeWeightType>
DynamicGraph<NodeType, EdgeWeightType> METISGraphReader<NodeType, EdgeWeightType>::read(std::string_view path) {

    METISParser parser(path);

    std::tuple<count, count, index, count> header = parser.getHeader();
    count n = std::get<0>(header);
    count m = std::get<1>(header);
    index fmt = std::get<2>(header);
    count ncon = std::get<3>(header);

    bool weighted;
    if (fmt % 10 == 1) {
        weighted = true;
        DEBUG("graph has been identified as weighted");
    } else {
        weighted = false;
    }
    count ignoreFirst = 0;
    if (fmt / 10 == 1) {
        DEBUG("first ", ncon, " value(s) will be ignored");
        ignoreFirst = ncon;
    }

    DynamicGraph<NodeType, EdgeWeightType> G(n, weighted);
    std::string graphName =
        Aux::StringTools::split(Aux::StringTools::split(path, '/').back(), '.').front();

    INFO("\n[BEGIN] reading graph G(n=", n, ", m=", m, ") from METIS file: ", graphName);

#ifndef NETWORKIT_RELEASE_LOGGING
    double p = 0.0; // percentage for progress bar
#endif
    NodeType u = 0; // begin with 0
    count edgeCounter = 0;
    count selfLoops = 0;
    if (weighted == 0) {
        while (parser.hasNext() && u < n) {
            auto adjacencies = parser.getNext(ignoreFirst);
            edgeCounter += adjacencies.size();
            for (index i = 0; i < adjacencies.size(); i++) {
                if (adjacencies[i] == 0) {
                    ERROR("METIS Node ID should not be 0, edge ignored.");
                    continue;
                }
                Aux::Checkers::Enforcer::enforce(adjacencies[i] > 0 && adjacencies[i] <= n);
                auto v = static_cast<NodeType>(adjacencies[i] - 1); // METIS-indices are 1-based
                // correct edgeCounter for selfloops
                if (u == v) {
                    edgeCounter++;
                    selfLoops++;
                }
                if (!G.addPartialEdge(unsafe, u, v, defaultEdgeWeight, 0, true))
                    WARN("Not adding edge ", u, "-", v, " since it is already present.");
            }
            u++; // next node
#ifndef NETWORKIT_RELEASE_LOGGING
            if ((u % ((n + 10) / 10)) == 0) {
                p = ((double)(u - 1) / (double)n) * 100;
                DEBUG(p, "% ");
            }
#endif
        }
    } else {
        while (parser.hasNext() && u < n) {
            auto adjacencies = parser.getNextWithWeights(ignoreFirst);
            edgeCounter += adjacencies.size();
            DEBUG("node ", u, " has ", adjacencies.size(), " edges");
            for (index i = 0; i < adjacencies.size(); i++) {
                if (adjacencies[i].first == 0) {
                    ERROR("METIS Node ID should not be 0, edge ignored.");
                    continue;
                }
                Aux::Checkers::Enforcer::enforce(adjacencies[i].first > 0
                                                 && adjacencies[i].first <= n);
                auto v = static_cast<NodeType>(adjacencies[i].first - 1); // METIS-indices are 1-based
                // correct edgeCounter for selfloops
                if (u == v) {
                    edgeCounter++;
                    selfLoops++;
                }
                auto weight = static_cast<EdgeWeightType>(adjacencies[i].second);
                if (!G.addPartialEdge(unsafe, u, v, weight, 0, true))
                    WARN("Not adding edge ", u, "-", v, " since it is already present.");
                else
                    TRACE("(", u, ",", v, ",", adjacencies[i].second, ")");
            }
            u += 1; // next node
#ifndef NETWORKIT_RELEASE_LOGGING
            if ((u % ((n + 10) / 10)) == 0) {
                p = ((double)(u - 1) / (double)n) * 100;
                DEBUG(p, "% ");
            }
#endif
        }
    }
    G.setEdgeCount(unsafe, edgeCounter / 2);
    G.setNumberOfSelfLoops(unsafe, selfLoops);
    if (G.numberOfEdges() != m) {
        ERROR("METIS file ", path,
              " is corrupted: actual number of added edges doesn't match the specifed number of "
              "edges");
    }
    if (edgeCounter != 2 * m) {
        WARN("METIS file is corrupted: not every edge is listed twice");
    }

    INFO("\n[DONE]\n");
    return G;
}

} /* namespace NetworKit */
#endif // NETWORKIT_IO_METIS_GRAPH_READER_HPP_
