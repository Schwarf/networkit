/*  NetworkSimplex.hpp
*
 *	Created on: 16.06.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#ifndef NETWORKIT_FLOW_NETWORKSIMPLEX_HPP_
#define NETWORKIT_FLOW_NETWORKSIMPLEX_HPP_
#include <networkit/graph/Graph.hpp>
#include <networkit/base/Algorithm.hpp>
#include <string>
#include <vector>
#include <utility>

namespace NetworKit {
class NetworkSimplex : public Algorithm {
public:
    NetworkSimplex(const Graph &G,
                   const std::string& demandAttributeName   = "demand",
                   const std::string& capacityAttributName  = "capacity",
                   const std::string& weightAttributeName   = "weight");




    private:
    const Graph *graph;
};
}
#endif //NETWORKSIMPLEX_H
