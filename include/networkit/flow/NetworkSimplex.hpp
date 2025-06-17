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
                   const std::string& capacityAttributeName  = "capacity");



    void run() override{}
    private:
    const Graph *graph;
    std::string demand;
    std::string capacity;
    std::string flow{"flow"};
};
}
#endif //NETWORKSIMPLEX_H
