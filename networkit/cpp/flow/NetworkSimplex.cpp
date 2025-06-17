/*  NetworkSimplex.cpp
*
 *	Created on: 16.06.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#include <networkit/flow/NetworkSimplex.hpp>
namespace NetworKit{
    NetworkSimplex::NetworkSimplex(const Graph & G, const std::string& demandAttributeName, const std::string& capacityAttributeName) :
    graph(&G), demand(demandAttributeName), capacity(capacityAttributeName){
        if (!graph->isWeighted()) {
            throw std::runtime_error("NetworkSimplex algorithm requires weighted graphs");
        }

        if (!graph->isDirected()) {
            throw std::runtime_error("Only directed graphs are supported");
        }
    }
}