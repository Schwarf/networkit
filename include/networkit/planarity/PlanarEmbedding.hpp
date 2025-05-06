//
// Created by andreas on 06.05.25.
//

#ifndef PLANAREMBEDDING_HPP
#define PLANAREMBEDDING_HPP
#include <networkit/graph/Graph.hpp>
namespace NetworKit {
class PlanarEmbedding
{
    public:
        explicit PlanarEmbedding(count n) {
            graph = Graph(n, false, true, false);
        }
        void addHalfEdge(node source, node target) {
            graph.addEdge(source, target);
        }



    private:

    Graph graph;

};
}
#endif //PLANAREMBEDDING_HPP
