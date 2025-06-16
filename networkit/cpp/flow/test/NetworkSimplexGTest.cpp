/*  NetworkSimplexGTest.cpp
*
 *	Created on: 16.06.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */
#include <gtest/gtest.h>
#include <networkit/graph/Graph.hpp>

namespace NetworKit {
TEST(GraphGTest, AttributeTest) {
    Graph G(4, /*weighted=*/false, /*directed=*/false, /*edgesIndexed=*/true);
    // 5. Assign capacity values to each edge by edge endpoints (u,v).
    // The operator()(u, v) returns a proxy that allows setting the attribute for edge (u,v).
    G.addEdge(0, 1);
    G.addEdge(1, 2);
    G.addEdge(2, 3);
    G.addEdge(3, 0);
    auto capacityAttr = G.attachEdgeDoubleAttribute("capacity");
    capacityAttr(0, 1) = 10.5;
    capacityAttr(1, 2) = 8.0;
    capacityAttr(2, 3) = 12.0;
    capacityAttr(3, 0) = 7.5;

    // 6. Query and print the capacity of a specific edge, e.g. edge (0,1)
    double cap01 = capacityAttr.get2(0, 1);  // get2(u,v) fetches the value by endpoints
    std::cout << "Capacity of edge (0,1) is " << cap01 << std::endl;

    // Alternatively, you could retrieve by edge ID:
    index eId = G.edgeId(0, 1);           // get the internal ID of edge (0,1)
    double cap_by_id = capacityAttr[eId];
    auto x = G.getEdgeDoubleAttribute("capacity");
    try {
        auto y = G.getEdgeDoubleAttribute("bert");
    }
    catch (const std::runtime_error &rt) {
        std::cout << "CATCHED" << std::endl;
    }
    int v =1 ;
    EXPECT_EQ(1, v);
    // auto y = G.getEdgeDoubleAttribute("bert");
}
}
