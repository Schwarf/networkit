//
// Created by andreas on 07.05.25.
//
#include <gtest/gtest.h>
#include <networkit/planarity/PlanarEmbedding.hpp>

namespace NetworKit {
TEST(PlanarEmbeddingTest, FirstHalfEdgeInsertion) {
    PlanarEmbedding pe(3);
    EXPECT_EQ(pe.getGraph().numberOfEdges(), 0u);

    pe.addHalfEdge(0, 1, /*ccw=*/false, /*ref=*/0);

    auto nbrs0 = pe.getClockWiseOrderedNeighbors(0);
    EXPECT_EQ(nbrs0.size(), 1u);
    EXPECT_EQ(nbrs0[0], 1);
    // graph has exactly one directed edge
    EXPECT_TRUE(pe.getGraph().hasEdge(0,1));
    EXPECT_EQ(pe.getGraph().numberOfEdges(), 1u);
}
}