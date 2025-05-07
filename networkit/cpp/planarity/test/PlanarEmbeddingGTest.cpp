//
// Created by andreas on 07.05.25.
//
#include <gtest/gtest.h>
#include <networkit/planarity/PlanarEmbedding.hpp>

namespace NetworKit {

TEST(PlanarEmbeddingTest, FirstHalfEdgeInsertion) {
    PlanarEmbedding embedding(3);
    EXPECT_EQ(embedding.getGraph().numberOfEdges(), 0u);

    embedding.addHalfEdge(0, 1, /*ccw=*/false, /*ref=*/0);

    auto neighbors0 = embedding.getClockWiseOrderedNeighborOf(0);
    EXPECT_EQ(neighbors0.size(), 1u);
    EXPECT_EQ(neighbors0[0], 1);
    // graph has exactly one directed edge
    EXPECT_TRUE(embedding.getGraph().hasEdge(0, 1));
    EXPECT_EQ(embedding.getGraph().numberOfEdges(), 1u);
}

TEST(PlanarEmbeddingTest, ClockwiseAndCounterClockwiseInsertion) {
    PlanarEmbedding embedding(4);
    // build three half-edges out of 0 in a star with mixed insertions
    embedding.addHalfEdge(0, 1, /*ccw=*/false, /*ref=*/0);
    // CW insert 2 after 1: [1,2]
    embedding.addHalfEdge(0, 2, /*ccw=*/false, /*ref=*/1);
    // CCW insert 3 before 1: [3,1,2]
    embedding.addHalfEdge(0, 3, /*ccw=*/true, /*ref=*/1);

    auto neighbors0 = embedding.getClockWiseOrderedNeighborOf(0);
    // Expected cycle: 3 → 1 → 2
    std::vector<node> expectedNeighbors0 = {3, 1, 2};
    EXPECT_EQ(neighbors0, expectedNeighbors0);

    // Graph should have 3 directed edges from 0
    EXPECT_EQ(embedding.getGraph().numberOfEdges(), 3u);
    EXPECT_TRUE(embedding.getGraph().hasEdge(0, 3));
}

TEST(PlanarEmbeddingTest, InvalidReferenceThrows) {
    PlanarEmbedding embedding(5);
    embedding.addHalfEdge(0, 1, /*ccw=*/false, /*ref=*/0);
    EXPECT_THROW(embedding.addHalfEdge(0, 2, /*ccw=*/false, /*ref=*/5), std::invalid_argument);
}

TEST(PlanarEmbeddingTest, GetNeighborsOutOfRangeThrows) {
    PlanarEmbedding embedding(2);
    embedding.addHalfEdge(0, 1, false, 0);
    EXPECT_THROW(embedding.getClockWiseOrderedNeighborOf(5), std::runtime_error);
}

TEST(PlanarEmbeddingTest, GetEmbeddingReturnsFullStructure) {
    PlanarEmbedding embedding(3);
    embedding.addHalfEdge(0, 1, false, 0); // [1]
    embedding.addHalfEdge(1, 2, false, 1); // [2]
    embedding.addHalfEdge(2, 0, false, 2); // [0]

    auto neigghborOrder = embedding.getClockwiseNeighborOrder();
    EXPECT_EQ(neigghborOrder.size(), 3u);
    EXPECT_EQ(neigghborOrder[0], std::vector<node>({1}));
    EXPECT_EQ(neigghborOrder[1], std::vector<node>({2}));
    EXPECT_EQ(neigghborOrder[2], std::vector<node>({0}));
}


TEST(PlanarEmbeddingTest, OperatorEqualitySimple) {
    // build two identical embeddings
    PlanarEmbedding embedding1(3), embedding2(3);
    embedding1.addHalfEdge(0,1,false,0);
    embedding1.addHalfEdge(1,2,false,1);
    embedding1.addHalfEdge(2,0,false,2);

    embedding2.addHalfEdge(0,1,false,0);
    embedding2.addHalfEdge(1,2,false,1);
    embedding2.addHalfEdge(2,0,false,2);

    EXPECT_TRUE(embedding1 == embedding2);
    EXPECT_FALSE(embedding1 != embedding2);
}

TEST(PlanarEmbeddingTest, OperatorEqualityCyclicShift) {
    // Build embedding1: 0->[1,2,3]
    PlanarEmbedding embedding1(4), embedding2(4);
    embedding1.addHalfEdge(0,1,false,0);
    embedding1.addHalfEdge(0,2,false,1);
    embedding1.addHalfEdge(0,3,false,2);

    // Build embedding2: 0->[2,3,1] (cyclic shift)
    embedding2.addHalfEdge(0,2,false,0);
    embedding2.addHalfEdge(0,3,false,2);
    embedding2.addHalfEdge(0,1,false,3);

    EXPECT_TRUE(embedding1 == embedding2) << "Cyclic shifts should compare equal";
    EXPECT_FALSE(embedding1 != embedding2) << "Cyclic shifts should compare equal";
}

TEST(PlanarEmbeddingTest, OperatorNotEqualityMirror) {
    // embedding1: 0->[1,2,3]
    PlanarEmbedding embedding1(4), embedding2(4);
    embedding1.addHalfEdge(0,1,false,0);
    embedding1.addHalfEdge(0,2,false,1);
    embedding1.addHalfEdge(0,3,false,2);

    // embedding2: build in reverse order 0->[3,2,1]
    embedding2.addHalfEdge(0,3,false,0);
    embedding2.addHalfEdge(0,2,false,3);
    embedding2.addHalfEdge(0,1,false,2);

    EXPECT_TRUE(embedding1 != embedding2) << "Mirror image should not compare equal";
    EXPECT_FALSE(embedding1 == embedding2) << "Mirror image should not compare equal";
}

TEST(PlanarEmbeddingTest, OperatorInequalityDifferentOrder) {
    PlanarEmbedding embedding1(4), embedding2(4);
    embedding1.addHalfEdge(0,1,false,0);
    embedding1.addHalfEdge(0,2,false,1);
    embedding1.addHalfEdge(0,3,false,2);

    // embedding2 with a different swap: 0->[1,3,2]
    embedding2.addHalfEdge(0,1,false,0);
    embedding2.addHalfEdge(0,3,false,1);
    embedding2.addHalfEdge(0,2,false,3);

    EXPECT_FALSE(embedding1 == embedding2);
    EXPECT_TRUE(embedding1 != embedding2);
}

TEST(PlanarEmbeddingTest, OperatorInequalityDifferentGraph) {
    // embedding1 has edge 0->1; embedding2 is empty
    PlanarEmbedding embedding1(2), embedding2(2);
    embedding1.addHalfEdge(0,1,false,0);

    EXPECT_FALSE(embedding1 == embedding2);
    EXPECT_TRUE(embedding1 != embedding2);
}
} // namespace NetworKit
