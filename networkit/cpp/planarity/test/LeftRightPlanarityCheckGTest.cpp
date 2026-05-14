/*
 * GraphGTest.cpp
 *
 *  Created on: 05.01.2025
 *      Author: Andreas Scharf (andreas.b.scharf@gmail.com)
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <networkit/io/METISGraphReader.hpp>
#include <networkit/planarity/LeftRightPlanarityCheck.hpp>

namespace NetworKit {

using GraphTypes = ::testing::Types<DynamicGraph<uint64_t, double>, DynamicGraph<int32_t, float>>;

template <class GraphType>
class LeftRightPlanarityCheckTestFixture : public testing::Test {
    using NodeType = GraphType::node_type;
    using EdgeType = GraphType::edge_weight_type;

public:
    static constexpr int maxNumberOfNodes{10};

    GraphType pathGraph(count numNodes) {
        GraphType graph;
        graph.addNodes(numNodes);
        for (count i = 0; i < numNodes - 1; ++i) {
            graph.addEdge(i, i + 1);
        }
        graph.indexEdges();
        return graph;
    }

    GraphType cycleGraph(count numNodes) {
        GraphType graph;
        graph.addNodes(numNodes);
        for (count i = 0; i < numNodes - 1; ++i) {
            graph.addEdge(i, i + 1);
        }
        if (numNodes > 2)
            graph.addEdge(numNodes - 2, 0);
        graph.indexEdges();
        return graph;
    }

    GraphType starGraph(count numNodes) {
        GraphType graph;
        graph.addNodes(numNodes);
        for (count i = 0; i < numNodes - 1; ++i) {
            graph.addEdge(i, i + 1);
        }
        if (numNodes > 2)
            graph.addEdge(numNodes - 2, 0);
        graph.indexEdges();
        return graph;
    }

    GraphType binaryTreeGraph(count numNodes) {
        GraphType graph(numNodes, true, false);
        for (count i = 0; i < numNodes; ++i) {
            count leftChild = 2 * i + 1;
            count rightChild = 2 * i + 2;
            if (leftChild < numNodes) {
                graph.addEdge(i, leftChild, static_cast<double>(i));
            }
            if (rightChild < numNodes) {
                graph.addEdge(i, rightChild, static_cast<double>(i));
            }
        }
        graph.indexEdges();
        return graph;
    }

    GraphType wheelGraph(count numNodes) {
        GraphType graph(numNodes, false, false);
        if (numNodes < 4) {
            throw std::invalid_argument("A wheel graph requires at least 4 nodes.");
        }
        // Form cycle
        for (count i = 1; i < numNodes - 1; ++i) {
            graph.addEdge(i, i + 1);
        }
        graph.addEdge(numNodes - 1, 1); // Close the cycle

        // Connect center to cycle
        for (count i = 1; i < numNodes; ++i) {
            graph.addEdge(0, i);
        }
        graph.indexEdges();
        return graph;
    }

    GraphType completeGraph(count numNodes) {
        GraphType graph(numNodes, true);

        for (count i = 0; i < numNodes; ++i) {
            for (count j = i + 1; j < numNodes; ++j) {
                graph.addEdge(i, j, static_cast<double>(j * (j + 1)));
            }
        }
        graph.indexEdges();
        return graph;
    }

    GraphType gridGraph(count rows, count columns) {
        GraphType graph(rows * columns);
        for (count row = 0; row < rows; ++row) {
            for (count col = 0; col < columns; ++col) {
                count currentNode = row * columns + col;

                // Connect to the right neighbor
                if (col + 1 < columns) {
                    graph.addEdge(currentNode, currentNode + 1);
                }

                // Connect to the bottom neighbor
                if (row + 1 < rows) {
                    graph.addEdge(currentNode, currentNode + columns);
                }
            }
        }
        graph.indexEdges();
        return graph;
    }

    GraphType petersenGraph(count n, count k) {
        GraphType graph(2 * n);

        for (count i = 0; i < n; ++i) {
            graph.addEdge(i, (i + 1) % n);
        }

        for (count i = 0; i < n; ++i) {
            graph.addEdge(n + i, n + (i + k) % n);
        }

        for (count i = 0; i < n; ++i) {
            graph.addEdge(i, n + i);
        }
        graph.indexEdges();
        return graph;
    }
};

TYPED_TEST_SUITE(LeftRightPlanarityCheckTestFixture, GraphTypes, );

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testNoEdgesIndexedGraphThrows) {
    TypeParam graph(0);
    try {
        LeftRightPlanarityCheck test(graph);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error &e) {
        EXPECT_STREQ(e.what(), "The graph has no edge IDs.");
    } catch (...) {
        FAIL() << "Expected std::runtime_error but got a different exception.";
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testDirectedGraphThrows) {
    TypeParam graph(0, false, true, false);
    try {
        LeftRightPlanarityCheck test(graph);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error &e) {
        EXPECT_STREQ(e.what(), "The graph is not an undirected graph.");
    } catch (...) {
        FAIL() << "Expected std::runtime_error but got a different exception.";
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testIsPlanarThrowsIfRunIsNotCalled) {
    TypeParam graph(0, false, false, true);
    LeftRightPlanarityCheck test(graph);
    try {
        test.isPlanar();
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error &e) {
        EXPECT_STREQ(e.what(), "Error, run must be called first");
    } catch (...) {
        FAIL() << "Expected std::runtime_error but got a different exception.";
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testPlanarEmptyGraph) {
    TypeParam graph(0, false, false, true);
    LeftRightPlanarityCheck test(graph);
    test.run();
    EXPECT_TRUE(test.isPlanar());
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testPlanarSingleNode) {
    TypeParam graph{1, false, false, true};
    LeftRightPlanarityCheck test(graph);
    test.run();
    EXPECT_TRUE(test.isPlanar());
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testPlanarPathGraphs) {
    for (count numberOfNodes = 2; numberOfNodes <= this->maxNumberOfNodes; ++numberOfNodes) {
        TypeParam graph = this->pathGraph(numberOfNodes);
        LeftRightPlanarityCheck test(graph);
        test.run();
        EXPECT_TRUE(test.isPlanar());
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testPlanarCycleGraphs) {
    for (count numberOfNodes = 2; numberOfNodes <= this->maxNumberOfNodes; ++numberOfNodes) {
        TypeParam graph = this->cycleGraph(numberOfNodes);
        LeftRightPlanarityCheck test(graph);
        test.run();
        EXPECT_TRUE(test.isPlanar());
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testPlanarStarGraphs) {
    for (count numberOfNodes = 2; numberOfNodes <= this->maxNumberOfNodes; ++numberOfNodes) {
        TypeParam graph = this->starGraph(numberOfNodes);
        LeftRightPlanarityCheck test(graph);
        test.run();
        EXPECT_TRUE(test.isPlanar());
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testPlanarTreeGraphs) {
    for (count numberOfNodes = 2; numberOfNodes <= this->maxNumberOfNodes; ++numberOfNodes) {
        TypeParam graph = this->binaryTreeGraph(numberOfNodes);
        LeftRightPlanarityCheck test(graph);
        test.run();
        EXPECT_TRUE(test.isPlanar());
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testPlanarWheelGraphs) {
    for (count numberOfNodes = 4; numberOfNodes <= this->maxNumberOfNodes; ++numberOfNodes) {
        TypeParam graph = this->wheelGraph(numberOfNodes);
        LeftRightPlanarityCheck test(graph);
        test.run();
        EXPECT_TRUE(test.isPlanar());
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testPlanarCompleteGraphs) {
    constexpr count maxNumberPlanar{5};
    for (count numberOfNodes = 2; numberOfNodes < maxNumberPlanar; ++numberOfNodes) {
        TypeParam graph = this->completeGraph(numberOfNodes);
        LeftRightPlanarityCheck test(graph);
        test.run();
        EXPECT_TRUE(test.isPlanar());
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testNonPlanarCompleteGraphsEulerCriterium) {
    for (count numberOfNodes = 5; numberOfNodes <= this->maxNumberOfNodes; ++numberOfNodes) {
        TypeParam graph = this->completeGraph(numberOfNodes);
        LeftRightPlanarityCheck test(graph);
        test.run();
        EXPECT_FALSE(test.isPlanar());
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testPlanarGridGraphs) {
    for (count numberOfRows = 2; numberOfRows < this->maxNumberOfNodes / 2; ++numberOfRows) {
        for (count numberOfColumns = 2; numberOfColumns < this->maxNumberOfNodes / 2; ++numberOfColumns) {
            TypeParam graph = this->gridGraph(numberOfRows, numberOfColumns);
            LeftRightPlanarityCheck test(graph);
            test.run();
            EXPECT_TRUE(test.isPlanar());
        }
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testNonPlanarCompleteBipartiteGraphK3_3) {
    TypeParam graph(6);
    graph.addEdge(0, 3);
    graph.addEdge(0, 4);
    graph.addEdge(0, 5);
    graph.addEdge(1, 3);
    graph.addEdge(1, 4);
    graph.addEdge(1, 5);
    graph.addEdge(2, 3);
    graph.addEdge(2, 4);
    graph.addEdge(2, 5);
    graph.indexEdges();

    LeftRightPlanarityCheck test(graph);
    test.run();
    EXPECT_FALSE(test.isPlanar());
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testNonPlanarCompleteTripartiteGraphK3_3_3) {
    TypeParam graph(9);
    graph.addEdge(0, 3);
    graph.addEdge(0, 4);
    graph.addEdge(0, 5);
    graph.addEdge(1, 3);
    graph.addEdge(1, 4);
    graph.addEdge(1, 5);
    graph.addEdge(2, 3);
    graph.addEdge(2, 4);
    graph.addEdge(2, 5);

    graph.addEdge(0, 6);
    graph.addEdge(0, 7);
    graph.addEdge(0, 8);
    graph.addEdge(1, 6);
    graph.addEdge(1, 7);
    graph.addEdge(1, 8);
    graph.addEdge(2, 6);
    graph.addEdge(2, 7);
    graph.addEdge(2, 8);

    graph.addEdge(3, 6);
    graph.addEdge(3, 7);
    graph.addEdge(3, 8);
    graph.addEdge(4, 6);
    graph.addEdge(4, 7);
    graph.addEdge(4, 8);
    graph.addEdge(5, 6);
    graph.addEdge(5, 7);
    graph.addEdge(5, 8);
    graph.indexEdges();
    LeftRightPlanarityCheck test(graph);
    test.run();
    EXPECT_FALSE(test.isPlanar());
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testOnePlanarOneNonPlanarSubGraph) {
    Graph graph(10);
    // complete bipartite graph K3,3 (non-planar)
    graph.addEdge(0, 3);
    graph.addEdge(0, 4);
    graph.addEdge(0, 5);
    graph.addEdge(1, 3);
    graph.addEdge(1, 4);
    graph.addEdge(1, 5);
    graph.addEdge(2, 3);
    graph.addEdge(2, 4);
    graph.addEdge(2, 5);
    // Simple cycle graph (planar)
    graph.addEdge(6, 7);
    graph.addEdge(7, 8);
    graph.addEdge(8, 9);
    graph.addEdge(9, 6);

    graph.indexEdges();
    LeftRightPlanarityCheck test(graph);
    test.run();
    EXPECT_FALSE(test.isPlanar());
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testPlanarPetersenGraphs) {
    for (count n = 3; n < this->maxNumberOfNodes; ++n) {
        for (count k = 1; k <= std::floor(n / 2); ++k) {
            const bool isPlanarPetersenGraph = k == 1 || (k == 2 && !(n & 1));
            if (isPlanarPetersenGraph) {
                TypeParam graph = this->petersenGraph(n, k);
                LeftRightPlanarityCheck test(graph);
                test.run();
                EXPECT_TRUE(test.isPlanar());
            }
        }
    }
}

TYPED_TEST(LeftRightPlanarityCheckTestFixture, testNonPlanarPetersenGraphs) {
    for (count n = 3; n < this->maxNumberOfNodes; ++n) {
        for (count k = 1; k <= std::floor(n / 2); ++k) {
            const bool isNonPlanarPetersenGraph = !(k == 1 || (k == 2 && !(n & 1)));
            if (isNonPlanarPetersenGraph) {
                TypeParam graph = this->petersenGraph(n, k);
                LeftRightPlanarityCheck test(graph);
                test.run();
                EXPECT_FALSE(test.isPlanar());
            }
        }
    }
}

TEST(LeftRightPlanarityCheckGTest, testPlanar4eltGraph) {
    METISGraphReader reader;
    auto graph = reader.read("input/4elt.graph");
    graph.indexEdges();
    LeftRightPlanarityCheck test(graph);
    test.run();
    EXPECT_TRUE(test.isPlanar());
}

TEST(LeftRightPlanarityCheckGTest, testNonPlanarHepthGraph) {
    METISGraphReader reader;
    auto graph = reader.read("input/hep-th.graph");
    graph.indexEdges();
    LeftRightPlanarityCheck test(graph);
    test.run();
    EXPECT_FALSE(test.isPlanar());
}

TEST(LeftRightPlanarityCheckGTest, testPlanarAirfoil1Graph) {
    METISGraphReader reader;
    auto graph = reader.read("input/airfoil1.graph");
    graph.indexEdges();
    LeftRightPlanarityCheck test(graph);
    test.run();
    EXPECT_TRUE(test.isPlanar());
}

TEST(LeftRightPlanarityCheckGTest, testNonPlanarAstroPhGraph) {
    METISGraphReader reader;
    auto graph = reader.read("input/astro-ph.graph");
    graph.indexEdges();
    LeftRightPlanarityCheck test(graph);
    test.run();
    EXPECT_FALSE(test.isPlanar());
}

} // namespace NetworKit
