/*
 * FloydWarshallBenchmark.cpp
 *
 *  Created on: 14.05.2026
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#include <benchmark/benchmark.h>

#include <cstdint>

#include <networkit/distance/FloydWarshall.hpp>
#include <networkit/graph/Graph.hpp>

namespace NetworKit {

template <class NodeType>
using WeightedGraph = DynamicGraph<NodeType, double>;

template <class GraphType>
static GraphType completeWeightedGraph(count n) {
    using NodeType = typename GraphType::node_type;

    GraphType graph(n, true, false);

    for (count u = 0; u < n; ++u) {
        for (count v = u + 1; v < n; ++v) {
            graph.addEdge(static_cast<NodeType>(u), static_cast<NodeType>(v),
                          static_cast<double>((u + v) % 17 + 1));
        }
    }

    return graph;
}

template <class GraphType>
static GraphType pathWeightedGraph(count n) {
    using NodeType = typename GraphType::node_type;

    GraphType graph(n, true, false);

    for (count u = 1; u < n; ++u) {
        graph.addEdge(static_cast<NodeType>(u - 1), static_cast<NodeType>(u),
                      static_cast<double>(u % 17 + 1));
    }

    return graph;
}

template <class GraphType>
static void BM_FloydWarshallCompleteGraph(benchmark::State &state) {
    const GraphType graph = completeWeightedGraph<GraphType>(static_cast<count>(state.range(0)));
    const auto target = static_cast<typename GraphType::node_type>(graph.numberOfNodes() - 1);

    for (auto _ : state) {
        FloydWarshall<GraphType> floydWarshall(graph);
        floydWarshall.run();
        benchmark::DoNotOptimize(floydWarshall.getDistance(0, target));
    }

}

template <class GraphType>
static void BM_FloydWarshallPathGraph(benchmark::State &state) {
    const GraphType graph = pathWeightedGraph<GraphType>(static_cast<count>(state.range(0)));
    const auto target = static_cast<typename GraphType::node_type>(graph.numberOfNodes() - 1);

    for (auto _ : state) {
        FloydWarshall<GraphType> floydWarshall(graph);
        floydWarshall.run();
        benchmark::DoNotOptimize(floydWarshall.getDistance(0, target));
    }

}

BENCHMARK(BM_FloydWarshallCompleteGraph<WeightedGraph<uint64_t>>)
    ->Name("BM_FloydWarshallCompleteGraph<uint64_t,double>")
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);
BENCHMARK(BM_FloydWarshallCompleteGraph<WeightedGraph<uint32_t>>)
    ->Name("BM_FloydWarshallCompleteGraph<uint32_t,double>")
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);
BENCHMARK(BM_FloydWarshallCompleteGraph<WeightedGraph<int32_t>>)
    ->Name("BM_FloydWarshallCompleteGraph<int32_t,double>")
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);

BENCHMARK(BM_FloydWarshallPathGraph<WeightedGraph<uint64_t>>)
    ->Name("BM_FloydWarshallPathGraph<uint64_t,double>")
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);
BENCHMARK(BM_FloydWarshallPathGraph<WeightedGraph<uint32_t>>)
    ->Name("BM_FloydWarshallPathGraph<uint32_t,double>")
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);
BENCHMARK(BM_FloydWarshallPathGraph<WeightedGraph<int32_t>>)
    ->Name("BM_FloydWarshallPathGraph<int32_t,double>")
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);

} // namespace NetworKit
