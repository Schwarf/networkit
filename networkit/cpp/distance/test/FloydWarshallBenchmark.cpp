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


template <class GraphType>
static GraphType completeWeightedGraph(count n) {
    using NodeType = typename GraphType::node_type;
    using EdgeWeightType = typename GraphType::edge_weight_type;

    GraphType graph(n, true, false);

    for (count u = 0; u < n; ++u) {
        for (count v = u + 1; v < n; ++v) {
            graph.addEdge(static_cast<NodeType>(u), static_cast<NodeType>(v),
                          static_cast<EdgeWeightType>((u + v) % 17 + 1));
        }
    }

    return graph;
}

template <class GraphType>
static GraphType pathWeightedGraph(count n) {
    using NodeType = typename GraphType::node_type;
    using EdgeWeightType = typename GraphType::edge_weight_type;

    GraphType graph(n, true, false);

    for (count u = 1; u < n; ++u) {
        graph.addEdge(static_cast<NodeType>(u - 1), static_cast<NodeType>(u),
                      static_cast<EdgeWeightType>(u % 17 + 1));
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
BENCHMARK(BM_FloydWarshallCompleteGraph<DynamicGraph<uint64_t, double>>)
    ->Name("BM_FloydWarshallCompleteGraph<uint64_t,double>")
    ->RangeMultiplier(2)
    ->Range(16, 512);

BENCHMARK(BM_FloydWarshallCompleteGraph<DynamicGraph<uint64_t, float>>)
    ->Name("BM_FloydWarshallCompleteGraph<uint64_t,float>")
    ->RangeMultiplier(2)
    ->Range(16, 512);

BENCHMARK(BM_FloydWarshallCompleteGraph<DynamicGraph<uint32_t, double>>)
    ->Name("BM_FloydWarshallCompleteGraph<uint32_t,double>")
    ->RangeMultiplier(2)
    ->Range(16, 512);

BENCHMARK(BM_FloydWarshallCompleteGraph<DynamicGraph<uint32_t, float>>)
    ->Name("BM_FloydWarshallCompleteGraph<uint32_t,float>")
    ->RangeMultiplier(2)
    ->Range(16, 512);

} // namespace NetworKit
