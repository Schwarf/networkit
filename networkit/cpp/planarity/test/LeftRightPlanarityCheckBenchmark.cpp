/*
 * LeftRightPlanarityCheckBenchmark.cpp
 *
 *  Created on: 15.05.2026
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#include <benchmark/benchmark.h>

#include <cstdint>

#include <networkit/io/METISGraphReader.hpp>
#include <networkit/planarity/LeftRightPlanarityCheck.hpp>

namespace NetworKit {

template <class NodeType, class EdgeWeightType>
using BenchmarkGraph = DynamicGraph<NodeType, EdgeWeightType>;

template <class GraphType>
static void BM_LeftRightPlanarityCheckHepThGraph(benchmark::State &state) {
    using NodeType = typename GraphType::node_type;
    using EdgeWeightType = typename GraphType::edge_weight_type;

    METISGraphReader<NodeType, EdgeWeightType> reader;
    auto graph = reader.read("input/4elt.graph");
    graph.indexEdges();

    for (auto _ : state) {
        LeftRightPlanarityCheck test(graph);
        test.run();
        benchmark::DoNotOptimize(test.isPlanar());
    }
}

BENCHMARK(BM_LeftRightPlanarityCheckHepThGraph<BenchmarkGraph<uint64_t, double>>)
    ->Name("BM_LeftRightPlanarityCheckHepThGraph<uint64_t,double>");

BENCHMARK(BM_LeftRightPlanarityCheckHepThGraph<BenchmarkGraph<uint64_t, float>>)
    ->Name("BM_LeftRightPlanarityCheckHepThGraph<uint64_t,float>");

BENCHMARK(BM_LeftRightPlanarityCheckHepThGraph<BenchmarkGraph<uint32_t, double>>)
    ->Name("BM_LeftRightPlanarityCheckHepThGraph<uint32_t,double>");

BENCHMARK(BM_LeftRightPlanarityCheckHepThGraph<BenchmarkGraph<uint32_t, float>>)
    ->Name("BM_LeftRightPlanarityCheckHepThGraph<uint32_t,float>");

} // namespace NetworKit
