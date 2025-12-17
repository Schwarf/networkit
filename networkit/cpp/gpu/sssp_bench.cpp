/*
 * sssp_bench_fixed.cpp
 *
 * Fixed-configuration benchmark + correctness harness for NetworKit GPU SsspRunner.
 *
 * Configuration (hard-coded):
 *  - undirected, weighted
 *  - fixed seeds
 *  - weights in [1..100]
 *  - Erdos-Renyi G(n,p) with p chosen to target ~m edges (variance expected)
 *  - graph generation forced to 1 OMP thread to be reproducible
 *
 * Build: add as executable next to your existing sssp_demo.cpp target.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <omp.h>
#include <cuda_runtime.h>

#include <networkit/distance/Dijkstra.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/graph/Graph.hpp>

#include <networkit/gpu/HostGraphCSR.hpp>
#include <networkit/gpu/DeviceGraphCSR.hpp>
#include <networkit/gpu/SSSPRunner.hpp>
#include <networkit/auxiliary/Random.hpp>

using NetworKit::Graph;
using NetworKit::node;

namespace {

// ---- Fixed configuration ----
constexpr std::size_t NUMBER_OF_NODES = 10'000;
constexpr std::size_t NUMBER_OF_EDGES = 1'000'000;

constexpr std::uint64_t SEED_GRAPH = 1117171;
constexpr std::uint64_t SEED_WEIGHTS = 298345;
constexpr std::uint64_t SEED_SOURCES = 3890191;

constexpr int W_MIN = 1;
constexpr int W_MAX = 100;

constexpr std::size_t K_CORRECTNESS = 128;  // compare GPU vs CPU on K sources
constexpr std::size_t B_BENCH = 2048;       // benchmark on B sources

constexpr double EPS = 1e-4;

// ---- Helpers ----
double computeProbUndirected(std::size_t n, std::size_t mTarget) {
    // For undirected ER: E[m] = n*(n-1)/2 * p  => p = 2*m / (n*(n-1))
    const double dn = static_cast<double>(n);
    const double denom = dn * (dn - 1.0) * 0.5;
    const double p = static_cast<double>(mTarget) / denom;
    return std::clamp(p, 0.0, 1.0);
}

Graph makeWeightedCopyUndirected(const Graph &G, std::uint64_t seed) {
    Graph Gw(G.numberOfNodes(), /*weighted=*/true, /*directed=*/false);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> distW(W_MIN, W_MAX);

    // Copy structure, assign deterministic weights
    G.forEdges([&](node u, node v) {
        const double w = static_cast<double>(distW(rng));
        Gw.addEdge(u, v, w);
    });

    return Gw;
}

std::vector<node> sampleSources(std::size_t n, std::size_t count, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint64_t> dist(0, n - 1);

    std::vector<node> srcs;
    srcs.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        srcs.push_back(static_cast<node>(dist(rng)));
    return srcs;
}

float timeGpuRunsMillis(NetworKit::GPU::SsspRunner<float> &runner,
                        const std::vector<node> &sources,
                        double &checksumOut) {
    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup (not timed)
    if (!sources.empty()) {
        auto d0 = runner.run(static_cast<NetworKit::GPU::SsspRunner<float>::node_t>(sources[0]));
        double s = 0.0;
        for (float x : d0)
            if (std::isfinite(x))
                s += x;
        checksumOut += s;
    }

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    double checksum = 0.0;
    for (node src : sources) {
        auto dist = runner.run(static_cast<NetworKit::GPU::SsspRunner<float>::node_t>(src));
        double local = 0.0;
        for (float x : dist)
            if (std::isfinite(x))
                local += x;
        checksum += local;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checksumOut += checksum;
    return ms;
}

double timeCpuRunsMillis(const Graph &G, const std::vector<node> &sources, double &checksumOut) {
    using clock = std::chrono::steady_clock;

    // Warmup (not timed)
    if (!sources.empty()) {
        NetworKit::Dijkstra dj(G, sources[0], /*storePaths=*/false);
        dj.run();
        const auto &dist = dj.getDistances();
        double s = 0.0;
        for (double x : dist)
            if (std::isfinite(x))
                s += x;
        checksumOut += s;
    }

    const auto t0 = clock::now();

    double checksum = 0.0;
    for (node src : sources) {
        NetworKit::Dijkstra dj(G, src, /*storePaths=*/false);
        dj.run();
        const auto &dist = dj.getDistances();
        double local = 0.0;
        for (double x : dist)
            if (std::isfinite(x))
                local += x;
        checksum += local;
    }

    const auto t1 = clock::now();
    checksumOut += checksum;

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

int main() {
    try {
        constexpr bool DIRECTED = false;
        constexpr bool SELF_LOOPS = false;

        const double p = computeProbUndirected(NUMBER_OF_NODES, NUMBER_OF_EDGES);

        std::cout << "=== SSSP bench (fixed config) ===\n";
        std::cout << "Graph: Erdos-Renyi G(n,p), undirected, weighted\n";
        std::cout << "n=" << NUMBER_OF_NODES << "  target_m~" << NUMBER_OF_EDGES << "  p=" << std::setprecision(6) << p << "\n";
        std::cout << "weights in [" << W_MIN << ".." << W_MAX << "]\n";
        std::cout << "seeds: graph=" << SEED_GRAPH << " weights=" << SEED_WEIGHTS
                  << " sources=" << SEED_SOURCES << "\n";
        std::cout << "Correctness K=" << K_CORRECTNESS << "  Benchmark B=" << B_BENCH << "\n";

        // --- Make generation reproducible: force 1 OMP thread during ER generation.
        const int oldOmp = omp_get_max_threads();
        omp_set_num_threads(1);

        // Seed NetworKit global RNG used by Aux::Random::getURNG() (thread-local).
        // If this line doesn't compile in your tree, paste the error and I'll adapt it.
        Aux::Random::setSeed(SEED_GRAPH, /*useThreadSeed=*/false);

        NetworKit::ErdosRenyiGenerator gen(NUMBER_OF_NODES, p, DIRECTED, SELF_LOOPS);
        Graph G = gen.generate();

        omp_set_num_threads(oldOmp);

        std::cout << "Generated: N(nodes)=" << G.numberOfNodes() << "  N(edges)=" << G.numberOfEdges() << "\n";

        // --- Assign weights deterministically (always weighted)
        Graph Gw = makeWeightedCopyUndirected(G, SEED_WEIGHTS);

        // --- Build CSR + runner once
        auto hostCsr = NetworKit::GPU::buildHostGraphCSR<float>(Gw);
        NetworKit::GPU::DeviceGraphCSR<float> deviceCsr(hostCsr);
        NetworKit::GPU::SsspRunner<float> runner(deviceCsr);

        // --- Sample sources deterministically
        const auto corrSources = sampleSources(NUMBER_OF_NODES, K_CORRECTNESS, SEED_SOURCES);
        const auto benchSources = sampleSources(NUMBER_OF_NODES, B_BENCH, SEED_SOURCES + 1);

        // --- Correctness (sampled)
        std::cout << "\n[Correctness] GPU vs CPU Dijkstra on K sources...\n";

        const float INF_F = std::numeric_limits<float>::infinity();
        const double INF_D = std::numeric_limits<double>::infinity();

        std::size_t mismatches = 0;
        for (node src : corrSources) {
            auto gpuDist = runner.run(static_cast<NetworKit::GPU::SsspRunner<float>::node_t>(src));

            NetworKit::Dijkstra dj(Gw, src, /*storePaths=*/false);
            dj.run();
            const auto &cpuDist = dj.getDistances();

            for (std::size_t v = 0; v < gpuDist.size(); ++v) {
                const float gd = gpuDist[v];
                const double cd = cpuDist[v];

                const bool gInf = (!std::isfinite(gd)) || (gd == INF_F);
                const bool cInf = (!std::isfinite(cd)) || (cd == INF_D);

                if (gInf != cInf) {
                    ++mismatches;
                    if (mismatches <= 10) {
                        std::cout << "  mismatch src=" << src << " v=" << v
                                  << " gpu=" << gd << " cpu=" << cd << "\n";
                    }
                    break;
                }

                if (!gInf) {
                    const double diff = std::abs(static_cast<double>(gd) - cd);
                    if (diff > EPS) {
                        ++mismatches;
                        if (mismatches <= 10) {
                            std::cout << "  mismatch src=" << src << " v=" << v
                                      << " gpu=" << gd << " cpu=" << cd
                                      << " diff=" << diff << "\n";
                        }
                        break;
                    }
                }
            }
        }

        if (mismatches == 0)
            std::cout << "Correctness: OK (no mismatches in K=" << K_CORRECTNESS << ")\n";
        else
            std::cout << "Correctness: FAIL (" << mismatches
                      << " sources had a mismatch; first 10 printed)\n";

        // --- Benchmark
        std::cout << "\n[Benchmark] throughput on B sources...\n";

        double gpuChecksum = 0.0;
        const float gpuMs = timeGpuRunsMillis(runner, benchSources, gpuChecksum);

        double cpuChecksum = 0.0;
        const double cpuMs = timeCpuRunsMillis(Gw, benchSources, cpuChecksum);

        const double gpuPerSrc = static_cast<double>(gpuMs) / std::max<std::size_t>(1, benchSources.size());
        const double cpuPerSrc = cpuMs / std::max<std::size_t>(1, benchSources.size());

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU total: " << gpuMs << " ms  (" << gpuPerSrc << " ms/src)"
                  << "  checksum=" << std::setprecision(2) << gpuChecksum << "\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "CPU total: " << cpuMs << " ms  (" << cpuPerSrc << " ms/src)"
                  << "  checksum=" << std::setprecision(2) << cpuChecksum << "\n";

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Speedup (CPU/GPU): " << (cpuMs / std::max(1e-9, static_cast<double>(gpuMs)))
                  << "x\n";

        return (mismatches == 0) ? 0 : 2;
    } catch (const std::exception &ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
