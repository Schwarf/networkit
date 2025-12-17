//
// Created by andreas on 17.12.25.
//
#include <iostream>

#include <networkit/gpu/HostGraphCSR.hpp>
#include <networkit/gpu/DeviceGraphCSR.hpp>
#include <networkit/gpu/SSSPRunner.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/io/METISGraphReader.hpp>
#include <networkit/distance/FloydWarshall.hpp>
using namespace NetworKit;

using Weight = float;
void runManySources(const NetworKit::Graph& G, std::vector<std::vector<Weight>> & result) {
    // 1) Build CSR on host (one-time)
    auto hostGraph = GPU::buildHostGraphCSR<Weight>(G);
    // NetworKit::GPU::HostGraphCSR<Weight> hostCsr(G);

    // 2) Upload CSR to device (one-time)
    GPU::DeviceGraphCSR<Weight> deviceGraph(hostGraph);

    // 3) Create runner (allocates reusable device buffers once)
    NetworKit::GPU::SsspRunner<Weight> runner(deviceGraph);

    // 4) Run many sources cheaply (only resets dist/frontier each run)
    for (node source = 0; source < 77 && source < G.numberOfNodes(); ++source) {
        auto dist = runner.run(static_cast<NetworKit::GPU::SsspRunner<Weight>::node_t>(source));
        result.push_back(dist);
    }
}
int main() {

    METISGraphReader reader;
    Graph graph = reader.read("/mnt/linux_data/projects/cpp/networkit/repo/networkit/input/lesmis.graph");
    std::vector<std::vector<Weight>> gpuResult;
    runManySources(graph, gpuResult);
    FloydWarshall floydWarshall(graph);
    floydWarshall.run();
    double total{};
    int count = 0;
    for(int i = 0; i < 77; ++i) {
        for(int j = 0; j < 77; ++j)
            if (static_cast<Weight>(floydWarshall.getDistance(i, j)) != gpuResult[i][j]) {
                std::cout << "ERROR" << i << " " << j << " " << ++count << std::endl;
            }
        else
            std::cout<< floydWarshall.getDistance(i, j) << "  " << gpuResult[i][j] << std::endl;
    }
    return 0;
}

// int main() {
//     // Directed, weighted graph with 4 nodes
//     Graph G(4, /*weighted=*/true, /*directed=*/true);
//
//     // Edges:
//     // 0->1 (1), 0->2 (4), 1->2 (1), 1->3 (2), 2->3 (1)
//     G.addEdge(0, 1, 1.0);
//     G.addEdge(0, 2, 4.0);
//     G.addEdge(1, 2, 0.1);
//     G.addEdge(1, 3, 2.0);
//     G.addEdge(2, 3, 1.0);
//
//     // Build device-friendly CSR (weights exist in G so no default needed)
//     auto dg = GPU::buildHostGraphCSR<float>(G, /*requireContinuousNodeIds=*/true);
//
//     // Run SSSP from source 0 on the GPU
//     auto distAll = NetworKit::GPU::ssspWorklistCuda(dg, 0);
//
//     std::cout << "Distances from 0:\n";
//     for (node v = 0; v < G.numberOfNodes(); ++v) {
//         std::cout << "  0 -> " << v << " = " << distAll[v] << "\n";
//     }
//
//     // Expected:
//     // 0->0 = 0
//     // 0->1 = 1
//     // 0->2 = 2  (0->1->2)
//     // 0->3 = 3  (0->1->3 or 0->1->2->3)
//     return 0;
// }