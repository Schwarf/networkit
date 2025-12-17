//
// Created by andreas on 17.12.25.
//
#include <iostream>
#include <limits>

#include <networkit/graph/Graph.hpp>
#include <networkit/gpu/DeviceGraph.hpp>
#include <networkit/gpu/SSSP.hpp>

using namespace NetworKit;

int main() {
    // Directed, weighted graph with 4 nodes
    Graph G(4, /*weighted=*/true, /*directed=*/true);

    // Edges:
    // 0->1 (1), 0->2 (4), 1->2 (1), 1->3 (2), 2->3 (1)
    G.addEdge(0, 1, 1.0);
    G.addEdge(0, 2, 4.0);
    G.addEdge(1, 2, 0.1);
    G.addEdge(1, 3, 2.0);
    G.addEdge(2, 3, 1.0);

    // Build device-friendly CSR (weights exist in G so no default needed)
    auto dg = GPU::buildDeviceGraph<float>(G, /*requireContinuousNodeIds=*/true);

    // Run SSSP from source 0 on the GPU
    auto distAll = NetworKit::GPU::ssspWorklistCuda(dg, 0);

    std::cout << "Distances from 0:\n";
    for (node v = 0; v < G.numberOfNodes(); ++v) {
        std::cout << "  0 -> " << v << " = " << distAll[v] << "\n";
    }

    // Expected:
    // 0->0 = 0
    // 0->1 = 1
    // 0->2 = 2  (0->1->2)
    // 0->3 = 3  (0->1->3 or 0->1->2->3)
    return 0;
}