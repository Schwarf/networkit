/*  SSSPRunner.hpp
 *
 *  Created on: 17.12.2025
 *  Authors: Andreas Scharf (andreas.b.scharf@gmail.com)
 *
 */

#ifndef NETWORKIT_GPU_SSSP_RUNNER_HPP_
#define NETWORKIT_GPU_SSSP_RUNNER_HPP_

#include <networkit/gpu/DeviceGraphCSR.hpp>

#include <vector>

namespace NetworKit::GPU {

template <typename WeightT>
class SsspRunner {
public:
    using DG = DeviceGraphCSR<WeightT>;
    using node_t = typename DG::node_t;
    using index_t = typename DG::index_t;

    explicit SsspRunner(const DG &deviceGraph);
    ~SsspRunner();

    SsspRunner(const SsspRunner &) = delete;
    SsspRunner &operator=(const SsspRunner &) = delete;
    SsspRunner(SsspRunner &&) noexcept;
    SsspRunner &operator=(SsspRunner &&) noexcept;

    std::vector<WeightT> run(node_t source);

private:
    DeviceCSRView<WeightT, index_t, node_t> view{};

    // Reused buffers
    WeightT *deviceDistances{};
    node_t *frontierPing{};
    node_t *frontierPong{};
    std::uint32_t *deviceFrontierCount{};

    node_t numberOfNodes{};
};

} // namespace NetworKit::GPU
#endif // NETWORKIT_GPU_SSSP_RUNNER_HPP_
