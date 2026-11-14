// networkit-format
// Define metal-cpp's implementation exactly once, shared by graph storage and algorithms.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <networkit/graph/gpu/MetalBackend.hpp>

namespace NetworKit::GPU {

bool MetalBackend::isAvailable() {
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    const auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    return device.get() != nullptr;
}

MetalBackend::Context::Context() {
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device) {
        throw std::runtime_error("Metal storage: no Metal device available");
    }
}

} // namespace NetworKit::GPU
