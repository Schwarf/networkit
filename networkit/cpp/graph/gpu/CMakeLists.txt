if(NETWORKIT_BUILD_CORE)
    if(NETWORKIT_MONOLITH)
        set(NETWORKIT_GRAPH_GPU_TARGET networkit)
    else()
        set(NETWORKIT_GRAPH_GPU_TARGET networkit_graph)
    endif()

    if(NETWORKIT_WITH_CUDA)
        # Public because CudaBackend.hpp contains calls into the CUDA runtime.
        target_link_libraries(${NETWORKIT_GRAPH_GPU_TARGET} PUBLIC CUDA::cudart)
    endif()

    if(NETWORKIT_WITH_METAL)
        target_sources(${NETWORKIT_GRAPH_GPU_TARGET} PRIVATE MetalBackend.cpp)
        target_include_directories(${NETWORKIT_GRAPH_GPU_TARGET} SYSTEM PUBLIC
            "${NETWORKIT_METAL_CPP_PATH}")
        target_link_libraries(${NETWORKIT_GRAPH_GPU_TARGET} PUBLIC
            "-framework Foundation" "-framework Metal")
    endif()
endif()

add_subdirectory(test)
