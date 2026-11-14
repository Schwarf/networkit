networkit_add_test(graph DeviceGraphCSRBuilderGTest graph)
networkit_add_test(graph DeviceBufferGTest graph)
networkit_add_test(graph DeviceGraphCSRGTest graph)

if(NETWORKIT_WITH_CUDA)
    networkit_add_test(graph CudaDeviceBufferGTest graph)
    networkit_add_test(graph CudaBackendGTest graph)
endif()

if(NETWORKIT_WITH_METAL)
    networkit_add_test(graph MetalDeviceBufferGTest graph)
endif()
