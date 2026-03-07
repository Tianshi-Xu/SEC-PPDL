#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#if defined(__has_include)
#if __has_include(<cub/util_allocator.cuh>)
#include <cub/util_allocator.cuh>
#elif __has_include(<cccl/cub/util_allocator.cuh>)
#include <cccl/cub/util_allocator.cuh>
#else
#error "CUB CachingDeviceAllocator header not found."
#endif
#else
#include <cub/util_allocator.cuh>
#endif

inline cub::CachingDeviceAllocator &phantom_device_allocator() {
    static cub::CachingDeviceAllocator allocator(true);
    return allocator;
}

inline cudaError_t pool_cudaMalloc(void **devPtr, size_t size) {
    return phantom_device_allocator().DeviceAllocate(devPtr, size);
}

inline cudaError_t pool_cudaFree(void *devPtr) {
    if (devPtr == nullptr) {
        return cudaSuccess;
    }
    return phantom_device_allocator().DeviceFree(devPtr);
}

inline cudaError_t pool_cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
    return phantom_device_allocator().DeviceAllocate(devPtr, size, stream);
}

inline cudaError_t pool_cudaFreeAsync(void *devPtr, cudaStream_t stream) {
    (void) stream;
    if (devPtr == nullptr) {
        return cudaSuccess;
    }
    return phantom_device_allocator().DeviceFree(devPtr);
}
