#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

#include <cuda_runtime_api.h>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace phantom::util {

    struct rmm_allocation_record {
        int device = 0;
        size_t bytes = 0;
    };

    class rmm_device_allocator {
    public:
        static rmm_device_allocator &instance() {
            static rmm_device_allocator allocator;
            return allocator;
        }

        void *allocate(size_t bytes, cudaStream_t stream) {
            if (bytes == 0) {
                return nullptr;
            }

            auto [device_id, resource] = resource_for_current_device();
            const rmm::cuda_stream_view rmm_stream = to_rmm_stream(stream);
            void *ptr = resource->allocate(bytes, rmm_stream);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                allocations_.emplace(ptr, rmm_allocation_record{device_id, bytes});
            }
            return ptr;
        }

        void deallocate(void *ptr, cudaStream_t stream) {
            if (ptr == nullptr) {
                return;
            }

            rmm_allocation_record record{};
            {
                std::lock_guard<std::mutex> lock(mutex_);
                const auto it = allocations_.find(ptr);
                if (it == allocations_.end()) {
                    throw rmm::logic_error("attempting to free an untracked device pointer");
                }
                record = it->second;
                allocations_.erase(it);
            }

            const rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{record.device});
            auto *resource = resource_for_device(record.device);
            const rmm::cuda_stream_view rmm_stream = to_rmm_stream(stream);
            resource->deallocate(ptr, record.bytes, rmm_stream);
        }

    private:
        rmm_device_allocator() = default;

        static rmm::cuda_stream_view to_rmm_stream(cudaStream_t stream) {
            if (stream == nullptr) {
                return rmm::cuda_stream_default;
            }
            if (stream == cudaStreamLegacy) {
                return rmm::cuda_stream_legacy;
            }
            if (stream == cudaStreamPerThread) {
                return rmm::cuda_stream_per_thread;
            }
            return rmm::cuda_stream_view{stream};
        }

        std::pair<int, rmm::mr::device_memory_resource *> resource_for_current_device() {
            int device_id = 0;
            const cudaError_t status = cudaGetDevice(&device_id);
            if (status != cudaSuccess) {
                throw rmm::cuda_error(cudaGetErrorString(status));
            }

            return {device_id, resource_for_device(device_id)};
        }

        rmm::mr::device_memory_resource *resource_for_device(int device_id) {
            std::lock_guard<std::mutex> lock(mutex_);
            return resource_for_device_unlocked(device_id);
        }

        rmm::mr::device_memory_resource *resource_for_device_unlocked(int device_id) {
            auto it = resources_.find(device_id);
            if (it != resources_.end()) {
                return it->second;
            }

            const rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{device_id});
            // Intentionally leak per-device CUDA memory pools.
            // RMM's cuda_async_memory_resource destroys the underlying pool in
            // its destructor; during process teardown this can run after CUDART
            // unload and trip an assert on cudaErrorCudartUnloading.
            auto *resource = new rmm::mr::cuda_async_memory_resource(
                std::nullopt,
                std::optional<std::size_t>{std::numeric_limits<std::size_t>::max()});
            resources_.emplace(device_id, resource);
            return resource;
        }

        std::mutex mutex_{};
        std::unordered_map<int, rmm::mr::cuda_async_memory_resource *> resources_{};
        std::unordered_map<void *, rmm_allocation_record> allocations_{};
    };

    inline cudaError_t wrap_rmm_exception(const std::exception &e) {
        (void)cudaGetLastError();
        (void)e;
        if (dynamic_cast<const rmm::out_of_memory *>(&e) != nullptr ||
            dynamic_cast<const rmm::bad_alloc *>(&e) != nullptr) {
            return cudaErrorMemoryAllocation;
        }
        if (dynamic_cast<const rmm::logic_error *>(&e) != nullptr) {
            return cudaErrorInvalidDevicePointer;
        }
        return cudaErrorUnknown;
    }

}

inline cudaError_t pool_cudaMalloc(void **devPtr, size_t size) {
    try {
        *devPtr = phantom::util::rmm_device_allocator::instance().allocate(size, cudaStreamPerThread);
        return cudaSuccess;
    } catch (const std::exception &e) {
        *devPtr = nullptr;
        return phantom::util::wrap_rmm_exception(e);
    }
}

inline cudaError_t pool_cudaFree(void *devPtr) {
    try {
        phantom::util::rmm_device_allocator::instance().deallocate(devPtr, cudaStreamPerThread);
        return cudaSuccess;
    } catch (const std::exception &e) {
        return phantom::util::wrap_rmm_exception(e);
    }
}

inline cudaError_t pool_cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
    try {
        *devPtr = phantom::util::rmm_device_allocator::instance().allocate(size, stream);
        return cudaSuccess;
    } catch (const std::exception &e) {
        *devPtr = nullptr;
        return phantom::util::wrap_rmm_exception(e);
    }
}

inline cudaError_t pool_cudaFreeAsync(void *devPtr, cudaStream_t stream) {
    try {
        phantom::util::rmm_device_allocator::instance().deallocate(devPtr, stream);
        return cudaSuccess;
    } catch (const std::exception &e) {
        return phantom::util::wrap_rmm_exception(e);
    }
}

// Compatibility shim: keeps existing call sites that eagerly initialize
// the device allocator while using the RMM-backed implementation.
inline phantom::util::rmm_device_allocator &phantom_device_allocator() {
    return phantom::util::rmm_device_allocator::instance();
}
