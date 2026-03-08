#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "phantom_memory_pool.cuh"

#define PHANTOM_CHECK_CUDA(val) check((val), #val, __FILE__, __LINE__)
#define PHANTOM_CHECK_CUDA_LAST() checkLast(__FILE__, __LINE__)

template<typename T>
inline void check(T err, const char *const func, const char *const file,
                  const int line) {
    if (err != cudaSuccess) {
        std::cerr << std::endl
                  << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func
                  << std::endl;
        throw std::runtime_error("CUDA Runtime Error");
    }
}

inline void checkLast(const char *const file, const int line) {
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA Runtime Error");
    }
}

namespace phantom::util {

    inline cudaStream_t normalize_stream(cudaStream_t stream) {
        return (stream == nullptr) ? cudaStreamPerThread : stream;
    }

    template<class T>
    inline cudaError_t malloc_device(T **ptr, size_t n, const cudaStream_t &stream) {
        const cudaStream_t effective_stream = normalize_stream(stream);
        return pool_cudaMallocAsync(reinterpret_cast<void **>(ptr), n * sizeof(T), effective_stream);
    }

    template<class T>
    inline cudaError_t free_device(T *ptr, const cudaStream_t &stream) {
        const cudaStream_t effective_stream = normalize_stream(stream);
        return pool_cudaFreeAsync(ptr, effective_stream);
    }

    class cuda_stream_wrapper {
    public:
        cuda_stream_wrapper() {
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        }

        ~cuda_stream_wrapper() {
            cudaStreamDestroy(stream_);
        }

        [[nodiscard]] auto &get_stream() const {
            return stream_;
        }

    private:
        cudaStream_t stream_{};
    };

    template<class T>
    class cuda_auto_ptr {

    private:
        T *ptr_ = nullptr;
        size_t n_ = 0;
        cudaStream_t cudaStream_ = cudaStreamPerThread;

    public:
        cuda_auto_ptr() = default;

        explicit cuda_auto_ptr(T *ptr, size_t n, const cudaStream_t &stream) {
            ptr_ = ptr;
            n_ = n;
            cudaStream_ = normalize_stream(stream);
        }

        // copy constructor
        cuda_auto_ptr(const cuda_auto_ptr &obj) {
            const cudaStream_t src_stream = normalize_stream(obj.cudaStream_);
            PHANTOM_CHECK_CUDA(malloc_device(&this->ptr_, obj.n_, src_stream));
            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(this->ptr_, obj.ptr_, obj.n_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                               src_stream));
            this->n_ = obj.n_;
            this->cudaStream_ = src_stream;
        }

        // copy assignment
        cuda_auto_ptr &operator=(const cuda_auto_ptr &obj) {
            if (this == &obj) {
                return *this;
            }

            reset();

            const cudaStream_t src_stream = normalize_stream(obj.cudaStream_);
            PHANTOM_CHECK_CUDA(malloc_device(&this->ptr_, obj.n_, src_stream));
            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(this->ptr_, obj.ptr_, obj.n_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                               src_stream));
            this->n_ = obj.n_;
            this->cudaStream_ = src_stream;
            return *this;
        }

        // move constructor
        cuda_auto_ptr(cuda_auto_ptr &&dyingObj) noexcept {
            // share the underlying pointer
            this->ptr_ = dyingObj.ptr_;
            this->n_ = dyingObj.n_;
            this->cudaStream_ = normalize_stream(dyingObj.cudaStream_);

            // reset the dying object
            dyingObj.ptr_ = nullptr;
            dyingObj.n_ = 0;
            dyingObj.cudaStream_ = nullptr;
        }

        // move assignment
        cuda_auto_ptr &operator=(cuda_auto_ptr &&dyingObj) noexcept {
            if (this == &dyingObj) {
                return *this;
            }

            reset();

            this->ptr_ = dyingObj.ptr_;
            this->n_ = dyingObj.n_;
            this->cudaStream_ = normalize_stream(dyingObj.cudaStream_);

            // reset the dying object
            dyingObj.ptr_ = nullptr;
            dyingObj.n_ = 0;
            dyingObj.cudaStream_ = nullptr;

            return *this;
        }

        ~cuda_auto_ptr() // destructor
        {
            reset();
        }

        T *get() const {
            return this->ptr_;
        }

        T *operator->() const {
            return this->ptr_;
        }

        T &operator*() const {
            return this->ptr_;
        }

        [[nodiscard]] auto &get_n() const {
            return this->n_;
        }

        [[nodiscard]] auto &get_stream() const {
            return this->cudaStream_;
        }

        void reset() {
            if (ptr_ == nullptr) {
                return;
            }
            T *ptr = ptr_;
            const size_t n = n_;
            cudaStream_t stream = cudaStream_;

            // Make reset idempotent: avoid double-free if reset() is called again (e.g., destructor after manual reset).
            ptr_ = nullptr;
            n_ = 0;
            cudaStream_ = cudaStreamPerThread;

            stream = normalize_stream(stream);
            auto err = free_device(ptr, stream);
            if (err != cudaSuccess) {
                std::cerr << "Error freeing " << n << " * " << sizeof(T) << " bytes at " << ptr
                          << " on stream " << stream << std::endl;
                std::cerr << "Error code: " << cudaGetErrorString(err) << std::endl;
            }
        }
    };

    template<class T>
    cuda_auto_ptr<T> make_cuda_auto_ptr(size_t n, const cudaStream_t &stream) {
        T *ptr;
        PHANTOM_CHECK_CUDA(malloc_device(&ptr, n, stream));
        return cuda_auto_ptr<T>(ptr, n, stream);
    }

    class CUDATimer {
    public:
        explicit CUDATimer(std::string func_name)
                : func_name_(std::move(func_name)) {
            stream_ = cudaStreamPerThread;
            cudaEventCreate(&start_event_);
            cudaEventCreate(&stop_event_);
        }

        ~CUDATimer() {
            cudaEventDestroy(start_event_);
            cudaEventDestroy(stop_event_);
            auto n_trials = time_.size();
            auto mean_time = mean(time_);
            auto min_time = min(time_);
            auto median_time = median(time_);
            auto stddev = std_dev(time_);
            std::cout << func_name_ << ","
                      << n_trials << ","
                      << median_time << ","
                      << mean_time << std::endl;
        }

        inline void start() const {
            cudaEventRecord(start_event_, stream_);
        }

        inline void stop() {
            cudaEventRecord(stop_event_, stream_);
            cudaEventSynchronize(stop_event_);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start_event_, stop_event_);
            time_.push_back(milliseconds);
        }

    private:
        std::string func_name_;
        cudaStream_t stream_;
        cudaEvent_t start_event_{}, stop_event_{};
        std::vector<float> time_;

        static float mean(std::vector<float> const &v) {
            if (v.empty())
                return 0;

            auto const count = static_cast<float>(v.size());
            return std::reduce(v.begin(), v.end()) / count * 1000;
        }

        static float median(std::vector<float> v) {
            size_t size = v.size();

            if (size == 0)
                return 0;

            sort(v.begin(), v.end());
            return v[size / 2] * 1000;
        }

        static float min(std::vector<float> v) {
            size_t size = v.size();

            if (size == 0)
                return 0;

            sort(v.begin(), v.end());
            return v.front() * 1000;
        }

        static float max(std::vector<float> v) {
            size_t size = v.size();

            if (size == 0)
                return 0;

            sort(v.begin(), v.end());
            return v.back() * 1000;
        }

        static double std_dev(std::vector<float> const &v) {
            if (v.empty())
                return 0;

            auto const count = static_cast<float>(v.size());
            float mean = std::reduce(v.begin(), v.end()) / count;

            std::vector<double> diff(v.size());

            std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
            double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            return std::sqrt(sq_sum / count);
        }
    };

}
