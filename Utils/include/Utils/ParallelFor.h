#pragma once
#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace Utils {
// Each worker owns a stable index for thread-local evaluators and buffers.
// Callers must keep network operations outside the parallel region.
template <typename Function>
void ParallelFor(size_t count, size_t threads, Function function) {
    if (!count) return;
    threads = std::min(count, std::max(size_t(1), threads));
    if (threads == 1) {
        for (size_t i = 0; i < count; ++i) function(i, size_t(0));
        return;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    std::vector<std::thread> workers;
    const auto run = [&](size_t worker) {
        try {
            while (!failed.load()) {
                const size_t i = next.fetch_add(1);
                if (i >= count) break;
                function(i, worker);
            }
        } catch (...) {
            failed.store(true);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
        }
    };
    try {
        for (size_t worker = 0; worker < threads; ++worker)
            workers.emplace_back(run, worker);
    } catch (...) {
        failed.store(true);
        for (auto &worker : workers) worker.join();
        throw;
    }
    for (auto &worker : workers) worker.join();
    if (error) std::rethrow_exception(error);
}
} // namespace Utils
