#pragma once

#include <algorithm>
#include <cstddef>
#include <future>
#include <vector>

namespace LinearLayer {

template <typename Work>
void RunConvWorkers(std::size_t count, std::size_t max_threads, const Work &work) {
    const std::size_t threads = std::min(count, std::max(std::size_t(1), max_threads));
    if (threads <= 1) {
        work(0, count);
        return;
    }

    std::vector<std::future<void>> workers;
    workers.reserve(threads - 1);
    for (std::size_t i = 1; i < threads; ++i) {
        workers.push_back(std::async(std::launch::async, work, count * i / threads, count * (i + 1) / threads));
    }
    work(0, count / threads);
    for (auto &worker : workers) {
        worker.get();
    }
}

}
