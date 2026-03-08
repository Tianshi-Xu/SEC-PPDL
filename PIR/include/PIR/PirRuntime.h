#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <phantom/context.cuh>

namespace secppdl::pir
{

void print_line(int line_number);
void print_parameters(const PhantomContext &context);
void gpu_sync_or_throw(const std::string &tag);
void select_cuda_device_or_throw(int gpu_id);

template <typename Fn>
double time_phase(Fn &&work, bool gpu_phase, const std::string &phase_name)
{
    if (gpu_phase)
    {
        gpu_sync_or_throw(phase_name + "/pre");
    }

    const auto start = std::chrono::high_resolution_clock::now();
    work();

    if (gpu_phase)
    {
        gpu_sync_or_throw(phase_name + "/post");
    }

    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace secppdl::pir
