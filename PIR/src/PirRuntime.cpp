#include "PIR/PirRuntime.h"

#include <phantom/context.cuh>

namespace secppdl::pir
{

void print_line(int line_number)
{
    std::cout << "Line " << std::setw(3) << line_number << " --> ";
}

void print_parameters(const PhantomContext &context)
{
    const auto &context_data = context.key_context_data();
    const auto &parms = context_data.parms();

    std::string scheme_name;
    switch (parms.scheme())
    {
    case phantom::scheme_type::bfv:
        scheme_name = "BFV";
        break;
    case phantom::scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    case phantom::scheme_type::bgv:
        scheme_name = "BGV";
        break;
    default:
        throw std::invalid_argument("Unsupported scheme");
    }

    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters:" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: " << parms.poly_modulus_degree() << std::endl;

    const auto &coeff_modulus = parms.coeff_modulus();
    std::size_t total_bits = 0;
    for (const auto &mod : coeff_modulus)
    {
        total_bits += static_cast<std::size_t>(mod.bit_count());
    }

    std::cout << "|   coeff_modulus size: " << total_bits << " (";
    for (std::size_t i = 0; i + 1 < coeff_modulus.size(); ++i)
    {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count() << ") bits" << std::endl;

    if (parms.scheme() == phantom::scheme_type::bfv)
    {
        std::cout << "|   plain_modulus: " << parms.plain_modulus().value() << std::endl;
    }
    std::cout << "\\" << std::endl;
}

void gpu_sync_or_throw(const std::string &tag)
{
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cudaDeviceSynchronize failed at " + tag + ": " + std::string(cudaGetErrorString(err)));
    }
}

void select_cuda_device_or_throw(int gpu_id)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaGetDeviceCount failed: " + std::string(cudaGetErrorString(err)));
    }
    if (device_count <= 0)
    {
        throw std::runtime_error("No CUDA device found.");
    }
    if (gpu_id < 0 || gpu_id >= device_count)
    {
        throw std::invalid_argument("gpu_id out of range, available device count: " + std::to_string(device_count));
    }

    err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaSetDevice failed: " + std::string(cudaGetErrorString(err)));
    }

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, gpu_id);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaGetDeviceProperties failed: " + std::string(cudaGetErrorString(err)));
    }

    std::cout << "INFO: Using GPU[" << gpu_id << "]: " << prop.name << std::endl;
}

} // namespace secppdl::pir
