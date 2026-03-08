#include "PIR/PirCiphertextOps.h"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace secppdl::pir
{

void copy_ciphertext_device_fast(const PhantomCiphertext &src, PhantomCiphertext &dst)
{
    const bool same_layout = dst.size() == src.size() && dst.chain_index() == src.chain_index() &&
                             dst.coeff_modulus_size() == src.coeff_modulus_size() &&
                             dst.poly_modulus_degree() == src.poly_modulus_degree() && dst.data() != nullptr;
    if (!same_layout)
    {
        dst = src;
        return;
    }

    const std::size_t coeff_count = src.size() * src.coeff_modulus_size() * src.poly_modulus_degree();
    const std::size_t bytes = coeff_count * sizeof(uint64_t);
    const cudaError_t err = cudaMemcpyAsync(dst.data(), src.data(), bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "copy_ciphertext_device_fast cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(err)));
    }

    dst.set_scale(src.scale());
    dst.set_ntt_form(src.is_ntt_form());
    dst.set_correction_factor(src.correction_factor());
    dst.SetNoiseScaleDeg(src.GetNoiseScaleDeg());
}

} // namespace secppdl::pir
