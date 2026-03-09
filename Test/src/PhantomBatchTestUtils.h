#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "ciphertext.h"
#include "context.cuh"
#include "plaintext.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace phantom_batch_test {

inline void check_cuda(cudaError_t err, const char *op) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(err));
    }
}

inline void sync_stream() {
    check_cuda(cudaStreamSynchronize(cudaStreamPerThread), "cudaStreamSynchronize");
}

inline PhantomContext make_test_context() {
    const std::size_t poly_degree = 4096;
    phantom::EncryptionParameters parms(phantom::scheme_type::bfv);
    parms.set_poly_modulus_degree(poly_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_degree, {40, 40, 40, 40}));
    parms.set_plain_modulus(phantom::arith::PlainModulus::Batching(poly_degree, 17));
    return PhantomContext(parms);
}

inline std::size_t data_chain_index(const PhantomContext &context) {
    return context.get_first_index();
}

inline std::vector<uint64_t> random_rns_data(
    const std::vector<phantom::arith::Modulus> &moduli, std::size_t poly_degree,
    std::size_t poly_count, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const std::size_t coeff_modulus_size = moduli.size();
    const std::size_t coeff_count = poly_degree * coeff_modulus_size;
    std::vector<uint64_t> out(poly_count * coeff_count, 0ULL);

    for (std::size_t poly_idx = 0; poly_idx < poly_count; ++poly_idx) {
        const std::size_t poly_base = poly_idx * coeff_count;
        for (std::size_t q = 0; q < coeff_modulus_size; ++q) {
            const uint64_t mod = moduli[q].value();
            std::uniform_int_distribution<uint64_t> dist(0ULL, mod - 1ULL);
            const std::size_t q_base = poly_base + q * poly_degree;
            for (std::size_t i = 0; i < poly_degree; ++i) {
                out[q_base + i] = dist(rng);
            }
        }
    }
    return out;
}

inline std::vector<uint64_t> random_plain_data(
    std::size_t poly_degree, uint64_t plain_modulus, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0ULL, plain_modulus - 1ULL);
    std::vector<uint64_t> out(poly_degree, 0ULL);
    for (std::size_t i = 0; i < poly_degree; ++i) {
        out[i] = dist(rng);
    }
    return out;
}

inline void copy_host_to_device(uint64_t *dst, const std::vector<uint64_t> &src) {
    check_cuda(cudaMemcpyAsync(
                   dst, src.data(), src.size() * sizeof(uint64_t), cudaMemcpyHostToDevice,
                   cudaStreamPerThread),
               "cudaMemcpyAsync H2D");
    sync_stream();
}

inline std::vector<uint64_t> copy_device_to_host(const uint64_t *src, std::size_t count) {
    std::vector<uint64_t> out(count, 0ULL);
    check_cuda(cudaMemcpyAsync(
                   out.data(), src, count * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                   cudaStreamPerThread),
               "cudaMemcpyAsync D2H");
    sync_stream();
    return out;
}

inline PhantomCiphertext make_random_ntt_cipher(
    const PhantomContext &context, std::size_t chain_index, std::size_t ct_size,
    std::uint64_t seed, double scale = 1.0) {
    PhantomCiphertext ct;
    ct.resize(context, chain_index, ct_size, cudaStreamPerThread);
    ct.set_ntt_form(true);
    ct.set_scale(scale);
    ct.set_correction_factor(1ULL);
    ct.SetNoiseScaleDeg(1);

    const auto &parms = context.get_context_data(chain_index).parms();
    auto host = random_rns_data(parms.coeff_modulus(), parms.poly_modulus_degree(), ct_size, seed);
    copy_host_to_device(ct.data(), host);
    return ct;
}

inline PhantomPlaintext make_random_ntt_plain(
    const PhantomContext &context, std::size_t chain_index, std::uint64_t seed,
    double scale = 1.0) {
    PhantomPlaintext pt;
    const auto &parms = context.get_context_data(chain_index).parms();
    const std::size_t coeff_modulus_size = parms.coeff_modulus().size();
    const std::size_t poly_degree = parms.poly_modulus_degree();
    pt.resize(coeff_modulus_size, poly_degree, cudaStreamPerThread);
    pt.set_chain_index(chain_index);
    pt.scale() = scale;

    auto host = random_rns_data(parms.coeff_modulus(), poly_degree, 1, seed);
    copy_host_to_device(pt.data(), host);
    return pt;
}

inline PhantomPlaintext make_random_coeff_plain(
    const PhantomContext &context, std::uint64_t seed) {
    PhantomPlaintext pt;
    const auto &parms = context.key_context_data().parms();
    const std::size_t poly_degree = parms.poly_modulus_degree();
    pt.resize(1, poly_degree, cudaStreamPerThread);
    pt.set_chain_index(0);
    pt.scale() = 1.0;

    auto host = random_plain_data(poly_degree, parms.plain_modulus().value(), seed);
    copy_host_to_device(pt.data(), host);
    return pt;
}

inline void expect_device_buffer_eq(
    const uint64_t *lhs, const uint64_t *rhs, std::size_t count) {
    const auto lhs_h = copy_device_to_host(lhs, count);
    const auto rhs_h = copy_device_to_host(rhs, count);
    EXPECT_EQ(lhs_h, rhs_h);
}

} // namespace phantom_batch_test
