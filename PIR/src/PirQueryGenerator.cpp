#include "PIR/PirQueryGenerator.h"

#include "PIR/PirMath.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <phantom/evaluate.cuh>
#include <phantom/polymath.cuh>

namespace secppdl::pir
{

namespace
{

void multiply_power_of_x_device(
    const PhantomCiphertext &encrypted, PhantomCiphertext &destination, uint32_t index, const PhantomContext &context)
{
    destination = encrypted;

    const auto coeff_count = static_cast<uint32_t>(encrypted.poly_modulus_degree());
    const auto coeff_mod_count = static_cast<uint32_t>(encrypted.coeff_modulus_size());
    const auto poly_count = static_cast<uint32_t>(encrypted.size());
    if (coeff_count == 0 || coeff_mod_count == 0 || poly_count == 0)
    {
        return;
    }

    const auto full_mod_count = static_cast<uint32_t>(context.gpu_rns_tables().size());
    if (coeff_mod_count > full_mod_count)
    {
        throw std::runtime_error("multiply_power_of_x_device: coeff_mod_count exceeds context modulus size.");
    }

    const uint64_t *moduli_device = reinterpret_cast<const uint64_t *>(context.gpu_rns_tables().modulus());
    if (!moduli_device)
    {
        throw std::runtime_error("multiply_power_of_x_device: null device modulus pointer.");
    }

    launch_negacyclic_shift_kernel(
        encrypted.data(), destination.data(), coeff_count, coeff_mod_count, poly_count, index, moduli_device,
        cudaStreamPerThread);

    const auto launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error(
            "multiply_power_of_x_device launch failed: " + std::string(cudaGetErrorString(launch_err)));
    }
}

inline void apply_galois_device(
    const PhantomContext &context, const PhantomCiphertext &src, uint32_t galois_elt, const PhantomGaloisKey &galois_keys,
    PhantomCiphertext &dst)
{
    dst = src;
    phantom::apply_galois_inplace(context, dst, static_cast<size_t>(galois_elt), galois_keys);
}

} // namespace

std::vector<uint32_t> required_galois_elts_for_pir(uint32_t poly_degree, uint32_t m, uint32_t chunks)
{
    std::vector<uint32_t> elts;
    const uint32_t logm = ceil_log2_u64(m);
    const uint32_t two_n = poly_degree << 1;

    for (uint32_t i = 0; i < logm; ++i)
    {
        elts.push_back((poly_degree + (1U << i)) >> i);
    }

    for (uint32_t step = 1; step < chunks; ++step)
    {
        elts.push_back(mod_pow_u32(3U, step, two_n));
    }

    std::sort(elts.begin(), elts.end());
    elts.erase(std::unique(elts.begin(), elts.end()), elts.end());
    return elts;
}

std::vector<PhantomCiphertext> expand_query_sealpir_device(
    const PhantomCiphertext &encrypted, uint32_t m, const PhantomContext &context, const PhantomGaloisKey &galois_keys)
{
    if (m == 0)
    {
        throw std::invalid_argument("expand_query_sealpir_device: m must be positive.");
    }

    const uint32_t n = static_cast<uint32_t>(context.key_context_data().parms().poly_modulus_degree());
    const uint32_t logn = ceil_log2_u64(n);
    const uint32_t logm = ceil_log2_u64(m);
    if (logm > logn)
    {
        throw std::logic_error("expand_query_sealpir_device: m > poly_modulus_degree is not allowed.");
    }

    if (m == 1)
    {
        return std::vector<PhantomCiphertext>{ encrypted };
    }

    std::vector<uint32_t> galois_elts(logn);
    for (uint32_t i = 0; i < logn; ++i)
    {
        galois_elts[i] = (n + (1U << i)) >> i;
    }

    std::vector<uint64_t> two_coeff(static_cast<size_t>(n), 0ULL);
    two_coeff[0] = 2ULL;
    PhantomPlaintext two;
    two.load(two_coeff.data(), context, 0, 1.0);

    std::vector<PhantomCiphertext> temp{ encrypted };
    PhantomCiphertext temp_rotated;
    PhantomCiphertext temp_shifted;
    PhantomCiphertext temp_rotated_shifted;

    for (uint32_t i = 0; i + 1 < logm; ++i)
    {
        std::vector<PhantomCiphertext> newtemp(temp.size() << 1);
        const uint32_t index_raw = (n << 1) - (1U << i);
        const uint32_t index = (index_raw * galois_elts[i]) % (n << 1);

        for (size_t a = 0; a < temp.size(); ++a)
        {
            apply_galois_device(context, temp[a], galois_elts[i], galois_keys, temp_rotated);
            newtemp[a] = temp[a];
            phantom::add_inplace(context, newtemp[a], temp_rotated);

            multiply_power_of_x_device(temp[a], temp_shifted, index_raw, context);
            multiply_power_of_x_device(temp_rotated, temp_rotated_shifted, index, context);
            newtemp[a + temp.size()] = temp_shifted;
            phantom::add_inplace(context, newtemp[a + temp.size()], temp_rotated_shifted);
        }

        temp.swap(newtemp);
    }

    std::vector<PhantomCiphertext> newtemp(temp.size() << 1);
    const uint32_t index_raw = (n << 1) - (1U << (logm - 1));
    const uint32_t index = (index_raw * galois_elts[logm - 1]) % (n << 1);

    for (size_t a = 0; a < temp.size(); ++a)
    {
        if (a >= (m - (1U << (logm - 1))))
        {
            newtemp[a] = temp[a];
            phantom::multiply_plain_inplace(context, newtemp[a], two);
        }
        else
        {
            apply_galois_device(context, temp[a], galois_elts[logm - 1], galois_keys, temp_rotated);
            newtemp[a] = temp[a];
            phantom::add_inplace(context, newtemp[a], temp_rotated);

            multiply_power_of_x_device(temp[a], temp_shifted, index_raw, context);
            multiply_power_of_x_device(temp_rotated, temp_rotated_shifted, index, context);
            newtemp[a + temp.size()] = temp_shifted;
            phantom::add_inplace(context, newtemp[a + temp.size()], temp_rotated_shifted);
        }
    }

    std::vector<PhantomCiphertext> expanded;
    expanded.reserve(m);
    for (uint32_t i = 0; i < m; ++i)
    {
        expanded.emplace_back(std::move(newtemp[i]));
    }
    return expanded;
}

PirQueryBundle generate_query_bundle(
    const PirShape &shape, uint64_t plain_mod, const PhantomContext &context, PhantomPublicKey &public_key,
    const PhantomBatchEncoder &batch_encoder)
{
    PirQueryBundle bundle;

    std::vector<uint64_t> query_x_coeff(shape.poly_modulus_degree, 0ULL);
    const uint64_t blocks_mod = shape.blocks_per_row % plain_mod;
    if (blocks_mod == 0)
    {
        throw std::invalid_argument("blocks_per_row has no inverse modulo plain_modulus.");
    }
    const uint64_t inv_scale = mod_inverse_u64(blocks_mod, plain_mod);
    query_x_coeff[shape.c_block] = inv_scale;

    PhantomPlaintext query_x_plain;
    query_x_plain.load(query_x_coeff.data(), context, 0, 1.0);

    std::vector<uint64_t> vec_ones(shape.slot_count, 1ULL);
    std::vector<uint64_t> vec_zeros(shape.slot_count, 0ULL);
    PhantomPlaintext query_y_one;
    PhantomPlaintext query_y_zero;
    batch_encoder.encode(context, vec_ones, query_y_one);
    batch_encoder.encode(context, vec_zeros, query_y_zero);

    public_key.encrypt_asymmetric(context, query_x_plain, bundle.compressed_x);

    bundle.row_selectors.assign(shape.h, PhantomCiphertext{});
    for (std::size_t r = 0; r < shape.h; ++r)
    {
        if (r == shape.r_idx)
        {
            public_key.encrypt_asymmetric(context, query_y_one, bundle.row_selectors[r]);
        }
        else
        {
            public_key.encrypt_asymmetric(context, query_y_zero, bundle.row_selectors[r]);
        }
    }

    return bundle;
}

BatchQueryBundle generate_batch_query_bundle(
    const PirShape &shape, const std::vector<QueryIndex> &query_indices, uint64_t plain_mod, const PhantomContext &context,
    PhantomPublicKey &public_key, const PhantomBatchEncoder &batch_encoder)
{
    BatchQueryBundle bundle;

    const std::size_t batch_size = query_indices.size();
    bundle.batch_compressed_x.assign(batch_size, PhantomCiphertext{});
    bundle.batch_row_selectors.assign(batch_size, std::vector<PhantomCiphertext>(shape.h));

    std::vector<uint64_t> vec_ones(shape.slot_count, 1ULL);
    std::vector<uint64_t> vec_zeros(shape.slot_count, 0ULL);
    PhantomPlaintext query_y_one;
    PhantomPlaintext query_y_zero;
    batch_encoder.encode(context, vec_ones, query_y_one);
    batch_encoder.encode(context, vec_zeros, query_y_zero);

    const uint64_t blocks_mod = shape.blocks_per_row % plain_mod;
    if (blocks_mod == 0)
    {
        throw std::invalid_argument("blocks_per_row has no inverse modulo plain_modulus.");
    }
    const uint64_t inv_scale = mod_inverse_u64(blocks_mod, plain_mod);

    for (std::size_t b = 0; b < batch_size; ++b)
    {
        std::vector<uint64_t> query_x_coeff(shape.poly_modulus_degree, 0ULL);
        query_x_coeff[query_indices[b].c_block] = inv_scale;

        PhantomPlaintext query_x_plain;
        query_x_plain.load(query_x_coeff.data(), context, 0, 1.0);

        public_key.encrypt_asymmetric(context, query_x_plain, bundle.batch_compressed_x[b]);

        for (std::size_t r = 0; r < shape.h; ++r)
        {
            if (r == query_indices[b].r_idx)
            {
                public_key.encrypt_asymmetric(context, query_y_one, bundle.batch_row_selectors[b][r]);
            }
            else
            {
                public_key.encrypt_asymmetric(context, query_y_zero, bundle.batch_row_selectors[b][r]);
            }
        }
    }

    return bundle;
}

} // namespace secppdl::pir
