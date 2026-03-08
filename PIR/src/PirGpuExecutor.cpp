#include "PIR/PirGpuApps.h"

#include "PIR/PirAnswerGenerator.h"
#include "PIR/PirCiphertextOps.h"
#include "PIR/PirDatabaseGenerator.h"
#include "PIR/PirQueryGenerator.h"
#include "PIR/PirRuntime.h"
#include "PIR/PirShapeBuilder.h"
#include "PIR/PirTypes.h"

#include <phantom/batchencoder.h>
#include <phantom/context.cuh>
#include <phantom/evaluate.cuh>
#include <phantom/phantom_memory_pool.cuh>
#include <phantom/secretkey.h>

#include <future>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using phantom::EncryptionParameters;
using phantom::scheme_type;
using phantom::arith::CoeffModulus;
using phantom::arith::PlainModulus;

namespace secppdl::pir
{
namespace
{

class PirGpuExecutor
{
public:
    PirGpuExecutor(
        const PhantomContext &context, const PhantomRelinKey &relin_keys, const PhantomGaloisKey &galois_keys,
        PirExecutionOptions options)
        : context_(context), relin_keys_(relin_keys), galois_keys_(galois_keys), options_(options)
    {
        if (options_.query_batch_size != 1)
        {
            throw std::invalid_argument("query_batch_size > 1 is reserved for next batch-enabled version.");
        }
        (void)phantom_device_allocator();
    }

    PirGpuComputeResult run(
        const Tensor3D<PhantomPlaintext> &db_plain, const std::vector<PhantomCiphertext> &expanded_x,
        const std::vector<PhantomCiphertext> &row_selectors, std::size_t row_count, std::size_t block_count,
        std::size_t chunk_count) const
    {
        if (row_count == 0 || block_count == 0 || chunk_count == 0)
        {
            throw std::invalid_argument("row_count, block_count, and chunk_count must be positive.");
        }
        if (row_selectors.size() != row_count)
        {
            throw std::invalid_argument("row_selectors size mismatch.");
        }
        if (expanded_x.size() != block_count)
        {
            throw std::invalid_argument("expanded_x size mismatch.");
        }

        PirGpuComputeResult result;
        std::vector<PhantomCiphertext> t_layer2(chunk_count);

        result.t_compute_muladd_ms = time_phase(
            [&]() {
                if (expanded_x.empty())
                {
                    throw std::invalid_argument("expanded_x is empty.");
                }
                const std::size_t query_chain_index = expanded_x.front().chain_index();

                std::vector<PhantomCiphertext> expanded_x_ntt = expanded_x;
                for (std::size_t c = 0; c < block_count; ++c)
                {
                    if (!expanded_x_ntt[c].is_ntt_form())
                    {
                        phantom::transform_to_ntt_inplace(context_, expanded_x_ntt[c]);
                    }
                    if (expanded_x_ntt[c].chain_index() != query_chain_index)
                    {
                        throw std::logic_error("expanded_x chain index mismatch in lazy-INTT path.");
                    }
                }

                std::vector<PhantomCiphertext> row_terms(row_count);
                PhantomCiphertext accum_ntt;
                PhantomCiphertext scratch_prod_ntt;
                for (std::size_t k = 0; k < chunk_count; ++k)
                {
                    for (std::size_t r = 0; r < row_count; ++r)
                    {
                        bool initialized = false;
                        for (std::size_t c = 0; c < block_count; ++c)
                        {
                            PhantomPlaintext plain_ntt = db_plain.at(r, c, k);
                            if (!plain_ntt.is_ntt_form())
                            {
                                phantom::transform_to_ntt_inplace(context_, plain_ntt, query_chain_index);
                            }
                            else if (plain_ntt.chain_index() != query_chain_index)
                            {
                                throw std::logic_error("plain_ntt chain index mismatch in lazy-INTT path.");
                            }

                            if (!initialized)
                            {
                                copy_ciphertext_device_fast(expanded_x_ntt[c], accum_ntt);
                                phantom::multiply_plain_ntt_inplace(context_, accum_ntt, plain_ntt);
                                initialized = true;
                            }
                            else
                            {
                                if (options_.enable_ct_pt_fusion)
                                {
                                    phantom::multiply_plain_ntt_and_add_inplace(
                                        context_, expanded_x_ntt[c], plain_ntt, accum_ntt);
                                }
                                else
                                {
                                    copy_ciphertext_device_fast(expanded_x_ntt[c], scratch_prod_ntt);
                                    phantom::multiply_plain_ntt_inplace(context_, scratch_prod_ntt, plain_ntt);
                                    phantom::add_inplace(context_, accum_ntt, scratch_prod_ntt);
                                }
                            }
                        }

                        if (!initialized)
                        {
                            throw std::logic_error("accum_ntt is uninitialized.");
                        }

                        phantom::transform_from_ntt_inplace(context_, accum_ntt);

                        if (options_.enable_ct_ct_fusion)
                        {
                            phantom::multiply_and_relin_inplace(context_, accum_ntt, row_selectors[r], relin_keys_);
                        }
                        else
                        {
                            phantom::multiply_inplace(context_, accum_ntt, row_selectors[r]);
                            phantom::relinearize_inplace(context_, accum_ntt, relin_keys_);
                        }

                        copy_ciphertext_device_fast(accum_ntt, row_terms[r]);
                    }
                    phantom::add_many(context_, row_terms, t_layer2[k]);
                }
            },
            true, "T_Compute_MulAddLazyINTT");

        if (options_.capture_chunk_answers)
        {
            result.chunk_answers_before_rotation = t_layer2;
        }

        result.t_compute_rot_ms = time_phase(
            [&]() {
                result.answer = t_layer2[0];
                if (chunk_count <= 1)
                {
                    return;
                }

                int current_device = 0;
                cudaError_t device_err = cudaGetDevice(&current_device);
                if (device_err != cudaSuccess)
                {
                    throw std::runtime_error(
                        "cudaGetDevice failed before async rotate: " + std::string(cudaGetErrorString(device_err)));
                }

                std::vector<std::future<void>> rotate_jobs;
                rotate_jobs.reserve(chunk_count - 1);
                for (std::size_t k = 1; k < chunk_count; ++k)
                {
                    rotate_jobs.emplace_back(std::async(std::launch::async, [&, k, current_device]() {
                        cudaError_t set_err = cudaSetDevice(current_device);
                        if (set_err != cudaSuccess)
                        {
                            throw std::runtime_error(
                                "cudaSetDevice failed in async rotate worker: " + std::string(cudaGetErrorString(set_err)));
                        }
                        phantom::rotate_inplace(context_, t_layer2[k], static_cast<int>(k), galois_keys_);
                    }));
                }
                for (auto &job : rotate_jobs)
                {
                    job.get();
                }

                for (std::size_t k = 1; k < chunk_count; ++k)
                {
                    phantom::add_inplace(context_, result.answer, t_layer2[k]);
                }
            },
            true, "T_Compute_RotReduceAsync");

        gpu_sync_or_throw("T_Compute_FinalAdd");
        return result;
    }

private:
    const PhantomContext &context_;
    const PhantomRelinKey &relin_keys_;
    const PhantomGaloisKey &galois_keys_;
    PirExecutionOptions options_{};
};

} // namespace

int run_test_pir_gpu_interactive()
{
    constexpr std::size_t kPolyModulusDegree = 32768;

    PirUserInput input = read_user_input();
    PirShape shape = build_shape(input, kPolyModulusDegree, kPolyModulusDegree);

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(kPolyModulusDegree);
    parms.set_coeff_modulus(CoeffModulus::Create(kPolyModulusDegree, { 54, 54, 54, 54 }));
    parms.set_plain_modulus(PlainModulus::Batching(kPolyModulusDegree, 17));
    parms.set_galois_elts(required_galois_elts_for_pir(
        static_cast<uint32_t>(kPolyModulusDegree), static_cast<uint32_t>(shape.blocks_per_row),
        static_cast<uint32_t>(shape.chunks)));

    PhantomContext context(parms);
    print_line(__LINE__);
    print_parameters(context);

    PhantomBatchEncoder batch_encoder(context);
    shape.slot_count = batch_encoder.slot_count();

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

    const uint64_t plain_mod = context.key_context_data().parms().plain_modulus().value();

    std::cout << "INFO: SmartPIR dimensions: h=" << shape.h << ", w=" << shape.w << ", w/N=" << shape.blocks_per_row
              << ", chunks=" << shape.chunks << std::endl;
    std::cout << "INFO: Query mapping: r_idx=" << shape.r_idx << ", c_idx=" << shape.c_idx
              << ", c_block=" << shape.c_block << ", oft=" << shape.oft << std::endl;
    std::cout << "INFO: Generating DB values on host and encoding DB plaintexts on GPU." << std::endl;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    DbValueTensor db_values = generate_db_values(shape, plain_mod, gen);

    PirPhaseLatency latency;

    Tensor3D<PhantomPlaintext> db_plain;
    latency.t_encode_ms = time_phase(
        [&]() {
            db_plain = encode_db_to_device(db_values, shape, context, batch_encoder);
        },
        true, "T_Encode");

    PirQueryBundle query_bundle;
    latency.t_encrypt_ms = time_phase(
        [&]() {
            query_bundle = generate_query_bundle(shape, plain_mod, context, public_key, batch_encoder);
        },
        true, "T_Encrypt");

    std::vector<PhantomCiphertext> expanded_x;
    latency.t_expand_ms = time_phase(
        [&]() {
            expanded_x = expand_query_sealpir_device(
                query_bundle.compressed_x, static_cast<uint32_t>(shape.blocks_per_row), context, galois_keys);
        },
        true, "T_ExpandX");

    PirExecutionOptions exec_options;
    exec_options.query_batch_size = 1;
    exec_options.enable_ct_pt_fusion = true;
    exec_options.enable_ct_ct_fusion = true;
    exec_options.capture_chunk_answers = true;

    PirGpuExecutor executor(context, relin_keys, galois_keys, exec_options);
    PirGpuComputeResult compute_result_baseline =
        executor.run(db_plain, expanded_x, query_bundle.row_selectors, shape.h, shape.blocks_per_row, shape.chunks);
    latency.t_compute_muladd_ms = compute_result_baseline.t_compute_muladd_ms;
    latency.t_compute_rot_ms = compute_result_baseline.t_compute_rot_ms;

    PhantomPlaintext answer_pt_baseline;
    latency.t_decrypt_ms = time_phase(
        [&]() {
            secret_key.decrypt(context, compute_result_baseline.answer, answer_pt_baseline);
        },
        true, "T_Decrypt_Baseline");

    std::vector<uint64_t> answer_plain_baseline;
    latency.t_decode_ms = time_phase(
        [&]() {
            batch_encoder.decode(context, answer_pt_baseline, answer_plain_baseline);
        },
        true, "T_Decode_Baseline");

    std::cout << "\n===== Phase Latency (ms) - Baseline =====" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "T_Encode:         " << latency.t_encode_ms << std::endl;
    std::cout << "T_Encrypt:        " << latency.t_encrypt_ms << std::endl;
    std::cout << "T_ExpandX:        " << latency.t_expand_ms << std::endl;
    std::cout << "T_Compute_MulAdd: " << latency.t_compute_muladd_ms << std::endl;
    std::cout << "T_Compute_Rot:    " << latency.t_compute_rot_ms << std::endl;
    std::cout << "T_Decrypt:        " << latency.t_decrypt_ms << std::endl;
    std::cout << "T_Decode:         " << latency.t_decode_ms << std::endl;

    if (shape.oft < answer_plain_baseline.size())
    {
        std::cout << "INFO: Final answer slot[" << shape.oft << "] = " << answer_plain_baseline[shape.oft] << std::endl;
    }

    std::cout << "\n===== Retrieval Spot Check (Baseline) =====" << std::endl;
    print_chunk_spot_check(
        compute_result_baseline.chunk_answers_before_rotation, shape, db_values, context, secret_key, batch_encoder);

    const std::size_t num_primes = context.key_context_data().parms().coeff_modulus().size();
    print_overhead_estimates(shape.h * shape.blocks_per_row, shape.chunks, shape.poly_modulus_degree, num_primes);

    return 0;
}

} // namespace secppdl::pir
