#include "PIR/PirGpuApps.h"

#include "PIR/PirAnswerGenerator.h"
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

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using phantom::EncryptionParameters;
using phantom::arith::CoeffModulus;
using phantom::arith::PlainModulus;
using phantom::scheme_type;

namespace secppdl::pir
{
namespace
{

class BatchPirGpuExecutor
{
public:
    BatchPirGpuExecutor(
        const PhantomContext &context, const PhantomRelinKey &relin_keys, const PhantomGaloisKey &galois_keys,
        BatchPirExecutionOptions options)
        : context_(context), relin_keys_(relin_keys), galois_keys_(galois_keys), options_(std::move(options))
    {
        (void)phantom_device_allocator();
    }

    std::vector<PirGpuComputeResult> run(
        Tensor3D<PhantomPlaintext> &db_plain, std::size_t batch_size,
        const std::vector<PhantomCiphertext> &batch_compressed_x,
        const std::vector<std::vector<PhantomCiphertext>> &batch_row_selectors, std::size_t row_count,
        std::size_t block_count, std::size_t chunk_count) const
    {
        if (batch_size == 0)
        {
            return {};
        }
        if (batch_compressed_x.size() != batch_size)
        {
            throw std::invalid_argument("batch_compressed_x size mismatch.");
        }
        if (batch_row_selectors.size() != batch_size)
        {
            throw std::invalid_argument("batch_row_selectors size mismatch.");
        }
        if (row_count == 0 || block_count == 0 || chunk_count == 0)
        {
            throw std::invalid_argument("row_count, block_count, and chunk_count must be positive.");
        }

        for (std::size_t b = 0; b < batch_size; ++b)
        {
            if (batch_row_selectors[b].size() != row_count)
            {
                throw std::invalid_argument("batch_row_selectors[b] row_count mismatch.");
            }
        }

        std::vector<std::vector<PhantomCiphertext>> batch_expanded_x(batch_size);
        time_phase(
            [&]() {
                const std::size_t group_size = std::max<std::size_t>(1, options_.expand_group_size);
                for (std::size_t base = 0; base < batch_size; base += group_size)
                {
                    const std::size_t end = std::min(batch_size, base + group_size);
                    for (std::size_t b = base; b < end; ++b)
                    {
                        batch_expanded_x[b] = expand_query_sealpir_device(
                            batch_compressed_x[b], static_cast<uint32_t>(block_count), context_, galois_keys_);
                        if (batch_expanded_x[b].size() != block_count)
                        {
                            throw std::logic_error("expand_query_sealpir_device block_count mismatch.");
                        }
                    }
                }
            },
            true, "Batch_T_ExpandX");

        if (batch_expanded_x.front().empty())
        {
            throw std::invalid_argument("expanded query selectors are empty.");
        }

        const std::size_t query_chain_index = batch_expanded_x.front().front().chain_index();
        time_phase(
            [&]() {
                for (std::size_t b = 0; b < batch_size; ++b)
                {
                    for (std::size_t c = 0; c < block_count; ++c)
                    {
                        PhantomCiphertext &selector = batch_expanded_x[b][c];
                        if (!selector.is_ntt_form())
                        {
                            phantom::transform_to_ntt_inplace(context_, selector);
                        }
                        if (selector.chain_index() != query_chain_index)
                        {
                            throw std::logic_error("expanded_x chain index mismatch in lazy-INTT path.");
                        }
                    }
                }
            },
            true, "Batch_T_ExpandX_ToNTT");

        std::vector<PirGpuComputeResult> results(batch_size);
        std::vector<std::vector<PhantomCiphertext>> t_layer2(batch_size, std::vector<PhantomCiphertext>(chunk_count));

        const double t_compute_muladd_ms = time_phase(
            [&]() {
                std::vector<PhantomCiphertext> row_accum_ntt(batch_size);
                std::vector<std::vector<PhantomCiphertext>> row_terms(batch_size, std::vector<PhantomCiphertext>(row_count));
                std::vector<const PhantomPlaintext *> plain_terms(block_count, nullptr);

                for (std::size_t r = 0; r < row_count; ++r)
                {
                    for (std::size_t c = 0; c < block_count; ++c)
                    {
                        for (std::size_t k = 0; k < chunk_count; ++k)
                        {
                            PhantomPlaintext &plain_ntt = db_plain.at(r, c, k);
                            if (!plain_ntt.is_ntt_form())
                            {
                                phantom::transform_to_ntt_inplace(context_, plain_ntt, query_chain_index);
                            }
                            else if (plain_ntt.chain_index() != query_chain_index)
                            {
                                throw std::logic_error("plain_ntt chain index mismatch in lazy-INTT path.");
                            }
                        }
                    }
                }

                for (std::size_t k = 0; k < chunk_count; ++k)
                {
                    for (std::size_t r = 0; r < row_count; ++r)
                    {
                        for (std::size_t c = 0; c < block_count; ++c)
                        {
                            plain_terms[c] = &db_plain.at(r, c, k);
                        }

                        for (std::size_t b = 0; b < batch_size; ++b)
                        {
                            phantom::multiply_plain_ntt_many_ptrs(
                                context_, batch_expanded_x[b], plain_terms, row_accum_ntt[b]);
                        }

                        for (std::size_t b = 0; b < batch_size; ++b)
                        {
                            phantom::transform_from_ntt_inplace(context_, row_accum_ntt[b]);
                            if (options_.enable_ct_ct_fusion)
                            {
                                phantom::multiply_and_relin_inplace(
                                    context_, row_accum_ntt[b], batch_row_selectors[b][r], relin_keys_);
                            }
                            else
                            {
                                phantom::multiply_inplace(context_, row_accum_ntt[b], batch_row_selectors[b][r]);
                                phantom::relinearize_inplace(context_, row_accum_ntt[b], relin_keys_);
                            }
                            // Keep zero-copy ownership transfer but avoid reusing a moved-from
                            // ciphertext object whose metadata may stay non-zero.
                            std::swap(row_terms[b][r], row_accum_ntt[b]);
                        }
                    }

                    for (std::size_t b = 0; b < batch_size; ++b)
                    {
                        phantom::add_many(context_, row_terms[b], t_layer2[b][k]);
                    }
                }
            },
            true, "Batch_T_Compute_MulAddLazyINTT");

        if (options_.capture_chunk_answers)
        {
            for (std::size_t b = 0; b < batch_size; ++b)
            {
                results[b].chunk_answers_before_rotation = t_layer2[b];
            }
        }

        std::vector<PhantomCiphertext> answers(batch_size);
        const double t_compute_rot_ms = time_phase(
            [&]() {
                for (std::size_t b = 0; b < batch_size; ++b)
                {
                    answers[b] = t_layer2[b][0];
                }

                if (chunk_count <= 1)
                {
                    return;
                }

                for (std::size_t k = 1; k < chunk_count; ++k)
                {
                    for (std::size_t b = 0; b < batch_size; ++b)
                    {
                        phantom::rotate_inplace(context_, t_layer2[b][k], static_cast<int>(k), galois_keys_);
                    }
                }

                for (std::size_t b = 0; b < batch_size; ++b)
                {
                    for (std::size_t k = 1; k < chunk_count; ++k)
                    {
                        if (options_.fuse_rotate_and_reduce)
                        {
                            phantom::add_inplace(context_, answers[b], t_layer2[b][k]);
                        }
                    }
                }
            },
            true,
            options_.fuse_rotate_and_reduce
                ? "Batch_T_Compute_RotReduce"
                : "Batch_T_Compute_Rot");

        for (std::size_t b = 0; b < batch_size; ++b)
        {
            if (options_.fuse_rotate_and_reduce)
            {
                results[b].answer = std::move(answers[b]);
            }
            else
            {
                results[b].answer = t_layer2[b][0];
                for (std::size_t k = 1; k < chunk_count; ++k)
                {
                    phantom::add_inplace(context_, results[b].answer, t_layer2[b][k]);
                }
            }
            results[b].t_compute_muladd_ms = t_compute_muladd_ms;
            results[b].t_compute_rot_ms = t_compute_rot_ms;
        }

        gpu_sync_or_throw("Batch_T_Compute_FinalAdd");
        return results;
    }

private:
    const PhantomContext &context_;
    const PhantomRelinKey &relin_keys_;
    const PhantomGaloisKey &galois_keys_;
    BatchPirExecutionOptions options_{};
};

} // namespace

int run_test_batch_pir_gpu(int argc, char **argv)
{
    constexpr std::size_t kPolyModulusDegree = 32768;

    try
    {
        const CliInput input = parse_cli(argc, argv);
        select_cuda_device_or_throw(input.gpu_id);
        (void)phantom_device_allocator();

        PirShape shape = build_shape(input.num, input.bit_width, kPolyModulusDegree, kPolyModulusDegree);

        EncryptionParameters parms(scheme_type::bfv);
        parms.set_poly_modulus_degree(kPolyModulusDegree);
        parms.set_coeff_modulus(CoeffModulus::Create(kPolyModulusDegree, { 54, 54, 54, 54 }));
        parms.set_plain_modulus(PlainModulus::Batching(kPolyModulusDegree, 17));
        parms.set_galois_elts(required_galois_elts_for_pir(
            static_cast<uint32_t>(kPolyModulusDegree), static_cast<uint32_t>(shape.blocks_per_row),
            static_cast<uint32_t>(shape.chunks)));

        PhantomContext context(parms);
        PhantomBatchEncoder batch_encoder(context);
        shape.slot_count = batch_encoder.slot_count();

        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
        PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

        const uint64_t plain_mod = context.key_context_data().parms().plain_modulus().value();

        std::cout << "INFO: num=" << input.num << ", bit_width=" << input.bit_width << ", batch_size=" << input.batch_size
                  << ", seed=" << input.seed << std::endl;
        std::cout << "INFO: SmartPIR shape: h=" << shape.h << ", w=" << shape.w << ", blocks_per_row="
                  << shape.blocks_per_row << ", chunks=" << shape.chunks << std::endl;

        std::mt19937_64 gen(input.seed);
        DbValueTensor db_values = generate_db_values(shape, plain_mod, gen);

        Tensor3D<PhantomPlaintext> db_plain;
        const double t_encode_ms = time_phase(
            [&]() {
                db_plain = encode_db_to_device(db_values, shape, context, batch_encoder);
            },
            true, "T_Encode");

        std::vector<std::size_t> query_ids(input.batch_size, 0);
        std::uniform_int_distribution<std::size_t> qdist(0, input.num - 1);
        for (std::size_t b = 0; b < input.batch_size; ++b)
        {
            query_ids[b] = qdist(gen);
        }

        std::vector<QueryIndex> query_indices(input.batch_size);
        for (std::size_t b = 0; b < input.batch_size; ++b)
        {
            query_indices[b] = map_query(query_ids[b], shape);
            std::cout << "INFO: query[" << b << "] id=" << query_indices[b].query_id << " -> (r=" << query_indices[b].r_idx
                      << ", c_block=" << query_indices[b].c_block << ", oft=" << query_indices[b].oft << ")"
                      << std::endl;
        }

        BatchQueryBundle query_bundle;
        const double t_encrypt_ms = time_phase(
            [&]() {
                query_bundle = generate_batch_query_bundle(shape, query_indices, plain_mod, context, public_key, batch_encoder);
            },
            true, "T_Encrypt_Batch");

        BatchPirExecutionOptions options;
        const bool ct_pt_fusion_enabled = [&]() {
            const char *env = std::getenv("PIR_CT_PT_FUSION");
            if (env == nullptr)
            {
                return true;
            }
            return std::string(env) != "0";
        }();
        options.enable_ct_pt_fusion = ct_pt_fusion_enabled;
        options.enable_ct_ct_fusion = true;
        options.fuse_rotate_and_reduce = true;
        options.capture_chunk_answers = true;
        options.expand_group_size = 4;

        BatchPirGpuExecutor executor(context, relin_keys, galois_keys, options);

        std::vector<PirGpuComputeResult> results;
        const double t_server_ms = time_phase(
            [&]() {
                results = executor.run(
                    db_plain, input.batch_size, query_bundle.batch_compressed_x, query_bundle.batch_row_selectors,
                    shape.h, shape.blocks_per_row, shape.chunks);
            },
            true, "T_Server_Batch");

        const bool ok = verify_batch_results(results, query_indices, db_values, shape, context, secret_key, batch_encoder);

        std::cout << "\n===== Batch GPU PIR Latency (ms) - Baseline =====" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "T_Encode:         " << t_encode_ms << std::endl;
        std::cout << "T_Encrypt_Batch:  " << t_encrypt_ms << std::endl;
        std::cout << "T_Server_Batch:   " << t_server_ms << std::endl;
        std::cout << "ct_pt_fusion:     " << (ct_pt_fusion_enabled ? "ON" : "OFF") << std::endl;
        if (!results.empty())
        {
            std::cout << "T_Compute_MulAdd: " << results[0].t_compute_muladd_ms << std::endl;
            std::cout << "T_Compute_Rot:    " << results[0].t_compute_rot_ms << std::endl;
        }

        if (ok)
        {
            std::cout << "RESULT: PASS (all batched queries verified)." << std::endl;
            return 0;
        }

        std::cout << "RESULT: FAIL (verification mismatch)." << std::endl;
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 2;
    }
}

} // namespace secppdl::pir
