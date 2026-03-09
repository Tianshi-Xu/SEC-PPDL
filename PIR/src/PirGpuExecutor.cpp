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
        (void)phantom_device_allocator();
    }

    PirGpuComputeResult run(
        Tensor3D<PhantomPlaintext> &db_plain, const std::vector<PhantomCiphertext> &expanded_x,
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
                std::vector<const PhantomPlaintext *> plain_terms(block_count, nullptr);
                const auto throw_cuda_error = [](const cudaError_t err, const char *op_name) {
                    if (err == cudaSuccess)
                    {
                        return;
                    }
                    throw std::runtime_error(std::string(op_name) + " failed: " + cudaGetErrorString(err));
                };

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

                const auto &context_data = context_.get_context_data(query_chain_index);
                const auto &parms = context_data.parms();
                const std::size_t poly_modulus_degree = parms.poly_modulus_degree();
                const std::size_t coeff_modulus_size = parms.coeff_modulus().size();
                const std::size_t rns_coeff_count = coeff_modulus_size * poly_modulus_degree;
                const std::size_t cipher_coeff_count = 2 * rns_coeff_count;
                const cudaStream_t stream = cudaStreamPerThread;

                bool use_2d_fusion = options_.enable_ct_pt_fusion && options_.enable_ct_pt_2d_fusion;
                if (use_2d_fusion)
                {
                    for (std::size_t c = 0; c < block_count; ++c)
                    {
                        const PhantomCiphertext &ct = expanded_x_ntt[c];
                        if (ct.size() != 2)
                        {
                            use_2d_fusion = false;
                            break;
                        }
                        if (ct.coeff_modulus_size() != coeff_modulus_size || ct.poly_modulus_degree() != poly_modulus_degree)
                        {
                            throw std::logic_error("expanded_x layout mismatch in ct-pt 2D fusion path.");
                        }
                    }
                }

                if (use_2d_fusion)
                {
                    const std::size_t tile_rows = std::min(row_count, std::max<std::size_t>(1, options_.ct_pt_2d_tile_rows));
                    auto ct_terms = phantom::util::make_cuda_auto_ptr<uint64_t>(block_count * cipher_coeff_count, stream);
                    auto pt_tile =
                        phantom::util::make_cuda_auto_ptr<uint64_t>(tile_rows * block_count * rns_coeff_count, stream);
                    auto ans_tile = phantom::util::make_cuda_auto_ptr<uint64_t>(tile_rows * cipher_coeff_count, stream);

                    for (std::size_t c = 0; c < block_count; ++c)
                    {
                        const std::size_t dst_offset = c * cipher_coeff_count;
                        throw_cuda_error(
                            cudaMemcpyAsync(
                                ct_terms.get() + dst_offset, expanded_x_ntt[c].data(), cipher_coeff_count * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, stream),
                            "copy expanded_x into ct_terms");
                    }

                    const double fused_scale = expanded_x_ntt.front().scale() * db_plain.at(0, 0, 0).scale();
                    const uint64_t fused_correction_factor = expanded_x_ntt.front().correction_factor();
                    const std::size_t fused_noise_scale_deg = expanded_x_ntt.front().GetNoiseScaleDeg();

                    for (std::size_t k = 0; k < chunk_count; ++k)
                    {
                        for (std::size_t row_base = 0; row_base < row_count; row_base += tile_rows)
                        {
                            const std::size_t active_rows = std::min(tile_rows, row_count - row_base);
                            for (std::size_t local_r = 0; local_r < active_rows; ++local_r)
                            {
                                const std::size_t r = row_base + local_r;
                                const std::size_t row_offset = local_r * block_count * rns_coeff_count;
                                for (std::size_t c = 0; c < block_count; ++c)
                                {
                                    const uint64_t *src = db_plain.at(r, c, k).data();
                                    uint64_t *dst = pt_tile.get() + row_offset + c * rns_coeff_count;
                                    throw_cuda_error(
                                        cudaMemcpyAsync(
                                            dst, src, rns_coeff_count * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream),
                                        "pack plaintext tile");
                                }
                            }

                            throw_cuda_error(
                                cudaMemsetAsync(ans_tile.get(), 0, active_rows * cipher_coeff_count * sizeof(uint64_t), stream),
                                "zero ans_tile");

                            phantom::launch_multiply_add_2d_fusion(
                                context_, pt_tile.get(), ct_terms.get(), ans_tile.get(), active_rows, block_count,
                                query_chain_index, block_count * rns_coeff_count, rns_coeff_count, cipher_coeff_count,
                                cipher_coeff_count, stream);
                            throw_cuda_error(cudaGetLastError(), "launch_multiply_add_2d_fusion");

                            for (std::size_t local_r = 0; local_r < active_rows; ++local_r)
                            {
                                const std::size_t r = row_base + local_r;
                                PhantomCiphertext &row_accum_ntt = row_terms[r];
                                row_accum_ntt.resize(context_, query_chain_index, 2, stream);
                                row_accum_ntt.set_ntt_form(true);
                                row_accum_ntt.set_scale(fused_scale);
                                row_accum_ntt.set_correction_factor(fused_correction_factor);
                                row_accum_ntt.SetNoiseScaleDeg(fused_noise_scale_deg);

                                const uint64_t *src = ans_tile.get() + local_r * cipher_coeff_count;
                                throw_cuda_error(
                                    cudaMemcpyAsync(
                                        row_accum_ntt.data(), src, cipher_coeff_count * sizeof(uint64_t),
                                        cudaMemcpyDeviceToDevice, stream),
                                    "unpack ans_tile");

                                phantom::transform_from_ntt_inplace(context_, row_accum_ntt);
                                if (options_.enable_ct_ct_fusion)
                                {
                                    phantom::multiply_and_relin_inplace(
                                        context_, row_accum_ntt, row_selectors[r], relin_keys_);
                                }
                                else
                                {
                                    phantom::multiply_inplace(context_, row_accum_ntt, row_selectors[r]);
                                    phantom::relinearize_inplace(context_, row_accum_ntt, relin_keys_);
                                }
                            }
                        }
                        phantom::add_many(context_, row_terms, t_layer2[k]);
                    }
                }
                else
                {
                    PhantomCiphertext accum_ntt;
                    for (std::size_t k = 0; k < chunk_count; ++k)
                    {
                        for (std::size_t r = 0; r < row_count; ++r)
                        {
                            for (std::size_t c = 0; c < block_count; ++c)
                            {
                                plain_terms[c] = &db_plain.at(r, c, k);
                            }

                            phantom::multiply_plain_ntt_many_ptrs(context_, expanded_x_ntt, plain_terms, accum_ntt);

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

                            // Keep zero-copy ownership transfer but avoid reusing a moved-from
                            // ciphertext object whose metadata may stay non-zero.
                            std::swap(row_terms[r], accum_ntt);
                        }
                        phantom::add_many(context_, row_terms, t_layer2[k]);
                    }
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

                for (std::size_t k = 1; k < chunk_count; ++k)
                {
                    phantom::rotate_inplace(context_, t_layer2[k], static_cast<int>(k), galois_keys_);
                }

                for (std::size_t k = 1; k < chunk_count; ++k)
                {
                    phantom::add_inplace(context_, result.answer, t_layer2[k]);
                }
            },
            true, "T_Compute_RotReduce");

        gpu_sync_or_throw("T_Compute_FinalAdd");
        return result;
    }

    std::vector<PirGpuComputeResult> run_batch(
        Tensor3D<PhantomPlaintext> &db_plain, const std::vector<std::vector<PhantomCiphertext>> &batch_expanded_x,
        const std::vector<std::vector<PhantomCiphertext>> &batch_row_selectors, std::size_t row_count,
        std::size_t block_count, std::size_t chunk_count) const
    {
        const std::size_t batch_size = batch_expanded_x.size();
        if (batch_size == 0)
        {
            return {};
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
            if (batch_expanded_x[b].size() != block_count)
            {
                throw std::invalid_argument("batch_expanded_x[b] block_count mismatch.");
            }
            if (batch_row_selectors[b].size() != row_count)
            {
                throw std::invalid_argument("batch_row_selectors[b] row_count mismatch.");
            }
        }

        std::vector<PirGpuComputeResult> results(batch_size);
        std::vector<std::vector<PhantomCiphertext>> t_layer2(batch_size, std::vector<PhantomCiphertext>(chunk_count));

        const double t_compute_muladd_ms = time_phase(
            [&]() {
                if (batch_expanded_x.front().empty())
                {
                    throw std::invalid_argument("batch_expanded_x is empty.");
                }
                const std::size_t query_chain_index = batch_expanded_x.front().front().chain_index();
                std::vector<std::vector<PhantomCiphertext>> batch_expanded_x_ntt = batch_expanded_x;

                for (std::size_t b = 0; b < batch_size; ++b)
                {
                    for (std::size_t c = 0; c < block_count; ++c)
                    {
                        if (!batch_expanded_x_ntt[b][c].is_ntt_form())
                        {
                            phantom::transform_to_ntt_inplace(context_, batch_expanded_x_ntt[b][c]);
                        }
                        if (batch_expanded_x_ntt[b][c].chain_index() != query_chain_index)
                        {
                            throw std::logic_error("expanded_x chain index mismatch in lazy-INTT path.");
                        }
                    }
                }

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

                std::vector<PhantomCiphertext> row_accum_ntt(batch_size);
                std::vector<std::vector<PhantomCiphertext>> row_terms(batch_size, std::vector<PhantomCiphertext>(row_count));
                std::vector<const PhantomPlaintext *> plain_terms(block_count, nullptr);

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
                                context_, batch_expanded_x_ntt[b], plain_terms, row_accum_ntt[b]);
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
                            std::swap(row_terms[b][r], row_accum_ntt[b]);
                        }
                    }
                    for (std::size_t b = 0; b < batch_size; ++b)
                    {
                        phantom::add_many(context_, row_terms[b], t_layer2[b][k]);
                    }
                }
            },
            true, "T_Compute_MulAddLazyINTT(Batch)");

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
                        phantom::add_inplace(context_, answers[b], t_layer2[b][k]);
                    }
                }
            },
            true, "T_Compute_RotReduce(Batch)");

        for (std::size_t b = 0; b < batch_size; ++b)
        {
            if (options_.capture_chunk_answers)
            {
                results[b].chunk_answers_before_rotation = t_layer2[b];
            }
            results[b].answer = std::move(answers[b]);
            results[b].t_compute_muladd_ms = t_compute_muladd_ms;
            results[b].t_compute_rot_ms = t_compute_rot_ms;
        }

        gpu_sync_or_throw("T_Compute_FinalAdd(Batch)");
        return results;
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

    PirExecutionOptions exec_options;
    exec_options.query_batch_size = [&]() {
        const char *env = std::getenv("PIR_QUERY_BATCH_SIZE");
        if (env == nullptr)
        {
            return std::size_t{ 1 };
        }
        const std::size_t parsed = static_cast<std::size_t>(std::stoull(env));
        if (parsed == 0)
        {
            throw std::invalid_argument("PIR_QUERY_BATCH_SIZE must be positive.");
        }
        return parsed;
    }();
    exec_options.enable_ct_pt_fusion = [&]() {
        const char *env = std::getenv("PIR_CT_PT_FUSION");
        if (env == nullptr)
        {
            return true;
        }
        return std::string(env) != "0";
    }();
    exec_options.enable_ct_pt_2d_fusion = [&]() {
        const char *env = std::getenv("PIR_CT_PT_2D_FUSION");
        if (env == nullptr)
        {
            return false;
        }
        return std::string(env) != "0";
    }();
    exec_options.ct_pt_2d_tile_rows = [&]() {
        const char *env = std::getenv("PIR_CT_PT_2D_TILE_ROWS");
        if (env == nullptr)
        {
            return std::size_t{ 32 };
        }
        const std::size_t parsed = static_cast<std::size_t>(std::stoull(env));
        if (parsed == 0)
        {
            throw std::invalid_argument("PIR_CT_PT_2D_TILE_ROWS must be positive.");
        }
        return parsed;
    }();
    exec_options.enable_ct_ct_fusion = true;
    exec_options.capture_chunk_answers = true;

    PirGpuExecutor executor(context, relin_keys, galois_keys, exec_options);
    if (exec_options.query_batch_size > 1)
    {
        std::vector<QueryIndex> query_indices(exec_options.query_batch_size);
        query_indices[0] = map_query(input.query_id, shape);
        std::random_device rd_batch;
        std::mt19937_64 gen_batch(rd_batch());
        std::uniform_int_distribution<std::size_t> qdist(0, input.num - 1);
        for (std::size_t b = 1; b < exec_options.query_batch_size; ++b)
        {
            query_indices[b] = map_query(qdist(gen_batch), shape);
        }

        BatchQueryBundle batch_query_bundle;
        latency.t_encrypt_ms = time_phase(
            [&]() {
                batch_query_bundle =
                    generate_batch_query_bundle(shape, query_indices, plain_mod, context, public_key, batch_encoder);
            },
            true, "T_Encrypt_Batch");

        std::vector<std::vector<PhantomCiphertext>> batch_expanded_x(exec_options.query_batch_size);
        latency.t_expand_ms = time_phase(
            [&]() {
                for (std::size_t b = 0; b < exec_options.query_batch_size; ++b)
                {
                    batch_expanded_x[b] = expand_query_sealpir_device(
                        batch_query_bundle.batch_compressed_x[b], static_cast<uint32_t>(shape.blocks_per_row), context,
                        galois_keys);
                }
            },
            true, "T_ExpandX_Batch");

        std::vector<PirGpuComputeResult> batch_results = executor.run_batch(
            db_plain, batch_expanded_x, batch_query_bundle.batch_row_selectors, shape.h, shape.blocks_per_row,
            shape.chunks);
        if (!batch_results.empty())
        {
            latency.t_compute_muladd_ms = batch_results[0].t_compute_muladd_ms;
            latency.t_compute_rot_ms = batch_results[0].t_compute_rot_ms;
        }

        const bool ok = verify_batch_results(
            batch_results, query_indices, db_values, shape, context, secret_key, batch_encoder);

        std::cout << "\n===== Phase Latency (ms) - Batch =====" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "batch_size:       " << exec_options.query_batch_size << std::endl;
        std::cout << "T_Encode:         " << latency.t_encode_ms << std::endl;
        std::cout << "T_Encrypt_Batch:  " << latency.t_encrypt_ms << std::endl;
        std::cout << "T_ExpandX_Batch:  " << latency.t_expand_ms << std::endl;
        std::cout << "ct_pt_fusion:     " << (exec_options.enable_ct_pt_fusion ? "ON" : "OFF") << std::endl;
        std::cout << "ct_pt_2d_fusion:  " << "N/A(single-query only)" << std::endl;
        std::cout << "T_Compute_MulAdd: " << latency.t_compute_muladd_ms << std::endl;
        std::cout << "T_Compute_Rot:    " << latency.t_compute_rot_ms << std::endl;
        std::cout << "RESULT:           " << (ok ? "PASS" : "FAIL") << std::endl;
        return ok ? 0 : 1;
    }

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
    std::cout << "ct_pt_fusion:     " << (exec_options.enable_ct_pt_fusion ? "ON" : "OFF") << std::endl;
    std::cout << "ct_pt_2d_fusion:  " << (exec_options.enable_ct_pt_2d_fusion ? "ON" : "OFF")
              << " (tile_rows=" << exec_options.ct_pt_2d_tile_rows << ")" << std::endl;
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
