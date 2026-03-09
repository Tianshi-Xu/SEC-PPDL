#include "PhantomBatchTestUtils.h"

#include "evaluate.cuh"
#include "batchencoder.h"
#include "ntt.cuh"
#include "secretkey.h"

#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace phantom_batch_test;

TEST(BatchOpsGTest, BatchNttKernelMatchesPerItemLoop) {
#ifndef RNS_POLY_BATCH
    GTEST_SKIP() << "RNS_POLY_BATCH is disabled; batch NTT kernel path is unavailable.";
#else
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);
    const auto &parms = context.get_context_data(chain).parms();
    const std::size_t coeff_modulus_size = parms.coeff_modulus().size();
    const std::size_t poly_degree = parms.poly_modulus_degree();
    const std::size_t coeff_count = coeff_modulus_size * poly_degree;
    const std::size_t batch_num = 4;

    auto input = random_rns_data(parms.coeff_modulus(), poly_degree, batch_num, 401);

    auto d_batch = phantom::util::make_cuda_auto_ptr<uint64_t>(input.size(), cudaStreamPerThread);
    auto d_loop = phantom::util::make_cuda_auto_ptr<uint64_t>(input.size(), cudaStreamPerThread);
    copy_host_to_device(d_batch.get(), input);
    copy_host_to_device(d_loop.get(), input);

    nwt_2d_radix8_forward_inplace(
        d_batch.get(), context.gpu_rns_tables(), coeff_modulus_size, 0, batch_num,
        cudaStreamPerThread);

    for (std::size_t b = 0; b < batch_num; ++b) {
        nwt_2d_radix8_forward_inplace(
            d_loop.get() + b * coeff_count, context.gpu_rns_tables(), coeff_modulus_size, 0,
            cudaStreamPerThread);
    }
    sync_stream();
    expect_device_buffer_eq(d_batch.get(), d_loop.get(), input.size());

    nwt_2d_radix8_backward_inplace(
        d_batch.get(), context.gpu_rns_tables(), coeff_modulus_size, 0, batch_num,
        cudaStreamPerThread);

    for (std::size_t b = 0; b < batch_num; ++b) {
        nwt_2d_radix8_backward_inplace(
            d_loop.get() + b * coeff_count, context.gpu_rns_tables(), coeff_modulus_size, 0,
            cudaStreamPerThread);
    }
    sync_stream();
    expect_device_buffer_eq(d_batch.get(), d_loop.get(), input.size());
    const auto output = copy_device_to_host(d_batch.get(), input.size());
    EXPECT_EQ(output, input);
#endif
}

TEST(BatchOpsGTest, MultiplyPlainNttManyPtrsMatchesScalarAccumulation) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    constexpr std::size_t kTermCount = 4;
    std::vector<PhantomCiphertext> encrypteds;
    std::vector<PhantomPlaintext> plains;
    std::vector<const PhantomPlaintext *> plain_ptrs;
    encrypteds.reserve(kTermCount);
    plains.reserve(kTermCount);
    plain_ptrs.reserve(kTermCount);

    for (std::size_t i = 0; i < kTermCount; ++i) {
        encrypteds.emplace_back(make_random_ntt_cipher(context, chain, 2, 501 + i, 6.0));
        plains.emplace_back(make_random_ntt_plain(context, chain, 601 + i, 3.0));
        plain_ptrs.push_back(&plains.back());
    }

    PhantomCiphertext reference =
        phantom::multiply_plain_ntt(context, encrypteds[0], plains[0]);
    for (std::size_t i = 1; i < kTermCount; ++i) {
        phantom::multiply_plain_ntt_and_add_inplace(
            context, encrypteds[i], plains[i], reference);
    }

    PhantomCiphertext fused;
    phantom::multiply_plain_ntt_many_ptrs(context, encrypteds, plain_ptrs, fused);
    sync_stream();

    const std::size_t data_count =
        fused.size() * fused.coeff_modulus_size() * fused.poly_modulus_degree();
    expect_device_buffer_eq(fused.data(), reference.data(), data_count);
    EXPECT_DOUBLE_EQ(fused.scale(), reference.scale());
    EXPECT_EQ(fused.chain_index(), reference.chain_index());
}

TEST(BatchOpsGTest, MultiplyPlainNttManyValueApiMatchesPtrApi) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    constexpr std::size_t kTermCount = 3;
    std::vector<PhantomCiphertext> encrypteds;
    std::vector<PhantomPlaintext> plains;
    std::vector<const PhantomPlaintext *> plain_ptrs;
    encrypteds.reserve(kTermCount);
    plains.reserve(kTermCount);
    plain_ptrs.reserve(kTermCount);

    for (std::size_t i = 0; i < kTermCount; ++i) {
        encrypteds.emplace_back(make_random_ntt_cipher(context, chain, 2, 701 + i, 4.0));
        plains.emplace_back(make_random_ntt_plain(context, chain, 801 + i, 2.0));
        plain_ptrs.push_back(&plains.back());
    }

    PhantomCiphertext from_value_api;
    PhantomCiphertext from_ptr_api;
    phantom::multiply_plain_ntt_many(context, encrypteds, plains, from_value_api);
    phantom::multiply_plain_ntt_many_ptrs(context, encrypteds, plain_ptrs, from_ptr_api);
    sync_stream();

    const std::size_t data_count =
        from_value_api.size() * from_value_api.coeff_modulus_size() * from_value_api.poly_modulus_degree();
    expect_device_buffer_eq(from_value_api.data(), from_ptr_api.data(), data_count);
    EXPECT_DOUBLE_EQ(from_value_api.scale(), from_ptr_api.scale());
}

TEST(BatchOpsGTest, MultiplyPlainNttManyPtrsDecryptMatchesExpectedSlots) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);
    const auto &parms = context.get_context_data(chain).parms();
    const std::size_t plain_modulus = parms.plain_modulus().value();

    PhantomSecretKey secret_key(context);
    PhantomBatchEncoder encoder(context);
    const std::size_t slot_count = encoder.slot_count();

    constexpr std::size_t kTermCount = 4;
    std::vector<PhantomCiphertext> encrypteds;
    std::vector<PhantomPlaintext> plains;
    std::vector<const PhantomPlaintext *> plain_ptrs;
    std::vector<std::vector<uint64_t>> ct_slots(kTermCount);
    std::vector<std::vector<uint64_t>> pt_slots(kTermCount);
    encrypteds.reserve(kTermCount);
    plains.reserve(kTermCount);
    plain_ptrs.reserve(kTermCount);

    std::mt19937_64 rng(9901);
    std::uniform_int_distribution<uint64_t> dist(0ULL, plain_modulus - 1ULL);
    auto random_slot_vector = [&]() {
        std::vector<uint64_t> out(slot_count, 0ULL);
        for (std::size_t i = 0; i < slot_count; ++i) {
            out[i] = dist(rng);
        }
        return out;
    };

    for (std::size_t i = 0; i < kTermCount; ++i) {
        ct_slots[i] = random_slot_vector();
        PhantomPlaintext ct_plain = encoder.encode(context, ct_slots[i]);
        PhantomCiphertext ct;
        secret_key.encrypt_symmetric(context, ct_plain, ct);
        phantom::transform_to_ntt_inplace(context, ct);
        encrypteds.emplace_back(std::move(ct));

        pt_slots[i] = random_slot_vector();
        PhantomPlaintext pt = encoder.encode(context, pt_slots[i]);
        phantom::transform_to_ntt_inplace(context, pt, chain);
        plains.emplace_back(std::move(pt));
        plain_ptrs.push_back(&plains.back());
    }

    PhantomCiphertext reference =
        phantom::multiply_plain_ntt(context, encrypteds[0], plains[0]);
    for (std::size_t i = 1; i < kTermCount; ++i) {
        phantom::multiply_plain_ntt_and_add_inplace(
            context, encrypteds[i], plains[i], reference);
    }

    PhantomCiphertext fused;
    phantom::multiply_plain_ntt_many_ptrs(context, encrypteds, plain_ptrs, fused);
    sync_stream();

    std::vector<uint64_t> expected(slot_count, 0ULL);
    for (std::size_t s = 0; s < slot_count; ++s) {
        unsigned __int128 acc = 0;
        for (std::size_t i = 0; i < kTermCount; ++i) {
            acc += static_cast<unsigned __int128>(ct_slots[i][s]) * pt_slots[i][s];
        }
        expected[s] = static_cast<uint64_t>(acc % plain_modulus);
    }

    auto decrypt_slots = [&](const PhantomCiphertext &ntt_ct) {
        PhantomCiphertext coeff_ct = ntt_ct;
        phantom::transform_from_ntt_inplace(context, coeff_ct);
        PhantomPlaintext plain;
        secret_key.decrypt(context, coeff_ct, plain);
        return encoder.decode(context, plain);
    };

    const auto ref_slots = decrypt_slots(reference);
    const auto fused_slots = decrypt_slots(fused);

    EXPECT_EQ(ref_slots, expected);
    EXPECT_EQ(fused_slots, expected);
    EXPECT_EQ(fused_slots, ref_slots);
}

TEST(BatchOpsGTest, MultiplyPlainNttManyPtrsPerformanceVsScalarAccumulation) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    constexpr std::size_t kTermCount = 16;
    constexpr std::size_t kWarmup = 5;
    constexpr std::size_t kIters = 30;

    std::vector<PhantomCiphertext> encrypteds;
    std::vector<PhantomPlaintext> plains;
    std::vector<const PhantomPlaintext *> plain_ptrs;
    encrypteds.reserve(kTermCount);
    plains.reserve(kTermCount);
    plain_ptrs.reserve(kTermCount);

    for (std::size_t i = 0; i < kTermCount; ++i) {
        encrypteds.emplace_back(make_random_ntt_cipher(context, chain, 2, 8801 + i, 8.0));
        plains.emplace_back(make_random_ntt_plain(context, chain, 8901 + i, 4.0));
        plain_ptrs.push_back(&plains.back());
    }

    PhantomCiphertext baseline_out;
    PhantomCiphertext fused_out;
    auto run_baseline = [&]() {
        baseline_out = phantom::multiply_plain_ntt(context, encrypteds[0], plains[0]);
        for (std::size_t i = 1; i < kTermCount; ++i) {
            phantom::multiply_plain_ntt_and_add_inplace(
                context, encrypteds[i], plains[i], baseline_out);
        }
    };
    auto run_fused = [&]() {
        phantom::multiply_plain_ntt_many_ptrs(context, encrypteds, plain_ptrs, fused_out);
    };

    const float baseline_ms = measure_cuda_average_ms(kWarmup, kIters, run_baseline);
    const float fused_ms = measure_cuda_average_ms(kWarmup, kIters, run_fused);
    sync_stream();

    const std::size_t data_count =
        fused_out.size() * fused_out.coeff_modulus_size() * fused_out.poly_modulus_degree();
    expect_device_buffer_eq(fused_out.data(), baseline_out.data(), data_count);

    const float speedup = baseline_ms / fused_ms;
    std::cout << "[Perf][BatchOps] multiply_plain_ntt_many_ptrs baseline_ms=" << baseline_ms
              << ", fused_ms=" << fused_ms << ", speedup=" << speedup << std::endl;

    EXPECT_GT(baseline_ms, 0.0f);
    EXPECT_GT(fused_ms, 0.0f);
    EXPECT_GE(speedup, 0.75f);
}

} // namespace
