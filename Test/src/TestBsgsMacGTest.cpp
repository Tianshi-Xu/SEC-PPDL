#include "PhantomBatchTestUtils.h"

#include "evaluate.cuh"
#include "batchencoder.h"
#include "secretkey.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace phantom_batch_test;

std::vector<PhantomCiphertext> build_reference_bsgs_outputs(
    const PhantomContext &context, const std::vector<PhantomCiphertext> &baby_ctxts,
    const std::vector<PhantomPlaintext> &plains, std::size_t baby_step_size,
    std::size_t total_diagonals) {
    const std::size_t giant_step_count =
        (total_diagonals + baby_step_size - 1) / baby_step_size;

    std::vector<PhantomCiphertext> refs;
    refs.reserve(giant_step_count);

    for (std::size_t giant_idx = 0; giant_idx < giant_step_count; ++giant_idx) {
        const std::size_t base = giant_idx * baby_step_size;
        const std::size_t terms = std::min(baby_step_size, total_diagonals - base);

        PhantomCiphertext accum =
            phantom::multiply_plain_ntt(context, baby_ctxts[0], plains[base]);
        for (std::size_t t = 1; t < terms; ++t) {
            phantom::multiply_plain_ntt_and_add_inplace(
                context, baby_ctxts[t], plains[base + t], accum);
        }
        refs.emplace_back(std::move(accum));
    }

    return refs;
}

TEST(BsgsMacGTest, BsgsMacMatchesReferenceReduction) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    constexpr std::size_t kBabyStep = 3;
    constexpr std::size_t kTotalDiagonals = 8;

    std::vector<PhantomCiphertext> baby_ctxts;
    baby_ctxts.reserve(kBabyStep);
    for (std::size_t i = 0; i < kBabyStep; ++i) {
        baby_ctxts.emplace_back(make_random_ntt_cipher(context, chain, 2, 1001 + i, 5.0));
    }

    std::vector<PhantomPlaintext> plains;
    plains.reserve(kTotalDiagonals);
    for (std::size_t i = 0; i < kTotalDiagonals; ++i) {
        plains.emplace_back(make_random_ntt_plain(context, chain, 1101 + i, 3.0));
    }

    const auto refs = build_reference_bsgs_outputs(
        context, baby_ctxts, plains, kBabyStep, kTotalDiagonals);

    std::vector<PhantomCiphertext> fused;
    phantom::multiply_plain_ntt_bsgs_mac(
        context, baby_ctxts, plains, kBabyStep, kTotalDiagonals, fused);
    sync_stream();

    ASSERT_EQ(fused.size(), refs.size());
    for (std::size_t i = 0; i < fused.size(); ++i) {
        expect_device_buffer_eq(fused[i].data(), refs[i].data(), refs[i].size() * refs[i].coeff_modulus_size() * refs[i].poly_modulus_degree());
        EXPECT_DOUBLE_EQ(fused[i].scale(), refs[i].scale());
        EXPECT_EQ(fused[i].chain_index(), refs[i].chain_index());
        EXPECT_EQ(fused[i].size(), refs[i].size());
    }

    std::vector<const PhantomPlaintext *> plain_ptrs;
    plain_ptrs.reserve(plains.size());
    for (const auto &plain : plains) {
        plain_ptrs.push_back(&plain);
    }

    std::vector<PhantomCiphertext> fused_ptrs;
    phantom::multiply_plain_ntt_bsgs_mac_ptrs(
        context, baby_ctxts, plain_ptrs, kBabyStep, kTotalDiagonals, fused_ptrs);
    sync_stream();

    ASSERT_EQ(fused_ptrs.size(), refs.size());
    for (std::size_t i = 0; i < fused_ptrs.size(); ++i) {
        expect_device_buffer_eq(
            fused_ptrs[i].data(), refs[i].data(),
            refs[i].size() * refs[i].coeff_modulus_size() * refs[i].poly_modulus_degree());
    }
}

TEST(BsgsMacGTest, BsgsMacValidatesInputShape) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    std::vector<PhantomCiphertext> baby_ctxts;
    baby_ctxts.emplace_back(make_random_ntt_cipher(context, chain, 2, 2001, 2.0));
    baby_ctxts.emplace_back(make_random_ntt_cipher(context, chain, 2, 2002, 2.0));

    std::vector<PhantomPlaintext> plains;
    plains.emplace_back(make_random_ntt_plain(context, chain, 2101, 2.0));
    plains.emplace_back(make_random_ntt_plain(context, chain, 2102, 2.0));
    plains.emplace_back(make_random_ntt_plain(context, chain, 2103, 2.0));

    std::vector<PhantomCiphertext> outputs;

    EXPECT_THROW(
        phantom::multiply_plain_ntt_bsgs_mac(context, baby_ctxts, plains, 3, 3, outputs),
        std::invalid_argument);

    EXPECT_THROW(
        phantom::multiply_plain_ntt_bsgs_mac(context, baby_ctxts, plains, 2, 2, outputs),
        std::invalid_argument);

    std::vector<const PhantomPlaintext *> plain_ptrs = {nullptr, &plains[1], &plains[2]};
    EXPECT_THROW(
        phantom::multiply_plain_ntt_bsgs_mac_ptrs(context, baby_ctxts, plain_ptrs, 2, 3, outputs),
        std::invalid_argument);
}

TEST(BsgsMacGTest, BsgsMacDecryptMatchesExpectedSlots) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);
    const auto &parms = context.get_context_data(chain).parms();
    const std::size_t plain_modulus = parms.plain_modulus().value();

    PhantomSecretKey secret_key(context);
    PhantomBatchEncoder encoder(context);
    const std::size_t slot_count = encoder.slot_count();

    constexpr std::size_t kBabyStep = 4;
    constexpr std::size_t kTotalDiagonals = 11;

    std::vector<PhantomCiphertext> baby_ctxts;
    std::vector<PhantomPlaintext> plains;
    std::vector<std::vector<uint64_t>> baby_slots(kBabyStep);
    std::vector<std::vector<uint64_t>> plain_slots(kTotalDiagonals);
    baby_ctxts.reserve(kBabyStep);
    plains.reserve(kTotalDiagonals);

    std::mt19937_64 rng(10901);
    std::uniform_int_distribution<uint64_t> dist(0ULL, plain_modulus - 1ULL);
    auto random_slot_vector = [&]() {
        std::vector<uint64_t> out(slot_count, 0ULL);
        for (std::size_t i = 0; i < slot_count; ++i) {
            out[i] = dist(rng);
        }
        return out;
    };

    for (std::size_t i = 0; i < kBabyStep; ++i) {
        baby_slots[i] = random_slot_vector();
        PhantomPlaintext plain = encoder.encode(context, baby_slots[i]);
        PhantomCiphertext ct;
        secret_key.encrypt_symmetric(context, plain, ct);
        phantom::transform_to_ntt_inplace(context, ct);
        baby_ctxts.emplace_back(std::move(ct));
    }

    for (std::size_t i = 0; i < kTotalDiagonals; ++i) {
        plain_slots[i] = random_slot_vector();
        PhantomPlaintext pt = encoder.encode(context, plain_slots[i]);
        phantom::transform_to_ntt_inplace(context, pt, chain);
        plains.emplace_back(std::move(pt));
    }

    std::vector<PhantomCiphertext> fused;
    phantom::multiply_plain_ntt_bsgs_mac(
        context, baby_ctxts, plains, kBabyStep, kTotalDiagonals, fused);
    sync_stream();

    const auto refs = build_reference_bsgs_outputs(
        context, baby_ctxts, plains, kBabyStep, kTotalDiagonals);

    auto decrypt_slots = [&](const PhantomCiphertext &ntt_ct) {
        PhantomCiphertext coeff_ct = ntt_ct;
        phantom::transform_from_ntt_inplace(context, coeff_ct);
        PhantomPlaintext plain;
        secret_key.decrypt(context, coeff_ct, plain);
        return encoder.decode(context, plain);
    };

    ASSERT_EQ(fused.size(), refs.size());
    for (std::size_t giant_idx = 0; giant_idx < fused.size(); ++giant_idx) {
        const std::size_t base = giant_idx * kBabyStep;
        const std::size_t term_count =
            std::min<std::size_t>(kBabyStep, kTotalDiagonals - base);

        std::vector<uint64_t> expected(slot_count, 0ULL);
        for (std::size_t s = 0; s < slot_count; ++s) {
            unsigned __int128 acc = 0;
            for (std::size_t t = 0; t < term_count; ++t) {
                acc += static_cast<unsigned __int128>(baby_slots[t][s]) *
                       plain_slots[base + t][s];
            }
            expected[s] = static_cast<uint64_t>(acc % plain_modulus);
        }

        const auto fused_slots = decrypt_slots(fused[giant_idx]);
        const auto ref_slots = decrypt_slots(refs[giant_idx]);
        EXPECT_EQ(fused_slots, expected);
        EXPECT_EQ(ref_slots, expected);
        EXPECT_EQ(fused_slots, ref_slots);
    }
}

TEST(BsgsMacGTest, BsgsMacPerformanceVsReferenceReduction) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    constexpr std::size_t kBabyStep = 8;
    constexpr std::size_t kTotalDiagonals = 64;
    constexpr std::size_t kWarmup = 5;
    constexpr std::size_t kIters = 20;

    std::vector<PhantomCiphertext> baby_ctxts;
    baby_ctxts.reserve(kBabyStep);
    for (std::size_t i = 0; i < kBabyStep; ++i) {
        baby_ctxts.emplace_back(make_random_ntt_cipher(context, chain, 2, 12001 + i, 5.0));
    }

    std::vector<PhantomPlaintext> plains;
    plains.reserve(kTotalDiagonals);
    for (std::size_t i = 0; i < kTotalDiagonals; ++i) {
        plains.emplace_back(make_random_ntt_plain(context, chain, 12101 + i, 3.0));
    }

    std::vector<PhantomCiphertext> baseline_out;
    std::vector<PhantomCiphertext> fused_out;
    auto run_baseline = [&]() {
        baseline_out = build_reference_bsgs_outputs(
            context, baby_ctxts, plains, kBabyStep, kTotalDiagonals);
    };
    auto run_fused = [&]() {
        phantom::multiply_plain_ntt_bsgs_mac(
            context, baby_ctxts, plains, kBabyStep, kTotalDiagonals, fused_out);
    };

    const float baseline_ms = measure_cuda_average_ms(kWarmup, kIters, run_baseline);
    const float fused_ms = measure_cuda_average_ms(kWarmup, kIters, run_fused);
    sync_stream();

    ASSERT_EQ(fused_out.size(), baseline_out.size());
    for (std::size_t i = 0; i < fused_out.size(); ++i) {
        const std::size_t data_count =
            fused_out[i].size() * fused_out[i].coeff_modulus_size() *
            fused_out[i].poly_modulus_degree();
        expect_device_buffer_eq(fused_out[i].data(), baseline_out[i].data(), data_count);
    }

    const float speedup = baseline_ms / fused_ms;
    std::cout << "[Perf][BsgsMac] baseline_ms=" << baseline_ms
              << ", fused_ms=" << fused_ms << ", speedup=" << speedup << std::endl;

    EXPECT_GT(baseline_ms, 0.0f);
    EXPECT_GT(fused_ms, 0.0f);
    EXPECT_GE(speedup, 0.75f);
}

} // namespace
