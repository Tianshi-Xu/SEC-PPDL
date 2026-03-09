#include "PhantomBatchTestUtils.h"

#include "evaluate.cuh"

#include <algorithm>
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

} // namespace
