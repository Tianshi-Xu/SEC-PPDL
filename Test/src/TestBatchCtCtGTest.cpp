#include "PhantomBatchTestUtils.h"

#include "evaluate.cuh"
#include "secretkey.h"

#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace phantom_batch_test;

PhantomContext make_test_context_bgv() {
    const std::size_t poly_degree = 4096;
    phantom::EncryptionParameters parms(phantom::scheme_type::bgv);
    parms.set_poly_modulus_degree(poly_degree);
    parms.set_coeff_modulus(
        phantom::arith::CoeffModulus::Create(poly_degree, {40, 40, 40, 40}));
    parms.set_plain_modulus(phantom::arith::PlainModulus::Batching(poly_degree, 17));
    return PhantomContext(parms);
}

PhantomContext make_sweep_context(
    phantom::scheme_type scheme, std::size_t coeff_mod_count) {
    const std::size_t poly_degree = 4096;
    phantom::EncryptionParameters parms(scheme);
    parms.set_poly_modulus_degree(poly_degree);
    parms.set_coeff_modulus(
        phantom::arith::CoeffModulus::Create(
            poly_degree, std::vector<int>(coeff_mod_count, 40)));
    parms.set_plain_modulus(phantom::arith::PlainModulus::Batching(poly_degree, 17));
    return PhantomContext(parms);
}

PhantomBatchCiphertext pack_batch(
    const PhantomContext &context, const std::vector<PhantomCiphertext> &items) {
    if (items.empty()) {
        throw std::invalid_argument("pack_batch: items cannot be empty");
    }
    PhantomBatchCiphertext batch;
    batch.resize_like(context, items.front(), items.size());
    for (std::size_t i = 0; i < items.size(); ++i) {
        batch.copy_from(i, items[i]);
    }
    return batch;
}

void expect_batch_matches_items(
    const PhantomBatchCiphertext &batch, const std::vector<PhantomCiphertext> &items) {
    ASSERT_EQ(batch.batch_size(), items.size());
    const std::size_t item_words = batch.item_data_count();
    for (std::size_t i = 0; i < items.size(); ++i) {
        const auto batch_h =
            copy_device_to_host(batch.data() + i * item_words, item_words);
        const auto item_h = copy_device_to_host(items[i].data(), item_words);
        EXPECT_EQ(batch_h, item_h);
    }
}

TEST(BatchCtCtGTest, BatchMultiplyMatchesPerItemLoopBgvNtt) {
    PhantomContext context = make_test_context_bgv();
    const std::size_t chain = data_chain_index(context);

    constexpr std::size_t kBatch = 12;
    std::vector<PhantomCiphertext> lhs_items;
    std::vector<PhantomCiphertext> rhs_items;
    lhs_items.reserve(kBatch);
    rhs_items.reserve(kBatch);
    for (std::size_t i = 0; i < kBatch; ++i) {
        lhs_items.emplace_back(make_random_ntt_cipher(context, chain, 2, 21001 + i, 4.0));
        rhs_items.emplace_back(make_random_ntt_cipher(context, chain, 2, 22001 + i, 4.0));
    }

    std::vector<PhantomCiphertext> baseline = lhs_items;
    for (std::size_t i = 0; i < kBatch; ++i) {
        phantom::multiply_inplace(context, baseline[i], rhs_items[i]);
    }
    sync_stream();

    PhantomBatchCiphertext lhs_batch = pack_batch(context, lhs_items);
    PhantomBatchCiphertext rhs_batch = pack_batch(context, rhs_items);
    phantom::multiply_inplace(context, lhs_batch, rhs_batch);
    sync_stream();

    expect_batch_matches_items(lhs_batch, baseline);
    EXPECT_EQ(lhs_batch.size(), 3U);
}

TEST(BatchCtCtGTest, BatchMultiplyAndRelinMatchesPerItemLoopBfv) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    PhantomSecretKey secret_key(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    constexpr std::size_t kBatch = 8;
    std::vector<PhantomCiphertext> lhs_items;
    std::vector<PhantomCiphertext> rhs_items;
    lhs_items.reserve(kBatch);
    rhs_items.reserve(kBatch);
    for (std::size_t i = 0; i < kBatch; ++i) {
        auto lhs = make_random_ntt_cipher(context, chain, 2, 23001 + i, 1.0);
        auto rhs = make_random_ntt_cipher(context, chain, 2, 24001 + i, 1.0);
        lhs.set_ntt_form(false);
        rhs.set_ntt_form(false);
        lhs_items.emplace_back(std::move(lhs));
        rhs_items.emplace_back(std::move(rhs));
    }

    std::vector<PhantomCiphertext> baseline = lhs_items;
    for (std::size_t i = 0; i < kBatch; ++i) {
        phantom::multiply_and_relin_inplace(context, baseline[i], rhs_items[i], relin_keys);
    }
    sync_stream();

    PhantomBatchCiphertext lhs_batch = pack_batch(context, lhs_items);
    PhantomBatchCiphertext rhs_batch = pack_batch(context, rhs_items);
    phantom::multiply_and_relin_inplace(context, lhs_batch, rhs_batch, relin_keys);
    sync_stream();

    expect_batch_matches_items(lhs_batch, baseline);
    EXPECT_EQ(lhs_batch.size(), 2U);
}

TEST(BatchCtCtGTest, BatchRelinearizeMatchesPerItemLoopBfv) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    PhantomSecretKey secret_key(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    constexpr std::size_t kBatch = 10;
    std::vector<PhantomCiphertext> lhs_items;
    std::vector<PhantomCiphertext> rhs_items;
    lhs_items.reserve(kBatch);
    rhs_items.reserve(kBatch);
    for (std::size_t i = 0; i < kBatch; ++i) {
        auto lhs = make_random_ntt_cipher(context, chain, 2, 29001 + i, 1.0);
        auto rhs = make_random_ntt_cipher(context, chain, 2, 30001 + i, 1.0);
        lhs.set_ntt_form(false);
        rhs.set_ntt_form(false);
        lhs_items.emplace_back(std::move(lhs));
        rhs_items.emplace_back(std::move(rhs));
    }

    std::vector<PhantomCiphertext> mul_items = lhs_items;
    for (std::size_t i = 0; i < kBatch; ++i) {
        phantom::multiply_inplace(context, mul_items[i], rhs_items[i]);
    }
    sync_stream();

    std::vector<PhantomCiphertext> baseline = mul_items;
    for (std::size_t i = 0; i < kBatch; ++i) {
        phantom::relinearize_inplace(context, baseline[i], relin_keys);
    }
    sync_stream();

    PhantomBatchCiphertext batch_mul = pack_batch(context, mul_items);
    phantom::relinearize_inplace(context, batch_mul, relin_keys);
    sync_stream();

    expect_batch_matches_items(batch_mul, baseline);
    EXPECT_EQ(batch_mul.size(), 2U);
}

TEST(BatchCtCtGTest, BatchMultiplyRelinSweepFixedSeedAcrossSchemesLevelsAndBatchSizes) {
    enum class SweepOp {
        MultiplyOnly,
        MultiplyAndRelin
    };

    struct SchemeConfig {
        std::string name;
        phantom::scheme_type scheme;
        SweepOp op;
        bool use_ntt_input;
        double scale;
        std::uint64_t seed_base;
    };

    const std::array<SchemeConfig, 2> kSchemes{{
        {"bfv", phantom::scheme_type::bfv, SweepOp::MultiplyAndRelin, false, 1.0, 510000ULL},
        {"bgv", phantom::scheme_type::bgv, SweepOp::MultiplyOnly, true, 4.0, 610000ULL},
    }};
    const std::array<std::size_t, 3> kCoeffModLevels{{3U, 4U, 5U}};
    const std::array<std::size_t, 3> kBatchSizes{{1U, 4U, 12U}};

    for (const auto &scheme_cfg : kSchemes) {
        for (const std::size_t coeff_mod_level : kCoeffModLevels) {
            PhantomContext context = make_sweep_context(scheme_cfg.scheme, coeff_mod_level);
            const std::size_t chain = data_chain_index(context);
            PhantomSecretKey secret_key(context);
            PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

            for (const std::size_t batch_size : kBatchSizes) {
                std::ostringstream trace;
                trace << "scheme=" << scheme_cfg.name
                      << ",coeff_mod_level=" << coeff_mod_level
                      << ",batch_size=" << batch_size;
                SCOPED_TRACE(trace.str());

                std::vector<PhantomCiphertext> lhs_items;
                std::vector<PhantomCiphertext> rhs_items;
                lhs_items.reserve(batch_size);
                rhs_items.reserve(batch_size);
                for (std::size_t i = 0; i < batch_size; ++i) {
                    const std::uint64_t seed_offset =
                        static_cast<std::uint64_t>(coeff_mod_level * 1000 + i * 17);
                    auto lhs = make_random_ntt_cipher(
                        context, chain, 2, scheme_cfg.seed_base + seed_offset,
                        scheme_cfg.scale);
                    auto rhs = make_random_ntt_cipher(
                        context, chain, 2, scheme_cfg.seed_base + 500000ULL + seed_offset,
                        scheme_cfg.scale);

                    if (!scheme_cfg.use_ntt_input) {
                        lhs.set_ntt_form(false);
                        rhs.set_ntt_form(false);
                    }

                    lhs_items.emplace_back(std::move(lhs));
                    rhs_items.emplace_back(std::move(rhs));
                }

                std::vector<PhantomCiphertext> baseline = lhs_items;
                for (std::size_t i = 0; i < batch_size; ++i) {
                    if (scheme_cfg.op == SweepOp::MultiplyAndRelin) {
                        phantom::multiply_and_relin_inplace(
                            context, baseline[i], rhs_items[i], relin_keys);
                    } else {
                        phantom::multiply_inplace(context, baseline[i], rhs_items[i]);
                    }
                }
                sync_stream();

                PhantomBatchCiphertext lhs_batch = pack_batch(context, lhs_items);
                PhantomBatchCiphertext rhs_batch = pack_batch(context, rhs_items);
                if (scheme_cfg.op == SweepOp::MultiplyAndRelin) {
                    phantom::multiply_and_relin_inplace(context, lhs_batch, rhs_batch, relin_keys);
                } else {
                    phantom::multiply_inplace(context, lhs_batch, rhs_batch);
                }
                sync_stream();

                expect_batch_matches_items(lhs_batch, baseline);
                const std::size_t expected_size =
                    (scheme_cfg.op == SweepOp::MultiplyAndRelin) ? 2U : 3U;
                EXPECT_EQ(lhs_batch.size(), expected_size);
            }
        }
    }
}

TEST(BatchCtCtGTest, BatchMultiplyPerformanceVsPerItemLoopBgvNtt) {
    PhantomContext context = make_test_context_bgv();
    const std::size_t chain = data_chain_index(context);

    constexpr std::size_t kBatch = 32;
    constexpr std::size_t kWarmup = 3;
    constexpr std::size_t kIters = 15;

    std::vector<PhantomCiphertext> lhs_seed;
    std::vector<PhantomCiphertext> rhs_seed;
    lhs_seed.reserve(kBatch);
    rhs_seed.reserve(kBatch);
    for (std::size_t i = 0; i < kBatch; ++i) {
        lhs_seed.emplace_back(make_random_ntt_cipher(context, chain, 2, 25001 + i, 8.0));
        rhs_seed.emplace_back(make_random_ntt_cipher(context, chain, 2, 26001 + i, 8.0));
    }

    std::vector<PhantomCiphertext> baseline_items;
    PhantomBatchCiphertext batch_lhs_seed = pack_batch(context, lhs_seed);
    PhantomBatchCiphertext batch_rhs = pack_batch(context, rhs_seed);
    PhantomBatchCiphertext batch_work;

    auto run_baseline = [&]() {
        baseline_items = lhs_seed;
        for (std::size_t i = 0; i < kBatch; ++i) {
            phantom::multiply_inplace(context, baseline_items[i], rhs_seed[i]);
        }
    };
    auto run_batch = [&]() {
        batch_work = batch_lhs_seed;
        phantom::multiply_inplace(context, batch_work, batch_rhs);
    };

    const float baseline_ms = measure_cuda_average_ms(kWarmup, kIters, run_baseline);
    const float batch_ms = measure_cuda_average_ms(kWarmup, kIters, run_batch);
    sync_stream();

    const float speedup = baseline_ms / batch_ms;
    std::cout << "[Perf][BatchCtCtMultiply] baseline_ms=" << baseline_ms
              << ", batch_ms=" << batch_ms << ", speedup=" << speedup << std::endl;

    EXPECT_GT(baseline_ms, 0.0f);
    EXPECT_GT(batch_ms, 0.0f);
}

TEST(BatchCtCtGTest, BatchMultiplyAndRelinPerformanceVsPerItemLoopBfv) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    PhantomSecretKey secret_key(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    constexpr std::size_t kBatch = 12;
    constexpr std::size_t kWarmup = 3;
    constexpr std::size_t kIters = 10;

    std::vector<PhantomCiphertext> lhs_seed;
    std::vector<PhantomCiphertext> rhs_seed;
    lhs_seed.reserve(kBatch);
    rhs_seed.reserve(kBatch);
    for (std::size_t i = 0; i < kBatch; ++i) {
        auto lhs = make_random_ntt_cipher(context, chain, 2, 27001 + i, 1.0);
        auto rhs = make_random_ntt_cipher(context, chain, 2, 28001 + i, 1.0);
        lhs.set_ntt_form(false);
        rhs.set_ntt_form(false);
        lhs_seed.emplace_back(std::move(lhs));
        rhs_seed.emplace_back(std::move(rhs));
    }

    std::vector<PhantomCiphertext> baseline_items;
    PhantomBatchCiphertext batch_lhs_seed = pack_batch(context, lhs_seed);
    PhantomBatchCiphertext batch_rhs = pack_batch(context, rhs_seed);
    PhantomBatchCiphertext batch_work;

    auto run_baseline = [&]() {
        baseline_items = lhs_seed;
        for (std::size_t i = 0; i < kBatch; ++i) {
            phantom::multiply_and_relin_inplace(
                context, baseline_items[i], rhs_seed[i], relin_keys);
        }
    };
    auto run_batch = [&]() {
        batch_work = batch_lhs_seed;
        phantom::multiply_and_relin_inplace(context, batch_work, batch_rhs, relin_keys);
    };

    const float baseline_ms = measure_cuda_average_ms(kWarmup, kIters, run_baseline);
    const float batch_ms = measure_cuda_average_ms(kWarmup, kIters, run_batch);
    sync_stream();

    const float speedup = baseline_ms / batch_ms;
    std::cout << "[Perf][BatchCtCtMulRelin] baseline_ms=" << baseline_ms
              << ", batch_ms=" << batch_ms << ", speedup=" << speedup << std::endl;

    EXPECT_GT(baseline_ms, 0.0f);
    EXPECT_GT(batch_ms, 0.0f);
}

} // namespace
