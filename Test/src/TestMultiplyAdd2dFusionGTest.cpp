#include "PhantomBatchTestUtils.h"

#include "evaluate.cuh"
#include "batchencoder.h"
#include "secretkey.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using namespace phantom_batch_test;

void reference_multiply_add_2d_fusion(
    const std::vector<uint64_t> &pt_matrix, const std::vector<uint64_t> &ct_terms,
    const std::vector<uint64_t> &ans_init, std::vector<uint64_t> &ans_out,
    const std::vector<phantom::arith::Modulus> &coeff_modulus,
    std::size_t poly_degree, std::size_t row_count, std::size_t col_count,
    std::size_t pt_row_stride, std::size_t pt_col_stride,
    std::size_t ct_stride, std::size_t ans_stride) {
    const std::size_t coeff_modulus_size = coeff_modulus.size();
    const std::size_t rns_coeff_count = poly_degree * coeff_modulus_size;
    ans_out = ans_init;

    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t coeff_idx = 0; coeff_idx < rns_coeff_count; ++coeff_idx) {
            const std::size_t mod_idx = coeff_idx / poly_degree;
            const uint64_t mod = coeff_modulus[mod_idx].value();

            const std::size_t ans_c0_idx = row * ans_stride + coeff_idx;
            const std::size_t ans_c1_idx = ans_c0_idx + rns_coeff_count;

            unsigned __int128 acc_c0 = ans_out[ans_c0_idx];
            unsigned __int128 acc_c1 = ans_out[ans_c1_idx];

            for (std::size_t col = 0; col < col_count; ++col) {
                const std::size_t pt_idx = row * pt_row_stride + col * pt_col_stride + coeff_idx;
                const std::size_t ct_c0_idx = col * ct_stride + coeff_idx;
                const std::size_t ct_c1_idx = ct_c0_idx + rns_coeff_count;

                const uint64_t pt_coeff = pt_matrix[pt_idx];
                acc_c0 += static_cast<unsigned __int128>(pt_coeff) * ct_terms[ct_c0_idx];
                acc_c1 += static_cast<unsigned __int128>(pt_coeff) * ct_terms[ct_c1_idx];
            }

            ans_out[ans_c0_idx] = static_cast<uint64_t>(acc_c0 % mod);
            ans_out[ans_c1_idx] = static_cast<uint64_t>(acc_c1 % mod);
        }
    }
}

PhantomCiphertext make_zero_ntt_cipher_2poly(
    const PhantomContext &context, std::size_t chain_index, std::size_t coeff_count,
    double scale = 1.0) {
    PhantomCiphertext ct;
    ct.resize(context, chain_index, 2, cudaStreamPerThread);
    ct.set_ntt_form(true);
    ct.set_scale(scale);
    ct.set_correction_factor(1ULL);
    ct.SetNoiseScaleDeg(1);

    check_cuda(
        cudaMemsetAsync(
            ct.data(), 0, 2 * coeff_count * sizeof(uint64_t), cudaStreamPerThread),
        "cudaMemsetAsync zero ciphertext");
    return ct;
}

TEST(MultiplyAdd2dFusionGTest, DenseLayoutMatchesReference) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);
    const auto &parms = context.get_context_data(chain).parms();
    const auto &coeff_modulus = parms.coeff_modulus();

    constexpr std::size_t kRows = 3;
    constexpr std::size_t kCols = 5;

    const std::size_t poly_degree = parms.poly_modulus_degree();
    const std::size_t coeff_modulus_size = coeff_modulus.size();
    const std::size_t rns_coeff_count = poly_degree * coeff_modulus_size;

    const std::size_t pt_col_stride = rns_coeff_count;
    const std::size_t pt_row_stride = kCols * pt_col_stride;
    const std::size_t ct_stride = 2 * rns_coeff_count;
    const std::size_t ans_stride = 2 * rns_coeff_count;

    const auto pt_matrix = random_rns_data(coeff_modulus, poly_degree, kRows * kCols, 3101);
    const auto ct_terms = random_rns_data(coeff_modulus, poly_degree, kCols * 2, 3201);
    const auto ans_init = random_rns_data(coeff_modulus, poly_degree, kRows * 2, 3301);

    std::vector<uint64_t> expected;
    reference_multiply_add_2d_fusion(
        pt_matrix, ct_terms, ans_init, expected, coeff_modulus, poly_degree,
        kRows, kCols, pt_row_stride, pt_col_stride, ct_stride, ans_stride);

    auto d_pt = phantom::util::make_cuda_auto_ptr<uint64_t>(pt_matrix.size(), cudaStreamPerThread);
    auto d_ct = phantom::util::make_cuda_auto_ptr<uint64_t>(ct_terms.size(), cudaStreamPerThread);
    auto d_ans = phantom::util::make_cuda_auto_ptr<uint64_t>(ans_init.size(), cudaStreamPerThread);

    copy_host_to_device(d_pt.get(), pt_matrix);
    copy_host_to_device(d_ct.get(), ct_terms);
    copy_host_to_device(d_ans.get(), ans_init);

    phantom::launch_multiply_add_2d_fusion(
        context, d_pt.get(), d_ct.get(), d_ans.get(),
        kRows, kCols, chain, 0, 0, 0, 0, cudaStreamPerThread);
    sync_stream();

    const auto actual = copy_device_to_host(d_ans.get(), ans_init.size());
    EXPECT_EQ(actual, expected);
}

TEST(MultiplyAdd2dFusionGTest, KernelMatchesCipherPlainLoopReference) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);
    const auto &parms = context.get_context_data(chain).parms();
    const auto &coeff_modulus = parms.coeff_modulus();

    constexpr std::size_t kRows = 3;
    constexpr std::size_t kCols = 4;

    const std::size_t poly_degree = parms.poly_modulus_degree();
    const std::size_t coeff_modulus_size = coeff_modulus.size();
    const std::size_t rns_coeff_count = poly_degree * coeff_modulus_size;
    const std::size_t cipher_coeff_count = 2 * rns_coeff_count;

    std::vector<PhantomCiphertext> ct_terms;
    ct_terms.reserve(kCols);
    for (std::size_t c = 0; c < kCols; ++c) {
        ct_terms.emplace_back(make_random_ntt_cipher(context, chain, 2, 5101 + c, 1.0));
    }

    std::vector<PhantomPlaintext> pt_matrix;
    pt_matrix.reserve(kRows * kCols);
    for (std::size_t r = 0; r < kRows; ++r) {
        for (std::size_t c = 0; c < kCols; ++c) {
            pt_matrix.emplace_back(make_random_ntt_plain(
                context, chain, 5201 + r * kCols + c, 1.0));
        }
    }

    std::vector<PhantomCiphertext> ans_cpu;
    ans_cpu.reserve(kRows);
    for (std::size_t r = 0; r < kRows; ++r) {
        ans_cpu.emplace_back(
            make_zero_ntt_cipher_2poly(context, chain, rns_coeff_count, 1.0));
    }
    sync_stream();

    for (std::size_t i = 0; i < kRows; ++i) {
        for (std::size_t j = 0; j < kCols; ++j) {
            phantom::multiply_plain_and_add_inplace(
                context, ct_terms[j], pt_matrix[i * kCols + j], ans_cpu[i]);
        }
    }
    sync_stream();

    std::vector<uint64_t> packed_pt(kRows * kCols * rns_coeff_count, 0ULL);
    for (std::size_t i = 0; i < kRows; ++i) {
        for (std::size_t j = 0; j < kCols; ++j) {
            const auto plain_host =
                copy_device_to_host(pt_matrix[i * kCols + j].data(), rns_coeff_count);
            const std::size_t dst_base = (i * kCols + j) * rns_coeff_count;
            std::copy_n(
                plain_host.begin(), rns_coeff_count, packed_pt.begin() + dst_base);
        }
    }

    std::vector<uint64_t> packed_ct(kCols * cipher_coeff_count, 0ULL);
    for (std::size_t j = 0; j < kCols; ++j) {
        const auto ct_host = copy_device_to_host(ct_terms[j].data(), cipher_coeff_count);
        const std::size_t dst_base = j * cipher_coeff_count;
        std::copy_n(ct_host.begin(), cipher_coeff_count, packed_ct.begin() + dst_base);
    }

    std::vector<uint64_t> ans_init(kRows * cipher_coeff_count, 0ULL);

    auto d_pt = phantom::util::make_cuda_auto_ptr<uint64_t>(packed_pt.size(), cudaStreamPerThread);
    auto d_ct = phantom::util::make_cuda_auto_ptr<uint64_t>(packed_ct.size(), cudaStreamPerThread);
    auto d_ans = phantom::util::make_cuda_auto_ptr<uint64_t>(ans_init.size(), cudaStreamPerThread);
    copy_host_to_device(d_pt.get(), packed_pt);
    copy_host_to_device(d_ct.get(), packed_ct);
    copy_host_to_device(d_ans.get(), ans_init);

    phantom::launch_multiply_add_2d_fusion(
        context, d_pt.get(), d_ct.get(), d_ans.get(),
        kRows, kCols, chain, 0, 0, 0, 0, cudaStreamPerThread);
    sync_stream();

    const auto kernel_ans = copy_device_to_host(d_ans.get(), ans_init.size());

    for (std::size_t i = 0; i < kRows; ++i) {
        const auto cpu_row = copy_device_to_host(ans_cpu[i].data(), cipher_coeff_count);
        const std::size_t row_base = i * cipher_coeff_count;
        std::vector<uint64_t> kernel_row(
            kernel_ans.begin() + row_base,
            kernel_ans.begin() + row_base + cipher_coeff_count);
        EXPECT_EQ(kernel_row, cpu_row);
    }
}

TEST(MultiplyAdd2dFusionGTest, DecryptedSlotsMatchExpectedAndReferenceLoop) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);
    const auto &parms = context.get_context_data(chain).parms();
    const std::size_t plain_modulus = parms.plain_modulus().value();

    PhantomSecretKey secret_key(context);
    PhantomBatchEncoder encoder(context);

    constexpr std::size_t kRows = 2;
    constexpr std::size_t kCols = 3;

    const std::size_t slot_count = encoder.slot_count();
    const std::size_t poly_degree = parms.poly_modulus_degree();
    const std::size_t coeff_modulus_size = parms.coeff_modulus().size();
    const std::size_t rns_coeff_count = poly_degree * coeff_modulus_size;
    const std::size_t cipher_coeff_count = 2 * rns_coeff_count;

    std::mt19937_64 rng(6101);
    std::uniform_int_distribution<uint64_t> dist(0ULL, plain_modulus - 1ULL);
    auto random_slot_vector = [&]() {
        std::vector<uint64_t> out(slot_count, 0ULL);
        for (std::size_t s = 0; s < slot_count; ++s) {
            out[s] = dist(rng);
        }
        return out;
    };

    std::vector<std::vector<uint64_t>> ct_slots(kCols);
    std::vector<PhantomCiphertext> ct_terms;
    ct_terms.reserve(kCols);
    for (std::size_t c = 0; c < kCols; ++c) {
        ct_slots[c] = random_slot_vector();
        PhantomPlaintext plain = encoder.encode(context, ct_slots[c]);
        PhantomCiphertext ct;
        secret_key.encrypt_symmetric(context, plain, ct);
        phantom::transform_to_ntt_inplace(context, ct);
        ct_terms.emplace_back(std::move(ct));
    }

    std::vector<std::vector<std::vector<uint64_t>>> pt_slots(
        kRows, std::vector<std::vector<uint64_t>>(kCols));
    std::vector<PhantomPlaintext> pt_matrix;
    pt_matrix.reserve(kRows * kCols);
    for (std::size_t r = 0; r < kRows; ++r) {
        for (std::size_t c = 0; c < kCols; ++c) {
            pt_slots[r][c] = random_slot_vector();
            PhantomPlaintext plain = encoder.encode(context, pt_slots[r][c]);
            phantom::transform_to_ntt_inplace(context, plain, chain);
            pt_matrix.emplace_back(std::move(plain));
        }
    }

    std::vector<PhantomCiphertext> ans_cpu;
    ans_cpu.reserve(kRows);
    for (std::size_t r = 0; r < kRows; ++r) {
        ans_cpu.emplace_back(
            make_zero_ntt_cipher_2poly(context, chain, rns_coeff_count, 1.0));
    }
    sync_stream();

    for (std::size_t i = 0; i < kRows; ++i) {
        for (std::size_t j = 0; j < kCols; ++j) {
            phantom::multiply_plain_and_add_inplace(
                context, ct_terms[j], pt_matrix[i * kCols + j], ans_cpu[i]);
        }
    }
    sync_stream();

    std::vector<uint64_t> packed_pt(kRows * kCols * rns_coeff_count, 0ULL);
    for (std::size_t i = 0; i < kRows; ++i) {
        for (std::size_t j = 0; j < kCols; ++j) {
            const auto plain_host =
                copy_device_to_host(pt_matrix[i * kCols + j].data(), rns_coeff_count);
            const std::size_t dst_base = (i * kCols + j) * rns_coeff_count;
            std::copy_n(plain_host.begin(), rns_coeff_count, packed_pt.begin() + dst_base);
        }
    }

    std::vector<uint64_t> packed_ct(kCols * cipher_coeff_count, 0ULL);
    for (std::size_t j = 0; j < kCols; ++j) {
        const auto ct_host = copy_device_to_host(ct_terms[j].data(), cipher_coeff_count);
        const std::size_t dst_base = j * cipher_coeff_count;
        std::copy_n(ct_host.begin(), cipher_coeff_count, packed_ct.begin() + dst_base);
    }

    std::vector<uint64_t> ans_init(kRows * cipher_coeff_count, 0ULL);
    auto d_pt = phantom::util::make_cuda_auto_ptr<uint64_t>(packed_pt.size(), cudaStreamPerThread);
    auto d_ct = phantom::util::make_cuda_auto_ptr<uint64_t>(packed_ct.size(), cudaStreamPerThread);
    auto d_ans = phantom::util::make_cuda_auto_ptr<uint64_t>(ans_init.size(), cudaStreamPerThread);
    copy_host_to_device(d_pt.get(), packed_pt);
    copy_host_to_device(d_ct.get(), packed_ct);
    copy_host_to_device(d_ans.get(), ans_init);

    phantom::launch_multiply_add_2d_fusion(
        context, d_pt.get(), d_ct.get(), d_ans.get(),
        kRows, kCols, chain, 0, 0, 0, 0, cudaStreamPerThread);
    sync_stream();

    const auto kernel_ans = copy_device_to_host(d_ans.get(), ans_init.size());

    for (std::size_t i = 0; i < kRows; ++i) {
        std::vector<uint64_t> expected(slot_count, 0ULL);
        for (std::size_t s = 0; s < slot_count; ++s) {
            uint64_t acc = 0ULL;
            for (std::size_t j = 0; j < kCols; ++j) {
                const uint64_t prod =
                    (ct_slots[j][s] * pt_slots[i][j][s]) % plain_modulus;
                acc += prod;
                if (acc >= plain_modulus) {
                    acc %= plain_modulus;
                }
            }
            expected[s] = acc % plain_modulus;
        }

        const auto cpu_row_data = copy_device_to_host(ans_cpu[i].data(), cipher_coeff_count);
        PhantomCiphertext cpu_ct =
            make_zero_ntt_cipher_2poly(context, chain, rns_coeff_count, 1.0);
        copy_host_to_device(cpu_ct.data(), cpu_row_data);
        phantom::transform_from_ntt_inplace(context, cpu_ct);
        PhantomPlaintext cpu_plain;
        secret_key.decrypt(context, cpu_ct, cpu_plain);
        const auto cpu_slots = encoder.decode(context, cpu_plain);

        const std::size_t row_base = i * cipher_coeff_count;
        std::vector<uint64_t> kernel_row_data(
            kernel_ans.begin() + row_base,
            kernel_ans.begin() + row_base + cipher_coeff_count);
        PhantomCiphertext kernel_ct =
            make_zero_ntt_cipher_2poly(context, chain, rns_coeff_count, 1.0);
        copy_host_to_device(kernel_ct.data(), kernel_row_data);
        phantom::transform_from_ntt_inplace(context, kernel_ct);
        PhantomPlaintext kernel_plain;
        secret_key.decrypt(context, kernel_ct, kernel_plain);
        const auto kernel_slots = encoder.decode(context, kernel_plain);

        EXPECT_EQ(cpu_slots, expected);
        EXPECT_EQ(kernel_slots, expected);
        EXPECT_EQ(kernel_slots, cpu_slots);
    }
}

TEST(MultiplyAdd2dFusionGTest, StridedLayoutMatchesReference) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);
    const auto &parms = context.get_context_data(chain).parms();
    const auto &coeff_modulus = parms.coeff_modulus();

    constexpr std::size_t kRows = 2;
    constexpr std::size_t kCols = 4;

    const std::size_t poly_degree = parms.poly_modulus_degree();
    const std::size_t coeff_modulus_size = coeff_modulus.size();
    const std::size_t rns_coeff_count = poly_degree * coeff_modulus_size;

    const std::size_t pt_col_stride = rns_coeff_count + 7;
    const std::size_t pt_row_stride = kCols * pt_col_stride + 13;
    const std::size_t ct_stride = 2 * rns_coeff_count + 11;
    const std::size_t ans_stride = 2 * rns_coeff_count + 17;

    std::vector<uint64_t> pt_matrix(kRows * pt_row_stride, 0ULL);
    std::vector<uint64_t> ct_terms(kCols * ct_stride, 0ULL);
    std::vector<uint64_t> ans_init(kRows * ans_stride, 0ULL);

    const auto pt_active = random_rns_data(coeff_modulus, poly_degree, kRows * kCols, 4101);
    const auto ct_active = random_rns_data(coeff_modulus, poly_degree, kCols * 2, 4201);
    const auto ans_active = random_rns_data(coeff_modulus, poly_degree, kRows * 2, 4301);

    for (std::size_t r = 0; r < kRows; ++r) {
        for (std::size_t c = 0; c < kCols; ++c) {
            const std::size_t src_base = (r * kCols + c) * rns_coeff_count;
            const std::size_t dst_base = r * pt_row_stride + c * pt_col_stride;
            std::copy_n(pt_active.begin() + src_base, rns_coeff_count, pt_matrix.begin() + dst_base);
        }
    }

    for (std::size_t c = 0; c < kCols; ++c) {
        const std::size_t dst_base = c * ct_stride;
        const std::size_t src_c0 = (c * 2) * rns_coeff_count;
        const std::size_t src_c1 = src_c0 + rns_coeff_count;
        std::copy_n(ct_active.begin() + src_c0, rns_coeff_count, ct_terms.begin() + dst_base);
        std::copy_n(ct_active.begin() + src_c1, rns_coeff_count, ct_terms.begin() + dst_base + rns_coeff_count);
    }

    for (std::size_t r = 0; r < kRows; ++r) {
        const std::size_t dst_base = r * ans_stride;
        const std::size_t src_c0 = (r * 2) * rns_coeff_count;
        const std::size_t src_c1 = src_c0 + rns_coeff_count;
        std::copy_n(ans_active.begin() + src_c0, rns_coeff_count, ans_init.begin() + dst_base);
        std::copy_n(ans_active.begin() + src_c1, rns_coeff_count, ans_init.begin() + dst_base + rns_coeff_count);
    }

    std::vector<uint64_t> expected;
    reference_multiply_add_2d_fusion(
        pt_matrix, ct_terms, ans_init, expected, coeff_modulus, poly_degree,
        kRows, kCols, pt_row_stride, pt_col_stride, ct_stride, ans_stride);

    auto d_pt = phantom::util::make_cuda_auto_ptr<uint64_t>(pt_matrix.size(), cudaStreamPerThread);
    auto d_ct = phantom::util::make_cuda_auto_ptr<uint64_t>(ct_terms.size(), cudaStreamPerThread);
    auto d_ans = phantom::util::make_cuda_auto_ptr<uint64_t>(ans_init.size(), cudaStreamPerThread);

    copy_host_to_device(d_pt.get(), pt_matrix);
    copy_host_to_device(d_ct.get(), ct_terms);
    copy_host_to_device(d_ans.get(), ans_init);

    phantom::launch_multiply_add_2d_fusion(
        context, d_pt.get(), d_ct.get(), d_ans.get(),
        kRows, kCols, chain,
        pt_row_stride, pt_col_stride, ct_stride, ans_stride, cudaStreamPerThread);
    sync_stream();

    const auto actual = copy_device_to_host(d_ans.get(), ans_init.size());
    EXPECT_EQ(actual, expected);
}

} // namespace
