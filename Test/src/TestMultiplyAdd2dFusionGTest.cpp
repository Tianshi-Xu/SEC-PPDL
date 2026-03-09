#include "PhantomBatchTestUtils.h"

#include "evaluate.cuh"

#include <cstdint>
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
