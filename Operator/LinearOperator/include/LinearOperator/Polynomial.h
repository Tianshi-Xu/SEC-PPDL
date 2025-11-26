#pragma once
#include <HE/HE.h>
#include <seal/util/common.h>
#include <complex>
#include <cstddef>
#include <vector>

namespace LinearOperator {
    Tensor<uint64_t> ElementWiseMul(Tensor<uint64_t> &x, Tensor<uint64_t> &y, HE::HEEvaluator* HE);

    using Complex128 = std::complex<int128_t>;

    std::vector<Complex128> CKKSInverseFFT(std::vector<Complex128> values, std::size_t degree, int fft_scale);
    std::vector<Complex128> CKKSForwardFFT(std::vector<Complex128> values, std::size_t degree, int fft_scale);
    Tensor<int128_t> CKKSDecode(const Tensor<Complex128> &values, std::size_t degree, int fft_scale);
    Tensor<int128_t> CKKSEncode(const Tensor<int128_t> &slots, std::size_t slot_count, int fft_scale);
} // namespace LinearOperator