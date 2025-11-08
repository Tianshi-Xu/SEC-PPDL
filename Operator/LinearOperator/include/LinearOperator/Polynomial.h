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
} // namespace LinearOperator