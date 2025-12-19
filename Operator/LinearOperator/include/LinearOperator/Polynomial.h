#pragma once
#include <HE/HE.h>
#include <seal/util/common.h>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace LinearOperator {
    Tensor<uint64_t> ElementWiseMulUnsigned(Tensor<uint64_t> &x, Tensor<uint64_t> &y, HE::HEEvaluator* HE);
    Tensor<int64_t> ElementWiseMulSigned(Tensor<int64_t> &x, Tensor<int64_t> &y, HE::HEEvaluator* HE);

    namespace detail {
        template <typename T>
        struct DependentFalse : std::false_type {};
    }

    template <typename T>
    Tensor<T> ElementWiseMul(Tensor<T> &x, Tensor<T> &y, HE::HEEvaluator* HE) {
        if constexpr (std::is_same_v<T, uint64_t>) {
            return ElementWiseMulUnsigned(x, y, HE);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return ElementWiseMulSigned(x, y, HE);
        } else {
            static_assert(detail::DependentFalse<T>::value, "ElementWiseMul only supports uint64_t or int64_t tensors");
        }
    }

    using Complex128 = std::complex<int128_t>;

    std::vector<Complex128> CKKSInverseFFT(std::vector<Complex128> values, std::size_t degree, int fft_scale);
    std::vector<Complex128> CKKSForwardFFT(std::vector<Complex128> values, std::size_t degree, int fft_scale);
    Tensor<int128_t> CKKSDecode(const Tensor<Complex128> &values, std::size_t degree, int fft_scale);
    Tensor<int128_t> CKKSEncode(const Tensor<int128_t> &slots, std::size_t slot_count, int fft_scale);
} // namespace LinearOperator