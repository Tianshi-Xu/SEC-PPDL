#include <LinearOperator/Polynomial.h>
#include <LinearOperator/Conversion.h>
#include <seal/ckks.h>
#include <seal/memorymanager.h>
#include <seal/util/common.h>
#include <seal/util/numth.h>
#include <cmath>
#include <stdexcept>

namespace LinearOperator {

// input and output are both secret shares, also supports square when x==y, the input can be any shape
// TODO: support x.size() % HE->polyModulusDegree != 0
Tensor<uint64_t> ElementWiseMul(Tensor<uint64_t> &x, Tensor<uint64_t> &y, HE::HEEvaluator* HE){
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y must have the same size in ElementWiseMul");
    }
    auto shape = x.shape();
    x.reshape({x.size()/HE->polyModulusDegree, HE->polyModulusDegree});
    Tensor<HE::unified::UnifiedCiphertext> x_ct = Operator::SSToHE(x, HE);
    Tensor<HE::unified::UnifiedCiphertext> z(x_ct.shape(), HE->GenerateZeroCiphertext(HE->Backend()));
    x.reshape(shape);
    if(&x==&y){
        // cout << "x==y" << endl;
        for(size_t i = 0; i < x_ct.size(); i++){
            HE->evaluator->square(x_ct(i), z(i));
            // z(i) = x_ct(i);
        }
    }else{
        y.reshape({y.size()/HE->polyModulusDegree, HE->polyModulusDegree});
        Tensor<HE::unified::UnifiedCiphertext> y_ct = Operator::SSToHE(y, HE);
        for(size_t i = 0; i < x_ct.size(); i++){
            HE->evaluator->multiply(x_ct(i), y_ct(i), z(i));
        }
        y.reshape(shape);
    }
    Tensor<uint64_t> z_ss = Operator::HEToSS(z, HE);
    z_ss.reshape(shape);
    return z_ss;
}

namespace {
template <typename Scalar>
Scalar ScaleToFixed(long double value, int scale_bits) {
    const long double scaled = std::ldexp(value, scale_bits);
    const long double rounded = (scaled >= 0.0L) ? std::floor(scaled + 0.5L) : std::ceil(scaled - 0.5L);
    return static_cast<Scalar>(rounded);
}

std::vector<std::size_t> BuildMatrixRepsIndexMap(std::size_t degree) {
    if (degree == 0) {
        throw std::invalid_argument("degree must be greater than zero in BuildMatrixRepsIndexMap");
    }
    if ((degree & (degree - 1)) != 0) {
        throw std::invalid_argument("degree must be a power of two in BuildMatrixRepsIndexMap");
    }

    const std::size_t slots = degree >> 1;
    std::vector<std::size_t> matrix_map(degree, 0);
    const std::size_t logn = seal::util::get_power_of_two(degree);
    const std::uint64_t m = static_cast<std::uint64_t>(degree) << 1;
    const std::uint64_t gen = 3;
    std::uint64_t pos = 1;

    for (std::size_t i = 0; i < slots; ++i) {
        const std::uint64_t index1 = (pos - 1) >> 1;
        const std::uint64_t index2 = (m - pos - 1) >> 1;

        matrix_map[i] = seal::util::safe_cast<std::size_t>(seal::util::reverse_bits(index1, logn));
        matrix_map[slots | i] = seal::util::safe_cast<std::size_t>(seal::util::reverse_bits(index2, logn));

        pos *= gen;
        pos &= (m - 1);
    }

    return matrix_map;
}
} // namespace

std::vector<Complex128> CKKSInverseFFT(std::vector<Complex128> values, std::size_t degree, int fft_scale) {
    using Scalar128 = int128_t;
    using Arithmetic128 = seal::util::Arithmetic<Complex128, Complex128, Scalar128>;
    using FFTHandler128 = seal::util::DWTHandler<Complex128, Complex128, Scalar128>;

    if (degree == 0) {
        throw std::invalid_argument("degree must be greater than zero in CKKSInverseFFT");
    }
    if (values.size() != degree) {
        throw std::invalid_argument("values size must match degree in CKKSInverseFFT");
    }
    if ((degree & (degree - 1)) != 0) {
        throw std::invalid_argument("degree must be a power of two in CKKSInverseFFT");
    }
    if (degree <= 1) {
        return values;
    }

    Arithmetic128 arithmetic(fft_scale);
    FFTHandler128 handler(arithmetic);

    std::vector<Complex128> inv_root_powers_2n_scaled(degree, Complex128(0, 0));
    const int logn = static_cast<int>(seal::util::get_power_of_two(degree));
    seal::util::ComplexRoots complex_roots(static_cast<std::size_t>(degree) << 1, seal::MemoryManager::GetPool());

    for (std::size_t i = 1; i < degree; ++i) {
        const auto reversed_index = seal::util::reverse_bits(i - 1, logn) + 1;
        const auto inv_root = std::conj(complex_roots.get_root(reversed_index));
        inv_root_powers_2n_scaled[i] = Complex128(
            ScaleToFixed<Scalar128>(static_cast<long double>(inv_root.real()), fft_scale),
            ScaleToFixed<Scalar128>(static_cast<long double>(inv_root.imag()), fft_scale));
    }

    const Scalar128 fix = ScaleToFixed<Scalar128>(1.0L / static_cast<long double>(degree), fft_scale);
    handler.transform_from_rev(values.data(), logn, inv_root_powers_2n_scaled.data(), &fix);

    return values;
}

std::vector<Complex128> CKKSForwardFFT(std::vector<Complex128> values, std::size_t degree, int fft_scale) {
    using Scalar128 = int128_t;
    using Arithmetic128 = seal::util::Arithmetic<Complex128, Complex128, Scalar128>;
    using FFTHandler128 = seal::util::DWTHandler<Complex128, Complex128, Scalar128>;

    if (degree == 0) {
        throw std::invalid_argument("degree must be greater than zero in CKKSForwardFFT");
    }
    if (values.size() != degree) {
        throw std::invalid_argument("values size must match degree in CKKSForwardFFT");
    }
    if ((degree & (degree - 1)) != 0) {
        throw std::invalid_argument("degree must be a power of two in CKKSForwardFFT");
    }
    if (degree <= 1) {
        return values;
    }

    Arithmetic128 arithmetic(fft_scale);
    FFTHandler128 handler(arithmetic);

    std::vector<Complex128> root_powers_2n_scaled(degree, Complex128(0, 0));
    const int logn = static_cast<int>(seal::util::get_power_of_two(degree));
    seal::util::ComplexRoots complex_roots(static_cast<std::size_t>(degree) << 1, seal::MemoryManager::GetPool());

    for (std::size_t i = 1; i < degree; ++i) {
        const auto reversed_index = seal::util::reverse_bits(i, logn);
        const auto root = complex_roots.get_root(reversed_index);
        root_powers_2n_scaled[i] = Complex128(
            ScaleToFixed<Scalar128>(static_cast<long double>(root.real()), fft_scale),
            ScaleToFixed<Scalar128>(static_cast<long double>(root.imag()), fft_scale));
    }

    handler.transform_to_rev(values.data(), logn, root_powers_2n_scaled.data(), nullptr);

    return values;
}

Tensor<int128_t> CKKSDecode(const Tensor<Complex128> &values, std::size_t degree, int fft_scale) {
    if (degree == 0) {
        throw std::invalid_argument("degree must be greater than zero in CKKSDecode");
    }
    if ((degree & (degree - 1)) != 0) {
        throw std::invalid_argument("degree must be a power of two in CKKSDecode");
    }
    if (values.size() != degree) {
        throw std::invalid_argument("values size must match degree in CKKSDecode");
    }

    std::vector<Complex128> frequency(values.data().begin(), values.data().end());
    frequency = CKKSForwardFFT(std::move(frequency), degree, fft_scale);
    const auto matrix_map = BuildMatrixRepsIndexMap(degree);

    Tensor<int128_t> decoded({degree >> 1});
    auto &decoded_data = decoded.data();
    for (std::size_t i = 0; i < (degree >> 1); ++i) {
        decoded_data[i] = frequency[matrix_map[i]].real();
    }

    return decoded;
}

Tensor<int128_t> CKKSEncode(const Tensor<int128_t> &slots, std::size_t slot_count, int fft_scale) {
    if (slot_count == 0) {
        throw std::invalid_argument("slot_count must be greater than zero in CKKSEncode");
    }
    if (slot_count != slots.size()) {
        throw std::invalid_argument("slot tensor size must match slot_count in CKKSEncode");
    }

    const std::size_t degree = slot_count << 1;
    if ((degree & (degree - 1)) != 0) {
        throw std::invalid_argument("twice slot_count must be a power of two in CKKSEncode");
    }

    auto matrix_map = BuildMatrixRepsIndexMap(degree);

    std::vector<Complex128> freq(degree, Complex128(0, 0));
    const auto &slot_data = slots.data();
    for (std::size_t i = 0; i < slot_count; ++i) {
        const Complex128 value(slot_data[i], 0);
        freq[matrix_map[i]] = value;
        freq[matrix_map[i + slot_count]] = std::conj(value);
    }

    auto time_domain = CKKSInverseFFT(std::move(freq), degree, fft_scale);

    Tensor<int128_t> encoded({degree});
    auto &encoded_data = encoded.data();
    for (std::size_t i = 0; i < degree; ++i) {
        encoded_data[i] = time_domain[i].real();
    }

    return encoded;
}

} // namespace LinearOperator