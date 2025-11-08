#include <LinearOperator/Polynomial.h>
#include <LinearOperator/Conversion.h>
#include <seal/ckks.h>
#include <seal/memorymanager.h>
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


} // namespace LinearOperator