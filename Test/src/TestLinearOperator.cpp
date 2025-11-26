#include <LinearOperator/Polynomial.h>
#include <Utils/ArgMapping/ArgMapping.h>
#include <seal/util/common.h>
#include <seal/util/numth.h>
#include <iostream>
#include <stdexcept>

int party, port = 32000;
int num_threads = 2;
std::string address = "127.0.0.1";

Utils::NetIO* netio;
HE::HEEvaluator* he;
using namespace std;
using namespace LinearOperator;
using namespace HE;

namespace {
std::vector<std::size_t> BuildMatrixMapForTest(std::size_t degree) {
    if (degree == 0) {
        throw std::invalid_argument("degree must be greater than zero");
    }
    if ((degree & (degree - 1)) != 0) {
        throw std::invalid_argument("degree must be a power of two");
    }
    const std::size_t slots = degree >> 1;
    std::vector<std::size_t> map(degree, 0);
    const std::size_t logn = seal::util::get_power_of_two(degree);
    const std::uint64_t m = static_cast<std::uint64_t>(degree) << 1;
    const std::uint64_t gen = 3;
    std::uint64_t pos = 1;
    for (std::size_t i = 0; i < slots; ++i) {
        const std::uint64_t index1 = (pos - 1) >> 1;
        const std::uint64_t index2 = (m - pos - 1) >> 1;
        map[i] = seal::util::safe_cast<std::size_t>(seal::util::reverse_bits(index1, logn));
        map[slots | i] = seal::util::safe_cast<std::size_t>(seal::util::reverse_bits(index2, logn));
        pos *= gen;
        pos &= (m - 1);
    }
    return map;
}

void test_poly(HE::HEEvaluator* he){;
    Tensor<uint64_t> x({8192});
    Tensor<uint64_t> y({8192});
    Tensor<uint64_t> z({8192});
    if(party == ALICE){
        for(uint32_t i = 0; i < 8192; i++){
            x(i) = i;
            y(i) = i;
        }
    }
    x.print(10);
    y.print(10);
    z = LinearOperator::ElementWiseMul(x, x, he);
    if (party == ALICE){
        netio->send_tensor(z);
    }else{
        Tensor<uint64_t> z0({8192});
        netio->recv_tensor(z0);
        auto z_result = z + z0;
        for(uint32_t i = 0; i < 8192; i++){
            z_result(i) = z_result(i) % he->plain_mod;
        }
        z_result.print(10);
    }
}

std::string Int128ToString(int128_t value) {
    if (value == 0) {
        return "0";
    }
    bool negative = value < 0;
    int128_t abs_value = negative ? -value : value;
    std::string digits;
    while (abs_value > 0) {
        auto remainder = static_cast<int32_t>(abs_value % 10);
        digits.push_back(static_cast<char>('0' + remainder));
        abs_value /= 10;
    }
    if (negative) {
        digits.push_back('-');
    }
    std::reverse(digits.begin(), digits.end());
    return digits;
}

void test_ckks_inverse_fft() {
    const std::size_t degree = 8;
    const int fft_scale = 40;              // keep transform scaling moderate to avoid overflow
    const int extra_shift = 32;            // boosts payload magnitude while staying within int128 range
    const int128_t scale = static_cast<int128_t>(int64_t{1}) << fft_scale;
    const int128_t payload_shift = static_cast<int128_t>(1) << extra_shift;
    const int128_t base = scale * payload_shift; // stored fixed-point unit for real value 1 << extra_shift

    std::vector<LinearOperator::Complex128> inputs(degree);
    std::cout << "CKKSInverseFFT inputs (fixed-point, scale=2^" << fft_scale
              << ", payload shift=2^" << extra_shift << "): ";
    for (std::size_t i = 0; i < degree; ++i) {
        int128_t magnitude = static_cast<int128_t>(static_cast<long long>(i + 1)) * base;
        inputs[i] = LinearOperator::Complex128(magnitude, static_cast<int128_t>(0));
        std::cout << Int128ToString(inputs[i].real());
        if (i + 1 != degree) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    auto outputs = LinearOperator::CKKSInverseFFT(inputs, degree, fft_scale);

    std::cout << "CKKSInverseFFT outputs (real parts): ";
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        std::cout << Int128ToString(outputs[i].real());
        if (i + 1 != outputs.size()) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    const int128_t numerator = static_cast<int128_t>(degree + 1);
    const int128_t denominator = static_cast<int128_t>(2);
    const int128_t average_scaled = (numerator * base) / denominator;
    std::cout << "Expected first coefficient (average * base): "
              << Int128ToString(average_scaled) << std::endl;
}

void test_ckks_fft_roundtrip() {
    const std::size_t degree = 8;
    const int fft_scale = 40;
    const int extra_shift = 32;
    const int128_t scale = static_cast<int128_t>(1) << fft_scale;
    const int128_t payload_shift = static_cast<int128_t>(1) << extra_shift;
    const int128_t base = scale * payload_shift;

    std::vector<LinearOperator::Complex128> original(degree);
    std::cout << "\n=== CKKS FFT Roundtrip Test ===" << std::endl;
    std::cout << "Original inputs (fixed-point, scale=2^" << fft_scale
              << ", payload shift=2^" << extra_shift << "): ";
    for (std::size_t i = 0; i < degree; ++i) {
        int128_t magnitude = static_cast<int128_t>(static_cast<long long>(i + 1)) * base;
        original[i] = LinearOperator::Complex128(magnitude, static_cast<int128_t>(0));
        std::cout << Int128ToString(original[i].real());
        if (i + 1 != degree) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    auto after_ifft = LinearOperator::CKKSInverseFFT(original, degree, fft_scale);
    std::cout << "After IFFT (real parts): ";
    for (std::size_t i = 0; i < after_ifft.size(); ++i) {
        std::cout << Int128ToString(after_ifft[i].real());
        if (i + 1 != after_ifft.size()) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    auto after_fft = LinearOperator::CKKSForwardFFT(after_ifft, degree, fft_scale);
    std::cout << "After FFT (real parts): ";
    for (std::size_t i = 0; i < after_fft.size(); ++i) {
        std::cout << Int128ToString(after_fft[i].real());
        if (i + 1 != after_fft.size()) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    std::cout << "Comparison (original vs roundtrip):" << std::endl;
    bool all_match = true;
    const double relative_tolerance = 0.01;
    for (std::size_t i = 0; i < degree; ++i) {
        int128_t original_val = original[i].real();
        int128_t roundtrip_val = after_fft[i].real();
        int128_t diff = original_val - roundtrip_val;
        if (diff < 0) diff = -diff;
        
        int128_t abs_original = original_val < 0 ? -original_val : original_val;
        double relative_error = abs_original > 0 
            ? static_cast<double>(diff) / static_cast<double>(abs_original)
            : (diff == 0 ? 0.0 : 1.0);
        
        bool match = relative_error <= relative_tolerance;
        if (!match) all_match = false;
        
        std::cout << "  [" << i << "] diff=" << Int128ToString(diff)
                  << ", relative_error=" << std::scientific << relative_error << std::fixed
                  << " (" << (relative_error * 100.0) << "%)"
                  << (match ? " ✓" : " ✗") << std::endl;
    }
    std::cout << (all_match ? "✓ Roundtrip test PASSED" : "✗ Roundtrip test FAILED") << std::endl;
}

void test_ckks_tensor_encode_decode() {
    const std::size_t slot_count = 4;
    const std::size_t degree = slot_count << 1;
    const int fft_scale = 40;
    const int extra_shift = 28;
    const int128_t scale = static_cast<int128_t>(1) << fft_scale;
    const int128_t payload_shift = static_cast<int128_t>(1) << extra_shift;
    const int128_t base = scale * payload_shift;
    const auto matrix_map = BuildMatrixMapForTest(degree);

    Tensor<int128_t> slots({slot_count});
    std::cout << "\n=== CKKS Encode/Decode Tensor Test ===" << std::endl;
    std::cout << "Original slot values: ";
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots(i) = static_cast<int128_t>(static_cast<long long>(i + 1)) * base;
        std::cout << Int128ToString(slots(i));
        if (i + 1 != slot_count) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    std::vector<LinearOperator::Complex128> frequency(degree, LinearOperator::Complex128(0, 0));
    for (std::size_t i = 0; i < slot_count; ++i) {
        const LinearOperator::Complex128 value(slots(i), static_cast<int128_t>(0));
        frequency[matrix_map[i]] = value;
        frequency[matrix_map[i + slot_count]] = std::conj(value);
    }

    auto time_domain = LinearOperator::CKKSInverseFFT(frequency, degree, fft_scale);
    Tensor<LinearOperator::Complex128> time_tensor({degree});
    for (std::size_t i = 0; i < degree; ++i) {
        time_tensor(i) = time_domain[i];
    }

    const int128_t imag_tolerance = static_cast<int128_t>(1) << 28;
    auto abs128 = [](int128_t v) { return v < 0 ? -v : v; };
    bool time_imag_match = true;
    std::cout << "Time-domain imaginary parts:" << std::endl;
    for (std::size_t i = 0; i < degree; ++i) {
        const int128_t imag_abs = abs128(time_domain[i].imag());
        if (imag_abs > imag_tolerance) {
            time_imag_match = false;
        }
        std::cout << "  time_imag[" << i << "]=" << Int128ToString(imag_abs)
                  << (imag_abs <= imag_tolerance ? " \u2713" : " \u2717") << std::endl;
    }

    auto recon_frequency = LinearOperator::CKKSForwardFFT(time_domain, degree, fft_scale);
    const long double base_ld = static_cast<long double>(base);
    bool freq_imag_match = true;
    std::cout << "Frequency-domain imaginary parts after decode path:" << std::endl;
    for (std::size_t i = 0; i < degree; ++i) {
        const int128_t imag_abs = abs128(recon_frequency[i].imag());
        long double rel = base_ld != 0.0L ? static_cast<long double>(imag_abs) / base_ld : 0.0L;
        if (imag_abs > imag_tolerance) {
            freq_imag_match = false;
        }
        std::cout << "  freq_imag[" << i << "]=" << Int128ToString(imag_abs)
                  << ", rel=" << std::scientific << rel << std::defaultfloat
                  << (imag_abs <= imag_tolerance ? " \u2713" : " \u2717") << std::endl;
    }

    auto decoded = LinearOperator::CKKSDecode(time_tensor, degree, fft_scale);
    std::cout << "Decoded slot values: ";
    for (std::size_t i = 0; i < decoded.size(); ++i) {
        std::cout << Int128ToString(decoded(i));
        if (i + 1 != decoded.size()) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    const int128_t tolerance = static_cast<int128_t>(1) << 32;
    bool decode_match = decoded.size() == slot_count;
    for (std::size_t i = 0; i < slot_count && decode_match; ++i) {
        int128_t diff = slots(i) - decoded(i);
        if (diff < 0) {
            diff = -diff;
        }
        if (diff > tolerance) {
            decode_match = false;
        }
        long double rel = slots(i) ? static_cast<long double>(diff) / static_cast<long double>(slots(i)) : 0.0L;
        std::cout << "  decode[" << i << "] diff=" << Int128ToString(diff)
              << ", rel=" << std::scientific << rel << std::defaultfloat
              << (diff <= tolerance ? " \u2713" : " \u2717") << std::endl;
    }

    auto encoded = LinearOperator::CKKSEncode(slots, slot_count, fft_scale);
    std::cout << "Encoded tensor length: " << encoded.size() << " (expected " << degree << ")" << std::endl;

    bool encode_match = encoded.size() == degree;
    for (std::size_t i = 0; i < degree && encode_match; ++i) {
        int128_t diff = encoded(i) - time_tensor(i).real();
        if (diff < 0) {
            diff = -diff;
        }
        if (diff > tolerance) {
            encode_match = false;
        }
        std::cout << "  encode[" << i << "] diff=" << Int128ToString(diff)
              << (diff <= tolerance ? " \u2713" : " \u2717") << std::endl;
    }

    if (time_imag_match && freq_imag_match && decode_match && encode_match) {
        std::cout << "\u2713 Encode/Decode tensor test PASSED" << std::endl;
    } else {
        std::cout << "\u2717 Encode/Decode tensor test FAILED" << std::endl;
    }
}
} // namespace

int main(int argc, char **argv){
    // ArgMapping amap;
    // amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2"); // 1 is server, 2 is client
    // amap.arg("p", port, "Port Number");
    // amap.arg("ip", address, "IP Address of server (ALICE)");
    // amap.parse(argc, argv);
    
    // netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
    // std::cout << "netio generated" << std::endl;
    // he = new HE::HEEvaluator(netio, party, 8192,32,Datatype::HOST);
    // he->GenerateNewKey();
    
    // test_poly(he);
    // test_ckks_inverse_fft();
    // test_ckks_fft_roundtrip();
    test_ckks_tensor_encode_decode();

    return 0;
}