#include <LinearOperator/Polynomial.h>
#include <LinearOperator/Conversion.h>
#include <Utils/ArgMapping/ArgMapping.h>
#include <seal/util/common.h>
#include <seal/util/numth.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int party, port = 32000;
int num_threads = 1;
std::string address = "127.0.0.1";

Utils::NetIO* netio;
HE::HEEvaluator* he;
using namespace std;
using namespace LinearOperator;
using namespace HE;

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

void test_poly(HE::HEEvaluator* he){
    const size_t slot_count = he->polyModulusDegree;
    constexpr double kMin = -5.0;
    constexpr double kMax = 5.0;
    constexpr int kScaleBits = 12;

    auto lerp = [&](size_t idx, size_t span) -> double {
        if (span <= 1) {
            return kMin;
        }
        double t = static_cast<double>(idx % span) / static_cast<double>(span - 1);
        return kMin + (kMax - kMin) * t;
    };
    auto primary_value = [&](size_t idx) -> double {
        return lerp(idx, slot_count);
    };
    auto secondary_value = [&](size_t idx) -> double {
        const size_t permuted = (idx * 7 + 3) % slot_count;
        return lerp(permuted, slot_count);
    };
    auto wrap_to_plain = [&]( __int128 value) -> uint64_t {
        const __int128 modulus = static_cast<__int128>(he->plain_mod);
        __int128 residue = value % modulus;
        if (residue < 0) {
            residue += modulus;
        }
        return static_cast<uint64_t>(residue);
    };
    auto encode_fixed = [&](double value) -> int64_t {
        const long double scaled = static_cast<long double>(value) * std::ldexp(1.0L, kScaleBits);
        return static_cast<int64_t>(std::llround(scaled));
    };

    auto expected_residue = [&](size_t idx, bool square_case) -> uint64_t {
        const int64_t xv = encode_fixed(primary_value(idx));
        const int64_t yv = square_case ? xv : encode_fixed(secondary_value(idx));
        return wrap_to_plain(static_cast<__int128>(xv) * static_cast<__int128>(yv));
    };

    Tensor<int64_t> x({slot_count}, static_cast<int32_t>(sizeof(int64_t) * 8), kScaleBits);
    Tensor<int64_t> y({slot_count}, static_cast<int32_t>(sizeof(int64_t) * 8), kScaleBits);
    if (party == ALICE) {
        Tensor<double> x_plain({slot_count});
        Tensor<double> y_plain({slot_count});
        for (size_t i = 0; i < slot_count; ++i) {
            x_plain(i) = primary_value(i);
            y_plain(i) = secondary_value(i);
        }
        x = Tensor<int64_t>::FromFloatTensorToFixed(x_plain, kScaleBits);
        y = Tensor<int64_t>::FromFloatTensorToFixed(y_plain, kScaleBits);
    }

    auto run_case = [&](const char* label, bool square_case) {
        Tensor<int64_t> lhs = x;
        Tensor<int64_t> rhs = y;
        Tensor<int64_t> result = square_case
            ? LinearOperator::ElementWiseMul(lhs, lhs, he)
            : LinearOperator::ElementWiseMul(lhs, rhs, he);

        if (party == ALICE) {
            netio->send_tensor(result);
        } else {
            Tensor<int64_t> peer_share(result.shape());
            netio->recv_tensor(peer_share);

            size_t mismatches = 0;
            const size_t sample = 8;
            std::vector<size_t> mismatch_indices;
            mismatch_indices.reserve(sample);

            for (size_t i = 0; i < slot_count; ++i) {
                const uint64_t combined = wrap_to_plain(static_cast<__int128>(result(i)) + peer_share(i));
                const uint64_t expected = expected_residue(i, square_case);
                if (combined != expected) {
                    if (mismatch_indices.size() < sample) {
                        mismatch_indices.push_back(i);
                    }
                    ++mismatches;
                }
                if (i < sample) {
                    std::cout << "[test_poly - " << label << "] slot " << i
                              << ": recon=" << combined << ", expect=" << expected << std::endl;
                }
            }

            if (mismatches == 0) {
                std::cout << "[test_poly - " << label << "] PASS: all " << slot_count
                          << " slots match signed inputs in [-5,5]" << std::endl;
            } else {
                std::cout << "[test_poly - " << label << "] FAIL: " << mismatches
                          << " mismatches detected" << std::endl;
                for (size_t idx : mismatch_indices) {
                    const uint64_t combined = wrap_to_plain(static_cast<__int128>(result(idx)) + peer_share(idx));
                    const uint64_t expected = expected_residue(idx, square_case);
                    std::cout << "  slot " << idx << ": recon=" << combined
                              << ", expect=" << expected << std::endl;
                }
            }
        }
    };

    run_case("x^2", true);
    run_case("x*y", false);
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

void test_sstohe_conversion(HE::HEEvaluator* he) {
    const size_t degree = he->polyModulusDegree;
    Tensor<uint64_t> share({1, degree});

    auto share_value = [plain = he->plain_mod](bool is_server, size_t idx) -> uint64_t {
        const uint64_t base = static_cast<uint64_t>(idx % plain);
        return is_server ? (base + 1) % plain : (2 * base + 3) % plain;
    };

    for (size_t j = 0; j < degree; ++j) {
        share(j) = share_value(he->server, j);
    }

    auto ct = Operator::SSToHE(share, he);

    if (he->server) {
        he->SendEncVec(ct);
        std::cout << "[SSToHE] server sent ciphertext for verification" << std::endl;
    } else {
        Tensor<HE::unified::UnifiedCiphertext> combined(ct.shape(), Datatype::HOST);
        he->ReceiveEncVec(combined);

        Plaintext pt;
        he->decryptor->decrypt(combined(0), pt);

        std::vector<uint64_t> decoded(degree);
        he->encoder->decode(pt, decoded);

        const size_t sample = 8;
        size_t mismatches = 0;
        for (size_t j = 0; j < degree; ++j) {
            const uint64_t expected = (share_value(true, j) + share_value(false, j)) % he->plain_mod;
            if (decoded[j] != expected) {
                ++mismatches;
                if (mismatches <= sample) {
                    std::cout << "[SSToHE] mismatch at " << j << ": decoded=" << decoded[j]
                              << ", expected=" << expected << std::endl;
                }
            }
        }

        std::cout << "[SSToHE] decoded first " << sample << " values (decoded/expected): ";
        for (size_t j = 0; j < sample && j < degree; ++j) {
            const uint64_t expected = (share_value(true, j) + share_value(false, j)) % he->plain_mod;
            std::cout << decoded[j] << "/" << expected;
            if (j + 1 != sample && j + 1 != degree) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
        std::cout << "[SSToHE] " << (mismatches == 0 ? "PASS" : "FAIL")
                  << " (" << mismatches << " mismatches)" << std::endl;
    }
}

void test_hetoss_roundtrip(HE::HEEvaluator* he) {
    const size_t degree = he->polyModulusDegree;
    Tensor<uint64_t> share({1, degree});
    const size_t sample = std::min<size_t>(8, degree);

    auto share_value = [plain = he->plain_mod](bool is_server, size_t idx) -> uint64_t {
        const uint64_t base = static_cast<uint64_t>(idx % plain);
        return is_server ? (base + 5) % plain : (3 * base + 7) % plain;
    };

    for (size_t j = 0; j < degree; ++j) {
        share(j) = share_value(he->server, j);
    }

    std::cout << "[HEToSS] input share (" << (he->server ? "server" : "client")
              << ") first " << sample << ": ";
    for (size_t j = 0; j < sample; ++j) {
        std::cout << share(j);
        if (j + 1 != sample) std::cout << ", ";
    }
    std::cout << std::endl;

    auto ct = Operator::SSToHE(share, he);
    auto ss = Operator::HEToSS(ct, he);

    std::cout << "[HEToSS] output share (" << (he->server ? "server" : "client")
              << ") first " << sample << ": ";
    for (size_t j = 0; j < sample; ++j) {
        std::cout << ss(j);
        if (j + 1 != sample) std::cout << ", ";
    }
    std::cout << std::endl;

    if (he->server) {  // ALICE
        Tensor<uint64_t> client_share({1, degree});
        netio->recv_tensor(client_share);

        Tensor<uint64_t> combined = ss + client_share;
        for (size_t j = 0; j < degree; ++j) {
            combined(j) %= he->plain_mod;
        }

        std::cout << "[HEToSS] reconstructed share first " << sample << ": ";
        for (size_t j = 0; j < sample; ++j) {
            std::cout << combined(j);
            if (j + 1 != sample) std::cout << ", ";
        }
        std::cout << std::endl;

        size_t mismatches = 0;
        for (size_t j = 0; j < degree; ++j) {
            const uint64_t expected = (share_value(true, j) + share_value(false, j)) % he->plain_mod;
            if (combined(j) != expected) {
                ++mismatches;
            }
        }

        std::cout << "[HEToSS] " << (mismatches == 0 ? "PASS" : "FAIL")
                  << " (" << mismatches << " mismatches)" << std::endl;
    } else {
        netio->send_tensor(ss);
    }
}

int main(int argc, char **argv){
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2"); // 1 is server, 2 is client
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.parse(argc, argv);
    
    netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
    he = new HE::HEEvaluator(netio, party, 8192,32,Datatype::HOST,{});
    he->GenerateNewKey();
    // test_sstohe_conversion(he);
    // test_hetoss_roundtrip(he);
    test_poly(he);
    // test_ckks_inverse_fft();
    // test_ckks_fft_roundtrip();
    // test_ckks_tensor_encode_decode();

    return 0;
}