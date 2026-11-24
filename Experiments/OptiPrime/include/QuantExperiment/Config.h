#pragma once
#include <QuantExperiment/PreparedOT.h>
#include <algorithm>
#include <string>
#include <vector>

namespace QuantExperiment {
enum class LinearBackend { Cheetah, CPU, GPU };
struct Config {
    int party = 1, port = 22000, threads = 16;
    OTBackend ot = OTBackend::IKNP;
    LinearBackend linear = LinearBackend::GPU;
    bool preprocess = false, parallel_conversions = true, legacy_conversion = false;
    bool power_of_two = false, ciphertext_inverse = false;
    bool empirical_range = false, approximate_relu = false;
    std::string address = "127.0.0.1";
    int activation_ring_bits = 10, accumulator_bits = 32;
    int activation_bits = 8; // Quantizer width, independent of the MPC share width.
    int ActivationMax() const {
        return (1 << activation_bits) - 1;
    }
    int ActivationMidpoint() const {
        return 1 << (activation_bits - 1);
    }
    int plaintext_bits = 26;
    size_t degree = 8192, output_mod_switches = 1;
    std::vector<int> ciphertext_prime_bits{44, 44, 44};
    int ClassifierBits() const {
        return std::max(26, accumulator_bits);
    }
    int ArgmaxBits() const {
        if (power_of_two)
            return ClassifierBits() + 1; // A difference of two signed logits needs one extra bit.
        return legacy_conversion ? accumulator_bits
                                 : std::min(ClassifierBits(), std::max(26, plaintext_bits));
    }
    void Validate() const;
};
} // namespace QuantExperiment
