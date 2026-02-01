#include <Utils/CyclicNTT.h>
#include <stdexcept>
#include <cstring>

namespace Utils {

uint64_t CyclicNTT::ModPow(uint64_t base, uint64_t exp, uint64_t mod) {
    __uint128_t result = 1;
    __uint128_t b = base % mod;
    while (exp > 0) {
        if (exp & 1) result = (result * b) % mod;
        b = (b * b) % mod;
        exp >>= 1;
    }
    return static_cast<uint64_t>(result);
}

uint64_t CyclicNTT::FindPrimitiveRoot() {
    // Find a primitive n-th root of unity
    // We need ω such that ω^n = 1 and ω^k ≠ 1 for 0 < k < n
    uint64_t p_minus_1 = p_ - 1;
    
    if (p_minus_1 % n_ != 0) {
        throw std::runtime_error("n does not divide p-1, cyclic NTT not possible");
    }
    
    // Try different generators until we find a primitive n-th root
    for (uint64_t g = 2; g < 10000; g++) {
        uint64_t candidate = ModPow(g, p_minus_1 / n_, p_);
        
        // Verify ω^n = 1
        if (ModPow(candidate, n_, p_) != 1) continue;
        
        // For power-of-2 n, only need to check ω^(n/2) ≠ 1
        if (ModPow(candidate, n_ / 2, p_) != 1) {
            return candidate;
        }
    }
    
    throw std::runtime_error("Could not find primitive n-th root of unity");
}

// Compute bit-reversal index
uint64_t CyclicNTT::BitReverse(uint64_t x, uint64_t log_n) {
    uint64_t result = 0;
    for (uint64_t i = 0; i < log_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

CyclicNTT::CyclicNTT(uint64_t n, uint64_t p)
    : n_(n), p_(p), ntt_2n_(2 * n, p)
{
    // Verify n is power of 2
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("n must be a power of 2");
    }
    
    // Compute log2(n)
    log_n_ = 0;
    uint64_t tmp = n;
    while (tmp > 1) {
        tmp >>= 1;
        log_n_++;
    }
    
    // Find primitive n-th root of unity
    uint64_t omega = FindPrimitiveRoot();
    
    // Precompute powers of omega for each level of Cooley-Tukey
    // omega_powers_[i] = omega^i for i in [0, n)
    omega_powers_.resize(n_);
    omega_powers_[0] = 1;
    for (uint64_t i = 1; i < n_; i++) {
        omega_powers_[i] = (static_cast<__uint128_t>(omega_powers_[i-1]) * omega) % p_;
    }
    
    // Precompute powers of omega^{-1}
    uint64_t omega_inv = ModPow(omega, p_ - 2, p_);
    omega_inv_powers_.resize(n_);
    omega_inv_powers_[0] = 1;
    for (uint64_t i = 1; i < n_; i++) {
        omega_inv_powers_[i] = (static_cast<__uint128_t>(omega_inv_powers_[i-1]) * omega_inv) % p_;
    }
    
    // Precompute bit-reversal permutation
    bit_rev_.resize(n_);
    for (uint64_t i = 0; i < n_; i++) {
        bit_rev_[i] = BitReverse(i, log_n_);
    }
    
    // Precompute n^{-1}
    n_inv_ = ModPow(n_, p_ - 2, p_);
}

void CyclicNTT::ComputeForward(uint64_t* result, const uint64_t* input) {
    /**
     * O(n log n) Cooley-Tukey NTT
     * 
     * Cyclic NTT: X[k] = sum_{j=0}^{n-1} x[j] * omega^{jk}
     * Uses iterative decimation-in-time algorithm with bit-reversal permutation.
     */
    
    // For in-place: need temporary buffer since bit-reversal reads from input
    std::vector<uint64_t> tmp;
    const uint64_t* src = input;
    if (result == input) {
        tmp.assign(input, input + n_);
        src = tmp.data();
    }
    
    // Copy input to result with bit-reversal permutation
    for (uint64_t i = 0; i < n_; i++) {
        result[bit_rev_[i]] = src[i];
    }
    
    // Cooley-Tukey butterfly
    for (uint64_t s = 1; s <= log_n_; s++) {
        uint64_t m = 1ULL << s;           // m = 2^s
        uint64_t m_half = m >> 1;         // m/2
        uint64_t step = n_ >> s;          // n / 2^s = stride in omega_powers
        
        for (uint64_t k = 0; k < n_; k += m) {
            for (uint64_t j = 0; j < m_half; j++) {
                uint64_t w = omega_powers_[j * step];
                uint64_t u = result[k + j];
                uint64_t t = (static_cast<__uint128_t>(w) * result[k + j + m_half]) % p_;
                
                result[k + j] = (u + t) % p_;
                result[k + j + m_half] = (u + p_ - t) % p_;
            }
        }
    }
}

void CyclicNTT::ComputeInverse(uint64_t* result, const uint64_t* input) {
    /**
     * O(n log n) Inverse NTT using Cooley-Tukey
     * 
     * Inverse: x[j] = (1/n) * sum_{k=0}^{n-1} X[k] * omega^{-jk}
     * Same algorithm but with omega^{-1} and final multiplication by n^{-1}
     */
    
    // For in-place: need temporary buffer since bit-reversal reads from input
    std::vector<uint64_t> tmp;
    const uint64_t* src = input;
    if (result == input) {
        tmp.assign(input, input + n_);
        src = tmp.data();
    }
    
    // Copy input to result with bit-reversal permutation
    for (uint64_t i = 0; i < n_; i++) {
        result[bit_rev_[i]] = src[i];
    }
    
    // Cooley-Tukey butterfly with inverse twiddle factors
    for (uint64_t s = 1; s <= log_n_; s++) {
        uint64_t m = 1ULL << s;
        uint64_t m_half = m >> 1;
        uint64_t step = n_ >> s;
        
        for (uint64_t k = 0; k < n_; k += m) {
            for (uint64_t j = 0; j < m_half; j++) {
                uint64_t w = omega_inv_powers_[j * step];
                uint64_t u = result[k + j];
                uint64_t t = (static_cast<__uint128_t>(w) * result[k + j + m_half]) % p_;
                
                result[k + j] = (u + t) % p_;
                result[k + j + m_half] = (u + p_ - t) % p_;
            }
        }
    }
    
    // Multiply by n^{-1}
    for (uint64_t i = 0; i < n_; i++) {
        result[i] = (static_cast<__uint128_t>(result[i]) * n_inv_) % p_;
    }
}

void CyclicNTT::ConvolveCyclic(uint64_t* result, const uint64_t* a, const uint64_t* b) {
    // Cyclic convolution using 2n negacyclic NTT (O(n log n))
    // 
    // Key insight: For cyclic conv mod (x^n - 1):
    // 1. Zero-pad a, b to 2n length
    // 2. Compute negacyclic conv mod (x^{2n} + 1)
    // 3. Result[i] = conv_2n[i] + conv_2n[i+n]
    
    std::vector<uint64_t> a_2n(2 * n_, 0);
    std::vector<uint64_t> b_2n(2 * n_, 0);
    
    std::memcpy(a_2n.data(), a, n_ * sizeof(uint64_t));
    std::memcpy(b_2n.data(), b, n_ * sizeof(uint64_t));
    
    // Forward NTT
    ntt_2n_.ComputeForward(a_2n.data(), a_2n.data(), 1, 1);
    ntt_2n_.ComputeForward(b_2n.data(), b_2n.data(), 1, 1);
    
    // Point-wise multiplication
    std::vector<uint64_t> c_2n(2 * n_);
    for (uint64_t i = 0; i < 2 * n_; i++) {
        c_2n[i] = (static_cast<__uint128_t>(a_2n[i]) * b_2n[i]) % p_;
    }
    
    // Inverse NTT
    ntt_2n_.ComputeInverse(c_2n.data(), c_2n.data(), 1, 1);
    
    // Combine: result[i] = c_2n[i] + c_2n[i+n]
    for (uint64_t i = 0; i < n_; i++) {
        result[i] = (c_2n[i] + c_2n[i + n_]) % p_;
    }
}

} // namespace Utils
