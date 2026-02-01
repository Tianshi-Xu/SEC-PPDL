#pragma once

#include <hexl/hexl.hpp>
#include <vector>
#include <cstdint>

namespace Utils {

/**
 * CyclicNTT: Implements cyclic convolution (mod x^n - 1) using HEXL's negacyclic NTT.
 * 
 * Background:
 * - HEXL provides negacyclic NTT: polynomial multiplication mod (x^n + 1)
 * - We need cyclic NTT: polynomial multiplication mod (x^n - 1)
 * 
 * Method: Use 2n-length negacyclic NTT to simulate cyclic NTT
 * - Embed n-length polynomials into 2n-length (zero-pad)
 * - Compute negacyclic convolution mod (x^{2n} + 1)
 * - Result: c[i] = c_2n[i] + c_2n[i+n] for i in [0, n)
 * 
 * Complexity: O(n log n) - same asymptotic complexity as direct cyclic NTT
 * 
 * Usage:
 *   CyclicNTT ntt(n, p);  // n must be power of 2
 *   ntt.ComputeForward(result, input);  // Forward cyclic NTT
 *   ntt.ComputeInverse(result, input);  // Inverse cyclic NTT
 *   ntt.ConvolveCyclic(result, a, b);   // Direct cyclic convolution
 */
class CyclicNTT {
public:
    /**
     * Constructor
     * @param n: polynomial degree, must be power of 2
     * @param p: prime modulus, must satisfy 2n | (p-1) for negacyclic NTT
     */
    CyclicNTT(uint64_t n, uint64_t p);
    
    /**
     * Compute forward cyclic NTT (DFT with n-th roots of unity)
     * @param result: output array of size n
     * @param input: input array of size n
     */
    void ComputeForward(uint64_t* result, const uint64_t* input);
    
    /**
     * Compute inverse cyclic NTT
     * @param result: output array of size n
     * @param input: input array of size n
     */
    void ComputeInverse(uint64_t* result, const uint64_t* input);
    
    /**
     * Compute cyclic convolution: result = a * b mod (x^n - 1)
     * @param result: output array of size n
     * @param a: first input array of size n
     * @param b: second input array of size n
     */
    void ConvolveCyclic(uint64_t* result, const uint64_t* a, const uint64_t* b);
    
    uint64_t GetN() const { return n_; }
    uint64_t GetModulus() const { return p_; }

private:
    uint64_t n_;           // cyclic NTT size
    uint64_t p_;           // modulus
    uint64_t log_n_;       // log2(n) for Cooley-Tukey
    intel::hexl::NTT ntt_2n_;  // 2n-length negacyclic NTT (for ConvolveCyclic)
    
    // Precomputed values for O(n log n) Cooley-Tukey
    std::vector<uint64_t> omega_powers_;      // powers of n-th root of unity
    std::vector<uint64_t> omega_inv_powers_;  // powers of inverse
    std::vector<uint64_t> bit_rev_;           // bit-reversal permutation
    uint64_t n_inv_;                          // n^{-1} mod p
    
    // Helper: modular exponentiation
    static uint64_t ModPow(uint64_t base, uint64_t exp, uint64_t mod);
    
    // Helper: find primitive n-th root of unity
    uint64_t FindPrimitiveRoot();
    
    // Helper: compute bit-reversal index
    static uint64_t BitReverse(uint64_t x, uint64_t log_n);
};

} // namespace Utils
