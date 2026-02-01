/**
 * TestNTT: Unit test for CyclicNTT
 */
#include <Utils/CyclicNTT.h>
#include <iostream>
#include <vector>

int main() {
    uint64_t p = 1152921504606830593ULL;  // plain_mod from HE
    
    std::cout << "Testing CyclicNTT (Cooley-Tukey O(n log n))..." << std::endl;
    
    // Test small NTT sizes
    for (uint64_t n : {8, 16, 32, 64, 128, 256}) {
        std::cout << "  n=" << n << "..." << std::flush;
        Utils::CyclicNTT ntt(n, p);
        
        std::vector<uint64_t> x(n), y(n), z(n);
        for (uint64_t i = 0; i < n; i++) x[i] = i + 1;
        
        // Forward NTT
        ntt.ComputeForward(y.data(), x.data());
        
        // Inverse NTT
        ntt.ComputeInverse(z.data(), y.data());
        
        // Check roundtrip
        bool pass = true;
        for (uint64_t i = 0; i < n; i++) {
            if (z[i] != x[i]) {
                std::cout << " FAIL at i=" << i << " got=" << z[i] << " expected=" << x[i] << std::endl;
                pass = false;
                break;
            }
        }
        if (pass) std::cout << " PASS" << std::endl;
    }
    
    std::cout << "\nTesting in-place computation..." << std::endl;
    {
        uint64_t n = 64;
        Utils::CyclicNTT ntt(n, p);
        
        std::vector<uint64_t> x(n), original(n);
        for (uint64_t i = 0; i < n; i++) {
            x[i] = i + 1;
            original[i] = i + 1;
        }
        
        // In-place forward
        ntt.ComputeForward(x.data(), x.data());
        // In-place inverse
        ntt.ComputeInverse(x.data(), x.data());
        
        bool pass = true;
        for (uint64_t i = 0; i < n; i++) {
            if (x[i] != original[i]) {
                std::cout << "  FAIL at i=" << i << std::endl;
                pass = false;
                break;
            }
        }
        if (pass) std::cout << "  In-place roundtrip: PASS" << std::endl;
    }
    
    return 0;
}
