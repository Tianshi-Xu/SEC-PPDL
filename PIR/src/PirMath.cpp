#include "PIR/PirMath.h"

#include <stdexcept>

namespace secppdl::pir
{

uint32_t ceil_log2_u64(uint64_t x)
{
    if (x <= 1)
    {
        return 0;
    }

    uint32_t lg = 0;
    --x;
    while (x)
    {
        x >>= 1;
        ++lg;
    }
    return lg;
}

uint64_t mod_pow_u64(uint64_t base, uint64_t exp, uint64_t mod)
{
    uint64_t result = 1 % mod;
    base %= mod;
    while (exp)
    {
        if (exp & 1ULL)
        {
            result = static_cast<uint64_t>((__uint128_t)result * base % mod);
        }
        base = static_cast<uint64_t>((__uint128_t)base * base % mod);
        exp >>= 1ULL;
    }
    return result;
}

uint32_t mod_pow_u32(uint32_t base, uint32_t exp, uint32_t mod)
{
    uint64_t result = 1ULL % mod;
    uint64_t cur = base % mod;
    while (exp)
    {
        if (exp & 1U)
        {
            result = (result * cur) % mod;
        }
        cur = (cur * cur) % mod;
        exp >>= 1U;
    }
    return static_cast<uint32_t>(result);
}

uint64_t mod_inverse_u64(uint64_t a, uint64_t mod)
{
    if (a == 0 || mod <= 2)
    {
        throw std::invalid_argument("Invalid modular inverse input.");
    }
    return mod_pow_u64(a, mod - 2, mod);
}

} // namespace secppdl::pir
