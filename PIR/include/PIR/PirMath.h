#pragma once

#include <cstdint>

namespace secppdl::pir
{

uint32_t ceil_log2_u64(uint64_t x);
uint64_t mod_pow_u64(uint64_t base, uint64_t exp, uint64_t mod);
uint32_t mod_pow_u32(uint32_t base, uint32_t exp, uint32_t mod);
uint64_t mod_inverse_u64(uint64_t a, uint64_t mod);

} // namespace secppdl::pir
