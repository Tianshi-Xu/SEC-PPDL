#pragma once

#include <phantom/ciphertext.h>

namespace secppdl::pir
{

void copy_ciphertext_device_fast(const PhantomCiphertext &src, PhantomCiphertext &dst);

} // namespace secppdl::pir
