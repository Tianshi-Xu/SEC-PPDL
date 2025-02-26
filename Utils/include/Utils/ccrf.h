#pragma once
#ifndef CCRF_H__
#define CCRF_H__
#include "Utils/aes-ni.h"
#include "Utils/aes_opt.h"
#include "Utils/prg.h"
#include <stdio.h>
/** @addtogroup BP
  @{
  */
namespace Utils {

inline void CCRF(block128 *y, block256 *k, int n) {
  AESNI_KEY aes[8];
  int r = n % 8;
  if (r == 0) {
    for (int i = 0; i < n / 8; i++) {
      AES_256_ks8(k + i * 8, aes);
      // AESNI_set_encrypt_key(&aes[j], k[i*8 + j]);
      AESNI_ecb_encrypt_blks_ks_x8(y + i * 8, 8, aes);
    }
  } else {
    for (int i = 0; i < (n - r) / 8; i++) {
      AES_256_ks8(k + i * 8, aes);
      // AESNI_set_encrypt_key(&aes[j], k[i*8 + j]);
      AESNI_ecb_encrypt_blks_ks_x8(y + i * 8, 8, aes);
    }
    for (int i = n - r; i < n; i++) {
      y[i] = one;
      AESNI_set_encrypt_key(&aes[0], k[i]);
      AESNI_ecb_encrypt_blks(y + i, 1, aes);
    }
  }
}

} // namespace Utils
/**@}*/
#endif // CCRF_H__
