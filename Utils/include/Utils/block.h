#pragma once
#include <algorithm>
#include <assert.h>
#include <bitset>
#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <smmintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wmmintrin.h>
#include <xmmintrin.h>

namespace Utils {
typedef __m128i block128;
typedef __m256i block256;

#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC)
static inline void __attribute__((__always_inline__))
_mm256_storeu2_m128i(__m128i *const hiaddr, __m128i *const loaddr,
                     const __m256i a) {
  _mm_storeu_si128(loaddr, _mm256_castsi256_si128(a));
  _mm_storeu_si128(hiaddr, _mm256_extracti128_si256(a, 1));
}
#endif /* defined(__GNUC__) */

inline void print(const uint64_t &value, const char *end = "\n", int len = 64,
                  bool reverse = false) {
  std::string tmp = std::bitset<64>(value).to_string();
  if (reverse)
    std::reverse(tmp.begin(), tmp.end());
  if (reverse)
    std::cout << tmp.substr(0, len); // std::cout << std::hex << buffer[i];
  else
    std::cout << tmp.substr(64 - len,
                            len); // std::cout << std::hex << buffer[i];
  std::cout << end;
}

inline void print(const uint8_t &value, const char *end = "\n", int len = 8,
                  bool reverse = false) {
  std::string tmp = std::bitset<8>(value).to_string();
  if (reverse)
    std::reverse(tmp.begin(), tmp.end());
  if (reverse)
    std::cout << tmp.substr(0, len); // std::cout << std::hex << buffer[i];
  else
    std::cout << tmp.substr(8 - len, len); // std::cout << std::hex <<
                                           // buffer[i];
  std::cout << end;
}

inline void print(const block128 &value, const char *end = "\n") {
  const size_t n = sizeof(__m128i) / sizeof(uint64_t);
  uint64_t buffer[n];
  _mm_storeu_si128((__m128i *)buffer, value);
  // std::cout << "0x";
  for (size_t i = 0; i < n; i++) {
    std::string tmp = std::bitset<64>(buffer[i]).to_string();
    std::reverse(tmp.begin(), tmp.end());
    std::cout << tmp; // std::cout << std::hex << buffer[i];
  }
  std::cout << end;
}

inline void print(const block256 &value, const char *end = "\n") {
  const size_t n = sizeof(__m256i) / sizeof(uint64_t);
  uint64_t buffer[n];
  _mm256_storeu_si256((__m256i *)buffer, value);
  // std::cout << "0x";
  for (size_t i = 0; i < n; i++) {
    std::string tmp = std::bitset<64>(buffer[i]).to_string();
    std::reverse(tmp.begin(), tmp.end());
    std::cout << tmp; // std::cout << std::hex << buffer[i];
  }
  std::cout << end;
}

inline bool getLSB(const block128 &x) { return (*((char *)&x) & 1) == 1; }
__attribute__((target("sse2"))) inline block128 makeBlock128(int64_t x,
                                                             int64_t y) {
  return _mm_set_epi64x(x, y);
}

__attribute__((target("avx"))) inline block256
makeBlock256(int64_t w, int64_t x, int64_t y,
             int64_t z) { // return w||x||y||z (MSB->LSB)
  return _mm256_set_epi64x(w, x, y, z);
}
__attribute__((target("avx2,avx"))) inline block256
makeBlock256(block128 x, block128 y) { // return x (MSB) || y (LSB)
  return _mm256_inserti128_si256(_mm256_castsi128_si256(y), x, 1);
  // return _mm256_loadu2_m128i(&x, &y);
}
__attribute__((target("sse2"))) inline block128 zero_block() {
  return _mm_setzero_si128();
}
inline block128 one_block() {
  return makeBlock128(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
}

const block128 select_mask[2] = {zero_block(), one_block()};

__attribute__((target("sse2"))) inline block128 make_delta(const block128 &a) {
  return _mm_or_si128(makeBlock128(0L, 1L), a);
}

typedef __m128i block_tpl[2];

__attribute__((target("sse2"))) inline block128 xorBlocks(block128 x,
                                                          block128 y) {
  return _mm_xor_si128(x, y);
}
__attribute__((target("avx2"))) inline block256 xorBlocks(block256 x,
                                                          block256 y) {
  return _mm256_xor_si256(x, y);
}
__attribute__((target("sse2"))) inline block128 andBlocks(block128 x,
                                                          block128 y) {
  return _mm_and_si128(x, y);
}
__attribute__((target("avx2"))) inline block256 andBlocks(block256 x,
                                                          block256 y) {
  return _mm256_and_si256(x, y);
}

inline void xorBlocks_arr(block128 *res, const block128 *x, const block128 *y,
                          int nblocks) {
  const block128 *dest = nblocks + x;
  for (; x != dest;) {
    *(res++) = xorBlocks(*(x++), *(y++));
  }
}
inline void xorBlocks_arr(block128 *res, const block128 *x, block128 y,
                          int nblocks) {
  const block128 *dest = nblocks + x;
  for (; x != dest;) {
    *(res++) = xorBlocks(*(x++), y);
  }
}

inline void xorBlocks_arr(block256 *res, const block256 *x, const block256 *y,
                          int nblocks) {
  const block256 *dest = nblocks + x;
  for (; x != dest;) {
    *(res++) = xorBlocks(*(x++), *(y++));
  }
}
inline void xorBlocks_arr(block256 *res, const block256 *x, block256 y,
                          int nblocks) {
  const block256 *dest = nblocks + x;
  for (; x != dest;) {
    *(res++) = xorBlocks(*(x++), y);
  }
}

__attribute__((target("sse4.1,sse2"))) inline bool
cmpBlock(const block128 *x, const block128 *y, int nblocks) {
  const block128 *dest = nblocks + x;
  for (; x != dest;) {
    __m128i vcmp = _mm_xor_si128(*(x++), *(y++));
    if (!_mm_testz_si128(vcmp, vcmp))
      return false;
  }
  return true;
}

__attribute__((target("avx2,avx"))) inline bool
cmpBlock(const block256 *x, const block256 *y, int nblocks) {
  const block256 *dest = nblocks + x;
  for (; x != dest;) {
    __m256i vcmp = _mm256_xor_si256(*(x++), *(y++));
    if (!_mm256_testz_si256(vcmp, vcmp))
      return false;
  }
  return true;
}

// deprecate soon
inline bool block_cmp(const block128 *x, const block128 *y, int nblocks) {
  return cmpBlock(x, y, nblocks);
}

inline bool block_cmp(const block256 *x, const block256 *y, int nblocks) {
  return cmpBlock(x, y, nblocks);
}

__attribute__((target("sse4.1"))) inline bool isZero(const block128 *b) {
  return _mm_testz_si128(*b, *b) > 0;
}

__attribute__((target("avx"))) inline bool isZero(const block256 *b) {
  return _mm256_testz_si256(*b, *b) > 0;
}

__attribute__((target("sse4.1,sse2"))) inline bool isOne(const block128 *b) {
  __m128i neq = _mm_xor_si128(*b, one_block());
  return _mm_testz_si128(neq, neq) > 0;
}

/* Linear orthomorphism function
 * [REF] Implementation of "Efficient and Secure Multiparty Computation from
 * Fixed-Key Block Ciphers" https://eprint.iacr.org/2019/074.pdf
 */
#ifdef __x86_64__
__attribute__((target("sse2")))
#endif
inline block128
sigma(block128 a) {
  return _mm_shuffle_epi32(a, 78) ^
         (a & makeBlock128(0xFFFFFFFFFFFFFFFF, 0x00));
}

inline block128 set_bit(const block128 &a, int i) {
  if (i < 64)
    return makeBlock128(0L, 1ULL << i) | a;
  else
    return makeBlock128(1ULL << (i - 64), 0) | a;
}

// Modified from
// https://mischasan.wordpress.com/2011/10/03/the-full-sse2-bit-matrix-transpose-routine/
// with inner most loops changed to _mm_set_epi8 and _mm_set_epi16
#define INP(x, y) inp[(x)*ncols / 8 + (y) / 8]
#define OUT(x, y) out[(y)*nrows / 8 + (x) / 8]

const char fix_key[] = "\x61\x7e\x8d\xa2\xa0\x51\x1e\x96"
                       "\x5e\x41\xc2\x9b\x15\x3f\xc7\x7a";

/*
        This file is part of JustGarble.

        JustGarble is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        JustGarble is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with JustGarble.  If not, see <http://www.gnu.org/licenses/>.

 */

/*------------------------------------------------------------------------
  / OCB Version 3 Reference Code (Optimized C)     Last modified 08-SEP-2012
  /-------------------------------------------------------------------------
  / Copyright (c) 2012 Ted Krovetz.
  /
  / Permission to use, copy, modify, and/or distribute this software for any
  / purpose with or without fee is hereby granted, provided that the above
  / copyright notice and this permission notice appear in all copies.
  /
  / THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
  / WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
  / MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
  / ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
  / WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
  / ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
  / OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
  /
  / Phillip Rogaway holds patents relevant to OCB. See the following for
  / his patent grant: http://www.cs.ucdavis.edu/~rogaway/ocb/grant.htm
  /
  / Special thanks to Keegan McAllister for suggesting several good improvements
  /
  / Comments are welcome: Ted Krovetz <ted@krovetz.net> - Dedicated to Laurel K
  /------------------------------------------------------------------------- */
__attribute__((target("sse2"))) inline block128 double_block(block128 bl) {
  const __m128i mask = _mm_set_epi32(135, 1, 1, 1);
  __m128i tmp = _mm_srai_epi32(bl, 31);
  tmp = _mm_and_si128(tmp, mask);
  tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2, 1, 0, 3));
  bl = _mm_slli_epi32(bl, 1);
  return _mm_xor_si128(bl, tmp);
}

__attribute__((target("sse2"))) inline block128 LEFTSHIFT1(block128 bl) {
  const __m128i mask = _mm_set_epi32(0, 0, (1 << 31), 0);
  __m128i tmp = _mm_and_si128(bl, mask);
  bl = _mm_slli_epi64(bl, 1);
  return _mm_xor_si128(bl, tmp);
}
__attribute__((target("sse2"))) inline block128 RIGHTSHIFT(block128 bl) {
  const __m128i mask = _mm_set_epi32(0, 1, 0, 0);
  __m128i tmp = _mm_and_si128(bl, mask);
  bl = _mm_slli_epi64(bl, 1);
  return _mm_xor_si128(bl, tmp);
}
} // namespace sci