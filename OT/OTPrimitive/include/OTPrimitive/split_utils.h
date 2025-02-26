#pragma once
#include "ot.h"

namespace OTPrimitive {
template <typename basetype>
void pack_a(basetype *a, basetype *a_unpacked, int asize, int bsize,
            int bitsize) {
  assert(a != nullptr && a_unpacked != nullptr);
  uint64_t start_pos = 0;
  uint64_t end_pos = 0;
  uint64_t start_block = 0;
  uint64_t end_block = 0;
  basetype temp_bl = 0;
  basetype mask = (1 << bitsize) - 1;
  if (8 * sizeof(basetype) == 64) {
    mask = (basetype)((1ULL << bitsize) - 1ULL);
  }
  if (8 * sizeof(basetype) == bitsize) {
    if (bitsize == 64) {
      mask = (basetype)(-1ULL);
    } else {
      mask = (basetype)(-1);
    }
  }
  uint64_t carriersize = 8 * (sizeof(basetype));
  for (int i = 0; i < asize; i++) {
    a[i] = 0;
  }
  for (int i = 0; i < bsize; i++) {
    start_pos = i * bitsize;
    end_pos = start_pos + bitsize - 1; // inclusive
    start_block = start_pos / carriersize;
    end_block = end_pos / carriersize;
    if (start_block == end_block) {
      a[start_block] ^= ((a_unpacked[i] & mask) << (start_pos % carriersize));
    } else {
      temp_bl = (a_unpacked[i] & mask);
      a[start_block] ^= (temp_bl) << (start_pos % carriersize);
      a[end_block] ^= (temp_bl) >> (carriersize - (start_pos % carriersize));
    }
  }
}

template <typename basetype>
void unpack_a(basetype *a, basetype *a_packed, int bsize, int bitsize) {
  assert(a != nullptr && a_packed != nullptr);
  uint64_t start_pos = 0;
  uint64_t end_pos = 0;
  uint64_t start_block = 0;
  uint64_t end_block = 0;
  basetype mask = (1 << bitsize) - 1;
  if (8 * sizeof(basetype) == 64) {
    mask = (basetype)((1ULL << bitsize) - 1ULL);
  }
  if (8 * sizeof(basetype) == bitsize) {
    if (bitsize == 64) {
      mask = (basetype)(-1ULL);
    } else {
      mask = (basetype)(-1);
    }
  }
  uint64_t carriersize = 8 * (sizeof(basetype));

  for (int i = 0; i < bsize; i++) {
    start_pos = i * bitsize;
    end_pos = start_pos + bitsize - 1; // inclusive
    start_block = start_pos / carriersize;
    end_block = end_pos / carriersize;
    if (start_block == end_block) {
      a[i] = (a_packed[start_block] >> (start_pos % carriersize)) & mask;
    } else {
      a[i] = 0;
      a[i] ^= (a_packed[start_block] >> (start_pos % carriersize));
      a[i] ^=
          (a_packed[end_block] << (carriersize - (start_pos % carriersize)));
      a[i] = a[i] & mask;
    }
  }
}

template <typename basetype>
void pack_messages(basetype *y, basetype **maskeddata, int ysize, int bsize,
                   int bitsize, int N) {
  assert(y != nullptr && maskeddata != nullptr);
  uint64_t start_pos = 0;
  uint64_t end_pos = 0;
  uint64_t start_block = 0;
  uint64_t end_block = 0;
  basetype temp_bl = 0;
  basetype mask = (1 << bitsize) - 1;
  if (8 * sizeof(basetype) == 64) {
    mask = (basetype)((1ULL << bitsize) - 1ULL);
  }
  if (8 * sizeof(basetype) == bitsize) {
    if (bitsize == 64) {
      mask = (basetype)(-1ULL);
    } else {
      mask = (basetype)(-1);
    }
  }
  uint64_t carriersize = 8 * (sizeof(basetype));
  for (int i = 0; i < ysize; i++) {
    y[i] = 0;
  }
  for (int i = 0; i < bsize; i++) {
    for (int k = 0; k < N; k++) {
      // OT message k
      start_pos = i * N * bitsize + k * bitsize; // inclusive
      end_pos = start_pos + bitsize;
      end_pos -= 1; // inclusive
      start_block = start_pos / carriersize;
      end_block = end_pos / carriersize;

      if (start_block == end_block) {
        y[start_block] ^= (maskeddata[i][k] & mask)
                          << (start_pos % carriersize);
      } else {
        temp_bl = (maskeddata[i][k] & mask);
        y[start_block] ^= (temp_bl) << (start_pos % carriersize);
        y[end_block] ^= (temp_bl) >> (carriersize - (start_pos % carriersize));
      }
    }
  }
}

template <typename basetype>
void unpack_messages(basetype *data, const uint8_t *r, basetype *recvd,
                     basetype *maskhash, int bsize, int bitsize, int N,
                     int &counter) {
  assert(data != nullptr && maskhash != nullptr);
  uint64_t start_pos = 0;
  uint64_t end_pos = 0;
  uint64_t start_block = 0;
  uint64_t end_block = 0;
  basetype mask = (1 << bitsize) - 1;
  if (8 * sizeof(basetype) == 64) {
    mask = (basetype)((1ULL << bitsize) - 1ULL);
  }
  if (8 * sizeof(basetype) == bitsize) {
    if (bitsize == 64) {
      mask = (basetype)(-1ULL);
    } else {
      mask = (basetype)(-1);
    }
  }
  uint64_t carriersize = 8 * (sizeof(basetype));

  for (int i = 0; i < bsize; i++) {
    start_pos = i * N * bitsize + r[i] * bitsize;
    end_pos = start_pos + bitsize - 1; // inclusive
    start_block = start_pos / carriersize;
    end_block = end_pos / carriersize;
    if (start_block == end_block) {
      data[i] = ((recvd[start_block] >> (start_pos % carriersize)) ^
                 maskhash[counter]) &
                mask;
    } else {
      data[i] = 0;
      data[i] ^= (recvd[start_block] >> (start_pos % carriersize));
      data[i] ^=
          (recvd[end_block] << (carriersize - (start_pos % carriersize)));
      data[i] = (data[i] ^ maskhash[counter]) & mask;
    }
    counter++;
  }
}
} // namespace OTPrimitive
