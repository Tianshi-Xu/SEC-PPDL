#include <OTProtocol/aux-protocols.h>

// #include <Operator/truncation.h>
// #include "<Operator/value-extension.h>

using namespace OTPrimitive;
using namespace Utils;
using namespace OTProtocol;
namespace OTProtocol {
AuxProtocols::AuxProtocols(int party, Utils::NetIO *io,
                           OTPack<Utils::NetIO> *otpack) {
  this->party = party;
  this->io = io;
  this->otpack = otpack;
  this->mill = new MillionaireProtocol<Utils::NetIO>(party, io, otpack);
  this->mill_and_eq =
      new MillionaireWithEquality<Utils::NetIO>(party, io, otpack);
}

AuxProtocols::~AuxProtocols() {
  delete mill;
  delete mill_and_eq;
}

void AuxProtocols::wrap_computation(uint64_t *x, uint8_t *y, int32_t size,
                                    int32_t bw_x) {
  assert(bw_x <= 64);
  uint64_t mask = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));

  uint64_t *tmp_x = new uint64_t[size];
  for (int i = 0; i < size; i++) {
    if (party == ALICE)
      tmp_x[i] = x[i] & mask;
    else
      tmp_x[i] = (mask - x[i]) & mask;  // 2^{bw_x} - 1 - x[i]
  }
  mill->compare(y, tmp_x, size, bw_x, true);  // computing greater_than

  delete[] tmp_x;
}

// 128-bit version
void AuxProtocols::wrap_computation(int128_t *x, uint8_t *y, int32_t size,
                                    int32_t bw_x) {
  assert(bw_x <= 128);
  int128_t mask = (bw_x == 128 ? -1 : ((int128_t(1) << bw_x) - 1));

  int128_t *tmp_x = new int128_t[size];
  for (int i = 0; i < size; i++) {
    if (party == ALICE)
      tmp_x[i] = x[i] & mask;
    else
      tmp_x[i] = (mask - x[i]) & mask;  // 2^{bw_x} - 1 - x[i]
  }
  mill->compare(y, tmp_x, size, bw_x, true);  // computing greater_than

  delete[] tmp_x;
}

void AuxProtocols::wrap_computation_prime(uint64_t *x, uint8_t *y, int32_t size,
                                    int32_t bw_x, uint64_t Q) {
  assert(bw_x <= 64);
  uint64_t *tmp_x = new uint64_t[size];
  for (int i = 0; i < size; i++) {
    if (party == ALICE)
      tmp_x[i] = x[i] % Q;
    else
      tmp_x[i] = (Q - 1 - x[i]) % Q;  // 2^{bw_x} - 1 - x[i]
  }
  mill->compare(y, tmp_x, size, bw_x, true);  // computing greater_than

  delete[] tmp_x;
}


void AuxProtocols::B2A(uint8_t *x, uint64_t *y, int32_t size, int32_t bw_y) {
  assert(bw_y <= 64 && bw_y >= 1);
  if (bw_y == 1) {
    for (int i = 0; i < size; i++) {
      y[i] = uint64_t(x[i]) & 1;
    }
    return;
  }
  uint64_t mask = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

  if (party == ALICE) {
    uint64_t *corr_data = new uint64_t[size];
    for (int i = 0; i < size; i++) {
      corr_data[i] = (-2 * uint64_t(x[i])) & mask;
    }
    otpack->iknp_straight->send_cot(y, corr_data, size, bw_y);

    for (int i = 0; i < size; i++) {
      y[i] = (uint64_t(x[i]) - y[i]) & mask;
    }
    delete[] corr_data;
  } else {  // party == Utils::BOB
    otpack->iknp_straight->recv_cot(y, (bool *)x, size, bw_y);

    for (int i = 0; i < size; i++) {
      y[i] = (uint64_t(x[i]) + y[i]) & mask;
    }
  }
}

template <typename T>
void AuxProtocols::lookup_table(T **spec, T *x, T *y, int32_t size,
                                int32_t bw_x, int32_t bw_y) {
  if (party == ALICE) {
    assert(x == nullptr);
    assert(y == nullptr);
  } else {  // party == BOB
    assert(spec == nullptr);
  }
  assert(bw_x <= 8 && bw_x >= 2);
  int32_t T_size = sizeof(T) * 8;
  assert(bw_y <= T_size);

  T mask_x = (bw_x == T_size ? -1 : ((1ULL << bw_x) - 1));
  T mask_y = (bw_y == T_size ? -1 : ((1ULL << bw_y) - 1));
  uint64_t N = 1 << bw_x;

  if (party == ALICE) {
    PRG128 prg;
    T **data = new T *[size];
    for (int i = 0; i < size; i++) {
      data[i] = new T[N];
      for (uint64_t j = 0; j < N; j++) {
        data[i][j] = spec[i][j];
      }
    }

    otpack->kkot[bw_x - 1]->send(data, size, bw_y);

    for (int i = 0; i < size; i++) delete[] data[i];
    delete[] data;
  } else {  // party == BOB
    uint8_t *choice = new uint8_t[size];
    for (int i = 0; i < size; i++) {
      choice[i] = x[i] & mask_x;
    }
    otpack->kkot[bw_x - 1]->recv(y, choice, size, bw_y);

    delete[] choice;
  }
}



void AuxProtocols::MSB_to_Wrap(uint64_t *x, uint8_t *msb_x, uint8_t *wrap_x,
                               int32_t size, int32_t bw_x) {
  assert(bw_x <= 64);
  if (party == ALICE) {
    PRG128 prg;
    prg.random_bool((bool *)wrap_x, size);
    uint8_t **spec = new uint8_t *[size];
    for (int i = 0; i < size; i++) {
      spec[i] = new uint8_t[4];
      uint8_t msb_xb = (x[i] >> (bw_x - 1)) & 1;
      for (int j = 0; j < 4; j++) {
        uint8_t bits_j[2];  // j0 || j1 (LSB to MSB)
        uint8_to_bool(bits_j, j, 2);
        spec[i][j] = (((1 ^ msb_x[i] ^ bits_j[0]) * (msb_xb ^ bits_j[1])) ^
                      (msb_xb * bits_j[1]) ^ wrap_x[i]) &
                     1;
      }
    }
    lookup_table<uint8_t>(spec, nullptr, nullptr, size, 2, 1);

    for (int i = 0; i < size; i++) delete[] spec[i];
    delete[] spec;
  } else {  // party == BOB
    uint8_t *lut_in = new uint8_t[size];
    for (int i = 0; i < size; i++) {
      lut_in[i] = (((x[i] >> (bw_x - 1)) & 1) << 1) | msb_x[i];
    }
    lookup_table<uint8_t>(nullptr, lut_in, wrap_x, size, 2, 1);

    delete[] lut_in;
  }
}

void AuxProtocols::z_extend(int32_t dim, uint64_t *inA, uint64_t *outB,
                          int32_t bwA, int32_t bwB, uint8_t *msbA) {
  if (bwA == bwB) {
    std::cout << "warning: z_extend with same bitwidth is not allowed" << std::endl;
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert(bwB > bwA && "Extended bitwidth should be > original");
  uint64_t mask_bwA = (bwA == 64 ? -1 : ((1ULL << bwA) - 1));
  uint64_t mask_bwB = (bwB == 64 ? -1 : ((1ULL << bwB) - 1));
  uint8_t *wrap = new uint8_t[dim];

  if (msbA != nullptr) {
    this->MSB_to_Wrap(inA, msbA, wrap, dim, bwA);
  } else {
    this->wrap_computation(inA, wrap, dim, bwA);
  }

  uint64_t *arith_wrap = new uint64_t[dim];
  this->B2A(wrap, arith_wrap, dim, (bwB - bwA));

  for (int i = 0; i < dim; i++) {
    outB[i] = ((inA[i] & mask_bwA) - (1ULL << bwA) * arith_wrap[i]) & mask_bwB;
  }

  delete[] wrap;
  delete[] arith_wrap;
}

void AuxProtocols::s_extend(int32_t dim, uint64_t *inA, uint64_t *outB,
                          int32_t bwA, int32_t bwB, uint8_t *msbA) {
  if (bwA == bwB) {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert(bwB > bwA && "Extended bitwidth should be > original");
  uint64_t mask_bwA = (bwA == 64 ? -1 : ((1ULL << bwA) - 1));
  uint64_t mask_bwB = (bwB == 64 ? -1 : ((1ULL << bwB) - 1));

  uint64_t *mapped_inA = new uint64_t[dim];
  uint64_t *mapped_outB = new uint64_t[dim];
  if (party ==ALICE) {
    for (int i = 0; i < dim; i++) {
      mapped_inA[i] = (inA[i] + (1ULL << (bwA - 1))) & mask_bwA;
    }
  } else { // BOB
    for (int i = 0; i < dim; i++) {
      mapped_inA[i] = inA[i];
    }
  }

  uint8_t *tmp_msbA = nullptr;
  if (msbA != nullptr) {
    tmp_msbA = new uint8_t[dim];
    for (int i = 0; i < dim; i++) {
      tmp_msbA[i] = (party ==ALICE ? msbA[i] ^ 1 : msbA[i]);
    }
  }
  this->z_extend(dim, mapped_inA, mapped_outB, bwA, bwB, tmp_msbA);
  if (msbA != nullptr) {
    delete[] tmp_msbA;
  }

  if (party ==ALICE) {
    for (int i = 0; i < dim; i++) {
      outB[i] = (mapped_outB[i] - (1ULL << (bwA - 1))) & mask_bwB;
    }
  } else { // BOB
    for (int i = 0; i < dim; i++) {
      outB[i] = (mapped_outB[i]) & mask_bwB;
    }
  }
  delete[] mapped_inA;
  delete[] mapped_outB;
}

// 128-bit versions of z_extend and s_extend
void AuxProtocols::z_extend(int32_t dim, int128_t *inA, int128_t *outB,
                          int32_t bwA, int32_t bwB, uint8_t *msbA) {
  if (bwA == bwB) {
    std::cout << "warning: z_extend with same bitwidth is not allowed" << std::endl;
    memcpy(outB, inA, sizeof(int128_t) * dim);
    return;
  }
  assert(bwB > bwA && "Extended bitwidth should be > original");
  assert(bwA <= 128 && bwB <= 128 && "Bitwidth must be <= 128");
  
  int128_t mask_bwA = (bwA == 128 ? -1 : ((int128_t(1) << bwA) - 1));
  int128_t mask_bwB = (bwB == 128 ? -1 : ((int128_t(1) << bwB) - 1));
  uint8_t *wrap = new uint8_t[dim];

  if (bwA <= 64) {
    // For bwA <= 64, we can use the existing 64-bit wrap_computation
    uint64_t *inA_64 = new uint64_t[dim];
    for (int i = 0; i < dim; i++) {
      inA_64[i] = static_cast<uint64_t>(inA[i] & mask_bwA);
    }
    
    if (msbA != nullptr) {
      uint8_t *msbA_64 = new uint8_t[dim];
      for (int i = 0; i < dim; i++) {
        msbA_64[i] = msbA[i];
      }
      this->MSB_to_Wrap(inA_64, msbA_64, wrap, dim, bwA);
      delete[] msbA_64;
    } else {
      this->wrap_computation(inA_64, wrap, dim, bwA);
    }
    delete[] inA_64;
  } else {
    // For bwA > 64, use 128-bit wrap_computation
    this->wrap_computation(inA, wrap, dim, bwA);
  }

  uint64_t *arith_wrap = new uint64_t[dim];
  int32_t extend_bits = bwB - bwA;
  this->B2A(wrap, arith_wrap, dim, std::min(64, extend_bits));

  int128_t shift_val = (int128_t(1) << bwA);
  for (int i = 0; i < dim; i++) {
    int128_t wrapped_part = shift_val * static_cast<int128_t>(arith_wrap[i]);
    outB[i] = ((inA[i] & mask_bwA) - wrapped_part) & mask_bwB;
  }

  delete[] wrap;
  delete[] arith_wrap;
}

void AuxProtocols::s_extend(int32_t dim, int128_t *inA, int128_t *outB,
                          int32_t bwA, int32_t bwB, uint8_t *msbA) {
  if (bwA == bwB) {
    memcpy(outB, inA, sizeof(int128_t) * dim);
    return;
  }
  assert(bwB > bwA && "Extended bitwidth should be > original");
  assert(bwA <= 128 && bwB <= 128 && "Bitwidth must be <= 128");
  
  int128_t mask_bwA = (bwA == 128 ? -1 : ((int128_t(1) << bwA) - 1));
  int128_t mask_bwB = (bwB == 128 ? -1 : ((int128_t(1) << bwB) - 1));

  int128_t *mapped_inA = new int128_t[dim];
  int128_t *mapped_outB = new int128_t[dim];
  
  int128_t offset = (int128_t(1) << (bwA - 1));
  if (party == ALICE) {
    for (int i = 0; i < dim; i++) {
      mapped_inA[i] = (inA[i] + offset) & mask_bwA;
    }
  } else { // BOB
    for (int i = 0; i < dim; i++) {
      mapped_inA[i] = inA[i];
    }
  }

  uint8_t *tmp_msbA = nullptr;
  if (msbA != nullptr) {
    tmp_msbA = new uint8_t[dim];
    for (int i = 0; i < dim; i++) {
      tmp_msbA[i] = (party == ALICE ? msbA[i] ^ 1 : msbA[i]);
    }
  }
  this->z_extend(dim, mapped_inA, mapped_outB, bwA, bwB, tmp_msbA);
  if (tmp_msbA != nullptr) {
    delete[] tmp_msbA;
  }

  if (party == ALICE) {
    for (int i = 0; i < dim; i++) {
      outB[i] = (mapped_outB[i] - offset) & mask_bwB;
    }
  } else { // BOB
    for (int i = 0; i < dim; i++) {
      outB[i] = (mapped_outB[i]) & mask_bwB;
    }
  }
  delete[] mapped_inA;
  delete[] mapped_outB;
}


// void AuxProtocols::msb0_to_wrap(uint64_t *x, uint8_t *wrap_x, int32_t size,
//                                 int32_t bw_x) {
//   assert(bw_x <= 64);
//   if (party == ALICE) {
//     PRG128 prg;
//     prg.random_bool((bool *)wrap_x, size);
//     uint8_t **spec = new uint8_t *[size];
//     for (int i = 0; i < size; i++) {
//       spec[i] = new uint8_t[2];
//       uint8_t msb_xb = (x[i] >> (bw_x - 1)) & 1;
//       spec[i][0] = wrap_x[i] ^ msb_xb;
//       spec[i][1] = wrap_x[i] ^ 1;
//     }
// #if USE_CHEETAH
//     otpack->silent_ot->send_ot_cm_cc<uint8_t>(spec, size, 1);
// #else
//     otpack->iknp_straight->send(spec, size, 1);
// #endif

//     for (int i = 0; i < size; i++) delete[] spec[i];
//     delete[] spec;
//   } else {  // party == BOB
//     uint8_t *msb_xb = new uint8_t[size];
//     for (int i = 0; i < size; i++) {
//       msb_xb[i] = (x[i] >> (bw_x - 1)) & 1;
//     }
// #if USE_CHEETAH
//     otpack->silent_ot->recv_ot_cm_cc<uint8_t>(wrap_x, msb_xb, size, 1);
// #else
//     otpack->iknp_straight->recv(wrap_x, msb_xb, size, 1);
// #endif

//     delete[] msb_xb;
//   }
// }

// void AuxProtocols::msb1_to_wrap(uint64_t *x, uint8_t *wrap_x, int32_t size,
//                                 int32_t bw_x) {
//   assert(bw_x <= 64);
//   if (party == ALICE) {
//     PRG128 prg;
//     prg.random_bool((bool *)wrap_x, size);
//     uint8_t **spec = new uint8_t *[size];
//     for (int i = 0; i < size; i++) {
//       spec[i] = new uint8_t[2];
//       uint8_t msb_xb = (x[i] >> (bw_x - 1)) & 1;
//       spec[i][0] = wrap_x[i];
//       spec[i][1] = wrap_x[i] ^ msb_xb;
//     }
// #if USE_CHEETAH
//     otpack->silent_ot->send_ot_cm_cc<uint8_t>(spec, size, 1);
// #else

//     otpack->iknp_straight->send(spec, size, 1);
// #endif

//     for (int i = 0; i < size; i++) delete[] spec[i];
//     delete[] spec;
//   } else {  // party == BOB
//     uint8_t *msb_xb = new uint8_t[size];
//     for (int i = 0; i < size; i++) {
//       msb_xb[i] = (x[i] >> (bw_x - 1)) & 1;
//     }
// #if USE_CHEETAH
//     otpack->silent_ot->recv_ot_cm_cc<uint8_t>(wrap_x, msb_xb, size, 1);
// #else
//     otpack->iknp_straight->recv(wrap_x, msb_xb, size, 1);
// #endif

//     delete[] msb_xb;
//   }
// }

void AuxProtocols::AND(uint8_t *x, uint8_t *y, uint8_t *z, int32_t size) {
  int old_size = size;
  size = ceil(size / 8.0) * 8;
  uint8_t *tmp_x = new uint8_t[size];
  uint8_t *tmp_y = new uint8_t[size];
  uint8_t *tmp_z = new uint8_t[size];
  memcpy(tmp_x, x, old_size * sizeof(uint8_t));
  memcpy(tmp_y, y, old_size * sizeof(uint8_t));

  // assert((size % 8) == 0);
  Triple triples_std(size, true);
  this->mill->triple_gen->generate(party, &triples_std, _16KKOT_to_4OT);

  uint8_t *ei = new uint8_t[(size) / 8];
  uint8_t *fi = new uint8_t[(size) / 8];
  uint8_t *e = new uint8_t[(size) / 8];
  uint8_t *f = new uint8_t[(size) / 8];

  // this->mill->AND_step_1(ei, fi, x, y, triples_std.ai, triples_std.bi, size);
  this->mill->AND_step_1(ei, fi, tmp_x, tmp_y, triples_std.ai, triples_std.bi,
                         size);

  int size_used = size / 8;
  if (party == ALICE) {
    // Send share of e and f
    io->send_data(ei, size_used);
    io->send_data(ei, size_used);
    io->send_data(fi, size_used);
    io->send_data(fi, size_used);
    // Receive share of e and f
    io->recv_data(e, size_used);
    io->recv_data(e, size_used);
    io->recv_data(f, size_used);
    io->recv_data(f, size_used);
  } else  // party = BOB
  {
    // Receive share of e and f
    io->recv_data(e, size_used);
    io->recv_data(e, size_used);
    io->recv_data(f, size_used);
    io->recv_data(f, size_used);
    // Send share of e and f
    io->send_data(ei, size_used);
    io->send_data(ei, size_used);
    io->send_data(fi, size_used);
    io->send_data(fi, size_used);
  }

  // Reconstruct e and f
  for (int i = 0; i < size_used; i++) {
    e[i] ^= ei[i];
    f[i] ^= fi[i];
  }

  // this->mill->AND_step_2(z, e, f, nullptr, nullptr, triples_std.ai,
  //         triples_std.bi, triples_std.ci, size);
  this->mill->AND_step_2(tmp_z, e, f, nullptr, nullptr, triples_std.ai,
                         triples_std.bi, triples_std.ci, size);
  memcpy(z, tmp_z, old_size * sizeof(uint8_t));

  // cleanup
  delete[] tmp_x;
  delete[] tmp_y;
  delete[] tmp_z;
  delete[] ei;
  delete[] fi;
  delete[] e;
  delete[] f;

  return;
}

// void AuxProtocols::reduce(int32_t dim, uint64_t *x, uint64_t *y, int32_t bw_x,
//                           int32_t bw_y) {
//   assert(bw_y <= bw_x);
//   uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

//   for (int i = 0; i < dim; i++) {
//     y[i] = x[i] & mask_y;
//   }
// }

// void AuxProtocols::digit_decomposition(int32_t dim, uint64_t *x,
//                                        uint64_t *x_digits, int32_t bw_x,
//                                        int32_t digit_size) {
//   assert(false && "Inefficient version of digit decomposition called");
//   int num_digits = ceil(double(bw_x) / digit_size);
//   int last_digit_size = bw_x - (num_digits - 1) * digit_size;
//   uint64_t digit_mask = (digit_size == 64 ? -1 : (1ULL << digit_size) - 1);
//   uint64_t last_digit_mask =
//       (last_digit_size == 64 ? -1 : (1ULL << last_digit_size) - 1);

//   Truncation trunc(this->party, this->io, this->otpack);
//   for (int i = 0; i < num_digits; i++) {
//     trunc.truncate_and_reduce(dim, x, x_digits + i * dim, i * digit_size, bw_x);
//     uint64_t mask = (i == (num_digits - 1) ? last_digit_mask : digit_mask);
//     for (int j = 0; j < dim; j++) {
//       x_digits[i * dim + j] &= mask;
//     }
//   }
// }

// void AuxProtocols::digit_decomposition_sci(int32_t dim, uint64_t *x,
//                                            uint64_t *x_digits, int32_t bw_x,
//                                            int32_t digit_size,
//                                            bool all_digit_size) {
//   int num_digits = ceil(double(bw_x) / digit_size);
//   int last_digit_size = bw_x - (num_digits - 1) * digit_size;
//   uint64_t digit_mask = (digit_size == 64 ? -1 : (1ULL << digit_size) - 1);
//   uint64_t last_digit_mask =
//       (last_digit_size == 64 ? -1 : (1ULL << last_digit_size) - 1);
//   for (int i = 0; i < num_digits; i++) {
//     for (int j = 0; j < dim; j++) {
//       x_digits[i * dim + j] = (x[j] >> (i * digit_size));
//       x_digits[i * dim + j] &=
//           (i == (num_digits - 1)) ? last_digit_mask : digit_mask;
//     }
//   }
//   uint8_t *wrap_ = new uint8_t[dim * (num_digits - 1)];
//   uint8_t *ones_ = new uint8_t[dim * (num_digits - 1)];
//   uint8_t *dp_wrap_entering = new uint8_t[dim * num_digits];
//   uint8_t *dp_temp = new uint8_t[dim * num_digits];
//   uint64_t *dp_wrap_arith = new uint64_t[dim * num_digits];
//   // Fill wrap_ and ones_
//   uint64_t *temp_x_digits = new uint64_t[dim * (num_digits - 1)];

//   for (int i = 0; i < (num_digits - 1); i++) {
//     for (int j = 0; j < dim; j++) {
//       if (party == ALICE)
//         temp_x_digits[i * dim + j] = x_digits[i * dim + j] & digit_mask;
//       else
//         temp_x_digits[i * dim + j] =
//             (digit_mask - x_digits[i * dim + j]) & digit_mask;
//     }
//   }
//   this->mill_and_eq->compare_with_eq(wrap_, ones_, temp_x_digits,
//                                      (dim * (num_digits - 1)), digit_size);

//   // DP steps proceed
//   for (int i = 0; i < num_digits; i++) {
//     if (i > 0) {
//       this->AND(ones_ + (i - 1) * dim, dp_wrap_entering + (i - 1) * dim,
//                 dp_temp + (i - 1) * dim, dim);
//     }
//     for (int j = 0; j < dim; j++) {
//       if (i == 0) {
//         dp_wrap_entering[i * dim + j] = 0;
//       } else {
//         dp_wrap_entering[i * dim + j] =
//             wrap_[(i - 1) * dim + j] ^ dp_temp[(i - 1) * dim + j];
//       }
//     }
//   }
//   this->B2A(dp_wrap_entering, dp_wrap_arith, num_digits * dim, digit_size);
//   for (int i = 0; i < num_digits; i++) {
//     for (int j = 0; j < dim; j++) {
//       x_digits[i * dim + j] += dp_wrap_arith[i * dim + j];
//       uint64_t temp_mask =
//           (i == (num_digits - 1)) ? last_digit_mask : digit_mask;
//       x_digits[i * dim + j] &= temp_mask;
//     }
//     if (all_digit_size) {
//       if (i == (num_digits - 1)) {
//         XTProtocol *xt = new XTProtocol(this->party, this->io, this->otpack);
//         uint64_t *temp_last_digs = new uint64_t[dim];
//         xt->z_extend(dim, x_digits + (num_digits - 1) * dim, temp_last_digs,
//                      last_digit_size, digit_size);
//         for (int j = 0; j < dim; j++) {
//           x_digits[i * dim + j] = temp_last_digs[j];
//           x_digits[i * dim + j] &= digit_mask;
//         }
//         delete xt;
//         delete[] temp_last_digs;
//       }
//     }
//   }

//   delete[] wrap_;
//   delete[] ones_;
//   delete[] dp_wrap_entering;
//   delete[] dp_temp;
//   delete[] dp_wrap_arith;
//   delete[] temp_x_digits;
// }

// uint64_t lookup_msnzb(uint64_t index) {
//   uint64_t ret = 0ULL;
//   ret = floor(log2(index));
//   if (index == 0) {
//     ret = 0ULL;
//   }
//   // In the above step only at max log(64) = 6 bits are filled.
//   ret <<= 1;
//   // Last bit stores 1 if index is 0, else 0.
//   if (index == 0) {
//     ret ^= 1ULL;
//   }
//   return ret;
// }

// void AuxProtocols::msnzb_sci(uint64_t *x, uint64_t *msnzb_index, int32_t bw_x,
//                              int32_t size, int32_t digit_size) {
//   // The protocol only works when digit_size divides bw_x.
//   int32_t last_digit_size = bw_x % digit_size;
//   uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
//   uint64_t digit_mask = (digit_size == 64 ? -1 : ((1ULL << digit_size) - 1));
//   uint64_t last_digit_mask =
//       (last_digit_size == 64 ? -1 : ((1ULL << last_digit_size) - 1));
//   if (last_digit_size == 0) {
//     last_digit_mask = digit_mask;
//     last_digit_size = digit_size;
//   }
//   int32_t num_digits = ceil((bw_x * 1.0) / digit_size);
//   uint64_t *x_digits = new uint64_t[num_digits * size];

//   XTProtocol *xt = new XTProtocol(this->party, this->io, this->otpack);

//   // Extract digits
//   this->digit_decomposition_sci(size, x, x_digits, bw_x, digit_size);

//   // Use LUTs for MSNZB on digits
//   int D = (1 << digit_size);
//   int DLast = (1 << last_digit_size);
//   uint8_t *z_ = new uint8_t[num_digits * size];
//   uint64_t *msnzb_ = new uint64_t[num_digits * size];
//   uint64_t *msnzb_extended = new uint64_t[num_digits * size];
//   int lookup_output_bits = (ceil(log2(digit_size))) + 1;
//   int mux_bits = ceil(log2(bw_x));
//   uint64_t msnzb_mask = (1ULL << (lookup_output_bits - 1)) - 1;
//   uint64_t mux_mask = (1ULL << mux_bits) - 1;
//   if (party == ALICE) {
//     uint64_t **spec;
//     spec = new uint64_t *[num_digits * size];
//     PRG128 prg;
//     prg.random_data(z_, size * sizeof(uint8_t));
//     prg.random_data(msnzb_, size * sizeof(uint64_t));
//     for (int i = 0; i < (num_digits - 1) * size; i++) {
//       spec[i] = new uint64_t[D];
//       z_[i] &= 1;
//       msnzb_[i] &= msnzb_mask;
//       for (int j = 0; j < D; j++) {
//         int idx = (x_digits[i] + j) & digit_mask;
//         uint64_t lookup_val = lookup_msnzb(idx);
//         spec[i][j] = ((lookup_val >> 1) - msnzb_[i]) & msnzb_mask;
//         spec[i][j] <<= 1;
//         spec[i][j] |=
//             ((uint64_t)(((uint8_t)(lookup_val & 1ULL)) ^ z_[i]) & 1ULL);
//       }
//     }
//     for (int i = (num_digits - 1) * size; i < num_digits * size; i++) {
//       spec[i] = new uint64_t[DLast];
//       z_[i] &= 1;
//       msnzb_[i] &= msnzb_mask;
//       for (int j = 0; j < DLast; j++) {
//         int idx = (x_digits[i] + j) & last_digit_mask;
//         uint64_t lookup_val = lookup_msnzb(idx);
//         spec[i][j] = ((lookup_val >> 1) - msnzb_[i]) & msnzb_mask;
//         spec[i][j] <<= 1;
//         spec[i][j] |=
//             ((uint64_t)(((uint8_t)(lookup_val & 1ULL)) ^ z_[i]) & 1ULL);
//       }
//     }
//     if (last_digit_size == digit_size) {
//       this->lookup_table<uint64_t>(spec, nullptr, nullptr, num_digits * size,
//                                    digit_size, lookup_output_bits);
//     } else {
//       this->lookup_table<uint64_t>(spec, nullptr, nullptr,
//                                    (num_digits - 1) * size, digit_size,
//                                    lookup_output_bits);
//       this->lookup_table<uint64_t>(spec + (num_digits - 1) * size, nullptr,
//                                    nullptr, size, last_digit_size,
//                                    lookup_output_bits);
//     }

//     // Zero extend to mux_bits
//     xt->z_extend(num_digits * size, msnzb_, msnzb_extended,
//                  lookup_output_bits - 1, mux_bits);

//     for (int i = 0; i < num_digits * size; i++) {
//       delete[] spec[i];
//     }
//     delete[] spec;
//   } else {  // BOB
//     if (last_digit_size == digit_size) {
//       this->lookup_table<uint64_t>(nullptr, x_digits, msnzb_, num_digits * size,
//                                    digit_size, lookup_output_bits);
//     } else {
//       this->lookup_table<uint64_t>(nullptr, x_digits, msnzb_,
//                                    (num_digits - 1) * size, digit_size,
//                                    lookup_output_bits);
//       this->lookup_table<uint64_t>(nullptr, x_digits + (num_digits - 1) * size,
//                                    msnzb_ + (num_digits - 1) * size, size,
//                                    last_digit_size, lookup_output_bits);
//     }

//     for (int i = 0; i < (num_digits * size); i++) {
//       z_[i] = (uint8_t)(msnzb_[i] & 1ULL);
//       msnzb_[i] >>= 1;
//     }

//     // Zero extend to mux_bits
//     xt->z_extend(num_digits * size, msnzb_, msnzb_extended,
//                  lookup_output_bits - 1, mux_bits);

//     for (int i = 0; i < num_digits; i++) {
//       for (int j = 0; j < size; j++) {
//         msnzb_extended[i * size + j] += (i * digit_size);
//         msnzb_extended[i * size + j] &= mux_mask;
//       }
//     }
//   }

//   // Combine MSNZB of digits
//   uint8_t *dp_zeros_ = new uint8_t[(num_digits - 1) * size];
//   uint8_t *one_xor_zeros_ = new uint8_t[(num_digits - 1) * size];
//   uint8_t *dp_zeros_final = new uint8_t[num_digits * size];

//   if (party == ALICE) {
//     for (int i = 0; i < size; i++) {
//       dp_zeros_final[(num_digits - 1) * size + i] =
//           z_[(num_digits - 1) * size + i];
//     }
//     for (int i = 0; i < (num_digits - 1); i++) {
//       for (int j = 0; j < size; j++) {
//         one_xor_zeros_[i * size + j] = z_[i * size + j];
//       }
//     }
//   } else {
//     for (int i = 0; i < size; i++) {
//       dp_zeros_final[(num_digits - 1) * size + i] =
//           (1 ^ z_[(num_digits - 1) * size + i]);
//     }
//     for (int i = 0; i < (num_digits - 1); i++) {
//       for (int j = 0; j < size; j++) {
//         one_xor_zeros_[i * size + j] = (1 ^ z_[i * size + j]);
//       }
//     }
//   }
//   for (int i = (num_digits - 2); i >= 0; i--) {
//     if (i == (num_digits - 2)) {
//       for (int j = 0; j < size; j++) {
//         dp_zeros_[i * size + j] = z_[(i + 1) * size + j];
//       }
//     } else {
//       this->AND(dp_zeros_ + (i + 1) * size, z_ + (i + 1) * size,
//                 dp_zeros_ + i * size, size);
//     }
//   }
//   this->AND(dp_zeros_, one_xor_zeros_, dp_zeros_final, (num_digits - 1) * size);

//   uint64_t *msnzb_muxed = new uint64_t[num_digits * size];
//   this->multiplexer(dp_zeros_final, msnzb_extended, msnzb_muxed,
//                     num_digits * size, mux_bits, mux_bits);

//   for (int i = 0; i < size; i++) {
//     msnzb_index[i] = 0ULL;
//     for (int j = 0; j < num_digits; j++) {
//       msnzb_index[i] += msnzb_muxed[j * size + i];
//       msnzb_index[i] &= mux_mask;
//     }
//   }

//   delete xt;
//   delete[] x_digits;
//   delete[] z_;
//   delete[] msnzb_;
//   delete[] msnzb_extended;
//   delete[] dp_zeros_;
//   delete[] one_xor_zeros_;
//   delete[] dp_zeros_final;
//   delete[] msnzb_muxed;
//   return;
// }

// void AuxProtocols::msnzb_one_hot(uint64_t *x, uint8_t *one_hot_vector,
//                                  int32_t bw_x, int32_t size,
//                                  int32_t digit_size) {
//   uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
//   int msnzb_index_bits = ceil(log2(bw_x));
//   uint64_t msnzb_index_mask = (1ULL << msnzb_index_bits) - 1;

//   uint64_t *msnzb_index = new uint64_t[size];

//   this->msnzb_sci(x, msnzb_index, bw_x, size, digit_size);

//   // use LUT to get the one-hot representation
//   int D = 1 << msnzb_index_bits;
//   uint64_t *xor_mask = new uint64_t[size];
//   if (party == ALICE) {
//     uint64_t **spec;
//     spec = new uint64_t *[size];
//     PRG128 prg;
//     prg.random_data(one_hot_vector, size * bw_x * sizeof(uint8_t));
//     for (int i = 0; i < size; i++) {
//       for (int j = 0; j < bw_x; j++) {
//         one_hot_vector[i * bw_x + j] &= 1;
//       }
//       xor_mask[i] = 0ULL;
//       for (int j = (bw_x - 1); j >= 0; j--) {
//         xor_mask[i] <<= 1;
//         xor_mask[i] ^= (uint64_t)one_hot_vector[i * bw_x + j];
//       }
//     }
//     for (int i = 0; i < size; i++) {
//       spec[i] = new uint64_t[D];
//       for (int j = 0; j < D; j++) {
//         int idx = (msnzb_index[i] + j) & msnzb_index_mask;
//         uint64_t lookup_val = (1ULL << idx);
//         lookup_val ^= xor_mask[i];
//         spec[i][j] = lookup_val;
//       }
//     }
//     this->lookup_table<uint64_t>(spec, nullptr, nullptr, size, msnzb_index_bits,
//                                  bw_x);

//     for (int i = 0; i < size; i++) {
//       delete[] spec[i];
//     }
//     delete[] spec;
//   } else {  // BOB
//     uint64_t *temp = new uint64_t[size];
//     this->lookup_table<uint64_t>(nullptr, msnzb_index, temp, size,
//                                  msnzb_index_bits, bw_x);
//     for (int i = 0; i < size; i++) {
//       for (int j = 0; j < bw_x; j++) {
//         one_hot_vector[i * bw_x + j] = (uint8_t)(temp[i] & 1ULL);
//         temp[i] >>= 1;
//       }
//     }
//     delete[] temp;
//   }
//   delete[] msnzb_index;
// }

// template void AuxProtocols::lookup_table(uint64_t **spec, uint64_t *x,
//                                          uint64_t *y, int32_t size,
//                                          int32_t bw_x, int32_t bw_y);
// template void AuxProtocols::lookup_table(uint8_t **spec, uint8_t *x, uint8_t *y,
//                                          int32_t size, int32_t bw_x,
//                                          int32_t bw_y);
} // namespace OTProtocol