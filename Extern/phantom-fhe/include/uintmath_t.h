#pragma once

#include <cstdint>

namespace phantom::arith {
typedef struct uint128_t {
  uint64_t hi;
  uint64_t lo;
  // TODO: implement uint128_t basic operations
  //    __device__ uint128_t &operator+=(const uint128_t &op);
} uint128_t;

struct uint128_t2 {
  uint128_t x;
  uint128_t y;
};

struct uint128_t4 {
  uint128_t x;
  uint128_t y;
  uint128_t z;
  uint128_t w;
};

struct double_t2 {
  double x;
  double y;
};

struct double_t4 {
  double x;
  double y;
  double z;
  double w;
};
} // namespace phantom::arith