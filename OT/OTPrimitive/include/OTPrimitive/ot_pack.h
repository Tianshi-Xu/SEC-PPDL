/*
Authors: Mayank Rathee, Deevashwer Rathee
Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
#define KKOT_TYPES 8
// #include "cf2_ot_pack.h"
#include "split_kkot.h"
#include "split_iknp.h"
namespace OTPrimitive {
template <typename IO>
class OTPack {
 public:
  OT<IO> *kkot[KKOT_TYPES];

  // iknp_straight and iknp_reversed: party
  // acts as sender in straight and receiver in reversed.
  // Needed for MUX calls.
  OT<IO> *iknp_straight;
  OT<IO> *iknp_reversed;
  IO *io;
  int party;
  bool do_setup = false;

  OTPack(IO *io, int party, bool do_setup = true) {
  };

  ~OTPack() {
  };

  void SetupBaseOTs() {};

  /*
   * DISCLAIMER:
   * OTPack copy method avoids computing setup keys for each OT instance by
   * reusing the keys generated (through base OTs) for another OT instance.
   * Ideally, the PRGs within OT instances, using the same keys, should use
   * mutually exclusive counters for security. However, the current
   * implementation does not support this.
   */

  void copy(OTPack<IO> *copy_from) {};
};

}  // namespace OTPrimitive