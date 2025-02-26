#include "millionaire.h"
#include "millionaire_with_equality.h"
#include <OTPrimitive/ot_primitive.h>
#pragma once

using namespace OTPrimitive;
using namespace Utils;
using namespace OTProtocol;
namespace OTProtocol {
class AuxProtocols {
public:
  int party;
  Utils::NetIO *io;
  OTPrimitive::OTPack<Utils::NetIO> *otpack;
  OTProtocol::MillionaireProtocol<Utils::NetIO> *mill;
  OTProtocol::MillionaireWithEquality<Utils::NetIO> *mill_and_eq;

  AuxProtocols(int party, Utils::NetIO *io, OTPrimitive::OTPack<Utils::NetIO> *otpack);

  ~AuxProtocols();

  void wrap_computation(
      // input vector
      uint64_t *x,
      // wrap-bit of shares of x
      uint8_t *y,
      // size of input vector
      int32_t size,
      // bitwidth of x
      int32_t bw_x);

  // y = sel * x
  template <typename T>
  void multiplexer(
      // selection bits
      uint8_t *sel,
      // input vector
      T *x,
      // output vector
      T *y,
      // size of vectors
      int32_t size,
      // bitwidth of x
      int32_t bw_x,
      // bitwidth of y
      int32_t bw_y)
    {
        assert(bw_x <= 64 && bw_y <= 64 && bw_y <= bw_x);
        uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
        uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

        uint64_t *corr_data = new uint64_t[size];
        uint64_t *data_S = new uint64_t[size];
        uint64_t *data_R = new uint64_t[size];

        // y = (sel_0 \xor sel_1) * (x_0 + x_1)
        // y = (sel_0 + sel_1 - 2*sel_0*sel_1)*x_0 + (sel_0 + sel_1 -
        // 2*sel_0*sel_1)*x_1 y = [sel_0*x_0 + sel_1*(x_0 - 2*sel_0*x_0)]
        //     + [sel_1*x_1 + sel_0*(x_1 - 2*sel_1*x_1)]
        for (int i = 0; i < size; i++) {
            corr_data[i] = (x[i] * (1 - 2 * uint64_t(sel[i]))) & mask_y;
        }
        if (party == Utils::ALICE) {
            otpack->iknp_straight->send_cot(data_S, corr_data, size, bw_y);
            otpack->iknp_reversed->recv_cot(data_R, (bool *)sel, size, bw_y);
        } else {  // party == BOB
            otpack->iknp_straight->recv_cot(data_R, (bool *)sel, size, bw_y);
            otpack->iknp_reversed->send_cot(data_S, corr_data, size, bw_y);
        }
        for (int i = 0; i < size; i++) {
            y[i] = ((x[i] * uint64_t(sel[i]) + data_R[i] - data_S[i]) & mask_y);
        }

        delete[] corr_data;
        delete[] data_S;
        delete[] data_R;
    }

  // Boolean to Arithmetic Shares
  void B2A(
      // input (boolean) vector
      uint8_t *x,
      // output vector
      uint64_t *y,
      // size of vector
      int32_t size,
      // bitwidth of y
      int32_t bw_y);

  template <typename T>
  void lookup_table(
      // table specification
      T **spec,
      // input vector
      T *x,
      // output vector
      T *y,
      // size of vector
      int32_t size,
      // bitwidth of input to LUT
      int32_t bw_x,
      // bitwidth of output of LUT
      int32_t bw_y);

  // MSB computation
  template <typename T>
  void MSB(
      // input vector
      T *x,
      // shares of MSB(x)
      uint8_t *msb_x,
      // size of input vector
      int32_t size,
      // bitwidth of x
      int32_t bw_x)
    {
        assert(bw_x <= 64);
        int32_t shift = bw_x - 1;
        uint64_t shift_mask = (shift == 64 ? -1 : ((1ULL << shift) - 1));

        uint64_t *tmp_x = new uint64_t[size];
        uint8_t *msb_xb = new uint8_t[size];
        for (int i = 0; i < size; i++) {
            tmp_x[i] = x[i] & shift_mask;
            msb_xb[i] = (x[i] >> shift) & 1;
            if (party == Utils::BOB) tmp_x[i] = (shift_mask - tmp_x[i]) & shift_mask;
        }

        mill->compare(msb_x, tmp_x, size, bw_x - 1, true);  // computing greater_than

        for (int i = 0; i < size; i++) {
            // cout << "msb_x[" << i << "] = " << msb_x[i] << endl;
            // cout << "msb_xb[" << i << "] = " << msb_xb[i] << endl;
            msb_x[i] = msb_x[i] ^ msb_xb[i];
        }

        delete[] tmp_x;
        delete[] msb_xb;
    }

//   // MSB to Wrap computation
  void MSB_to_Wrap(
      // input vector
      uint64_t *x,
      // shares of MSB(x)
      uint8_t *msb_x,
      // output shares of Wrap(x)
      uint8_t *wrap_x,
      // size of input vector
      int32_t size,
      // bitwidth of x
      int32_t bw_x);

//   // Simple MSB to Wrap computation
//   void msb0_to_wrap(
//       // input vector
//       uint64_t *x,
//       // output shares of Wrap(x)
//       uint8_t *wrap_x,
//       // size of input vector
//       int32_t size,
//       // bitwidth of x
//       int32_t bw_x);

//     // Simple MSB to Wrap computation
//   void msb1_to_wrap(
//       // input vector
//       uint64_t *x,
//       // output shares of Wrap(x)
//       uint8_t *wrap_x,
//       // size of input vector
//       int32_t size,
//       // bitwidth of x
//       int32_t bw_x);

//   // Bitwise AND
  void AND(
      // input A (boolean) vector
      uint8_t *x,
      // input B (boolean) vector
      uint8_t *y,
      // output vector
      uint8_t *z,
      // size of vector
      int32_t size);

//   void digit_decomposition(int32_t dim, uint64_t *x, uint64_t *x_digits,
//                            int32_t bw_x, int32_t digit_size);

//   void digit_decomposition_sci(
//       int32_t dim, uint64_t *x, uint64_t *x_digits, int32_t bw_x,
//       int32_t digit_size,
//       // set true if the bitlength of all output digits is digit_size
//       // leave false, if the last digit is output over <= digit_size bits
//       bool all_digit_size = false);

//   void reduce(int32_t dim, uint64_t *x, uint64_t *y, int32_t bw_x,
//               int32_t bw_y);

//   // Make x positive: pos_x = x * (1 - 2*MSB(x))
//   void make_positive(
//       // input vector
//       uint64_t *x,
//       // input (boolean) vector containing MSB(x)
//       uint8_t *msb_x,
//       // output vector
//       uint64_t *pos_x,
//       // size of vector
//       int32_t size);

//   // Outputs index and not one-hot vector
//   void msnzb_sci(uint64_t *x, uint64_t *msnzb_index, int32_t bw_x, int32_t size,
//                  int32_t digit_size = 8);

//   // Wrapper over msnzb_sci. Outputs one-hot vector
//   void msnzb_one_hot(uint64_t *x,
//                      // size: bw_x * size
//                      uint8_t *one_hot_vector, int32_t bw_x, int32_t size,
//                      int32_t digit_size = 8);
};
} // namespace OTProtocol
