/*
Original Work Copyright (c) 2018 Xiao Wang (wangxiao@gmail.com)
Modified Work Copyright (c) 2020 Microsoft Research

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

Enquiries about further applications and development opportunities are welcome.

Modified by Deevashwer Rathee, Nishant Kumar, Mayank Rathee
*/

#pragma once
#include <Utils/emp-tool.h>
using namespace Utils;

namespace OTPrimitive {

template <typename IO> 
class OT {
public:
  OT() {}
  virtual ~OT() {}
  virtual void send(const block128 *data0, const block128 *data1, int length){}
  virtual void recv(block128 *data, const bool *b, int length){}
  virtual void send(const block256 *data0, const block256 *data1, int length){}
  virtual void recv(block256 *data, const bool *b, int length){}
  virtual void send(block128 **data, int length, int N){}
  virtual void recv(block128 *data, const uint8_t *b, int length, int N){}
  virtual void send(uint8_t **data, int length, int N, int l){}
  virtual void recv(uint8_t *data, const uint8_t *b, int length, int N, int l){}
  virtual void recv(uint8_t *data, uint8_t *b, int length, int N, int l){}
  virtual void send(uint8_t **data, int length, int l){}
  virtual void recv(uint8_t *data, const uint8_t *b, int length, int l){}
  virtual void recv(uint8_t *data, uint8_t *b, int length, int l){}
  virtual void send(uint64_t **data, int length, int l){}
  virtual void recv(uint64_t *data, const uint8_t *b, int length, int l){}
  virtual void recv(uint64_t *data, uint8_t *b, int length, int l){}

  virtual void send_cot(uint64_t *data0, uint64_t *corr, int length, int l){}
  virtual void recv_cot(uint64_t *data, bool *b, int length, int l){}

  template <typename intType>
  void send_cot_matmul(intType *rdata, const intType *corr,
                       const uint32_t *chunkSizes, const uint32_t *numChunks,
                       const int numOTs, int senderMatmulDims){}

  template <typename intType>
  void recv_cot_matmul(intType *data, const uint8_t *choices,
                       const uint32_t *chunkSizes, const uint32_t *numChunks,
                       const int numOTs, int senderMatmulDims){}

  virtual void send(uint8_t **data, int length, int N, int l, bool type){}
  virtual void recv(uint8_t *data, const uint8_t *b, int length, int N, int l,
            bool type){}
  virtual void recv(uint8_t *data, uint8_t *b, int length, int N, int l, bool type){}
  virtual void send(uint8_t **data, int length, int l, bool type){}
  virtual void recv(uint8_t *data, const uint8_t *b, int length, int l, bool type){}
  virtual void recv(uint8_t *data, uint8_t *b, int length, int l, bool type){}


  virtual void setup_send(block128 *in_k0 = nullptr, bool *in_s = nullptr){}
  virtual void setup_send(bool is256, block256 *in_k0 = nullptr, bool *in_s = nullptr){}
  virtual void setup_recv(block128 *in_k0 = nullptr, block128 *in_k1 = nullptr){}
  virtual void setup_recv(bool is256, block256 *in_k0 = nullptr, block256 *in_k1 = nullptr){}
};
} // namespace OTPrimitive