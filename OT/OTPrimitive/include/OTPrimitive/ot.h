#pragma once
#include <Utils/emp-tool.h>
using namespace Utils;

namespace OTPrimitive {

template <typename IO> 
class OT {
public:
  int party;
  OT() {}
  virtual ~OT() {}
  virtual void send(const block128 *data0, const block128 *data1, int length){Utils::error("send not implemented");}
  virtual void recv(block128 *data, const bool *b, int length){Utils::error("recv not implemented");}
  virtual void send(const block256 *data0, const block256 *data1, int length){Utils::error("send not implemented");}
  virtual void recv(block256 *data, const bool *b, int length){Utils::error("recv not implemented");}
  virtual void send(block128 **data, int length, int N){Utils::error("send not implemented");}
  virtual void recv(block128 *data, const uint8_t *b, int length, int N){Utils::error("recv not implemented");}
  virtual void send(uint8_t **data, int length, int N, int l){Utils::error("send not implemented");}
  virtual void recv(uint8_t *data, const uint8_t *b, int length, int N, int l){Utils::error("recv not implemented");}
  virtual void recv(uint8_t *data, uint8_t *b, int length, int N, int l){Utils::error("recv not implemented");}
  virtual void send(uint8_t **data, int length, int l){Utils::error("send not implemented");}
  virtual void recv(uint8_t *data, const uint8_t *b, int length, int l){Utils::error("recv not implemented");}
  
  virtual void recv(uint8_t *data, uint8_t *b, int length, int l){
    Utils::error("recv not implemented");
  }
  virtual void send(uint64_t **data, int length, int l){Utils::error("send not implemented");}
  virtual void recv(uint64_t *data, const uint8_t *b, int length, int l){Utils::error("recv not implemented");}
  virtual void recv(uint64_t *data, uint8_t *b, int length, int l){Utils::error("recv not implemented");}

  virtual void send_cot(uint64_t *data0, uint64_t *corr, int length, int l){Utils::error("send_cot not implemented");}
  virtual void recv_cot(uint64_t *data, bool *b, int length, int l){Utils::error("recv_cot not implemented");}

  template <typename intType>
  void send_cot_matmul(intType *rdata, const intType *corr,
                       const uint32_t *chunkSizes, const uint32_t *numChunks,
                       const int numOTs, int senderMatmulDims){Utils::error("send_cot_matmul not implemented");}

  template <typename intType>
  void recv_cot_matmul(intType *data, const uint8_t *choices,
                       const uint32_t *chunkSizes, const uint32_t *numChunks,
                       const int numOTs, int senderMatmulDims){Utils::error("recv_cot_matmul not implemented");}

  virtual void send(uint8_t **data, int length, int N, int l, bool type){Utils::error("send not implemented");}
  virtual void recv(uint8_t *data, const uint8_t *b, int length, int N, int l,
            bool type){Utils::error("recv not implemented");}
  virtual void recv(uint8_t *data, uint8_t *b, int length, int N, int l, bool type){Utils::error("recv not implemented");}
  virtual void send(uint8_t **data, int length, int l, bool type){Utils::error("send not implemented");}
  virtual void recv(uint8_t *data, const uint8_t *b, int length, int l, bool type){Utils::error("recv not implemented");}
  virtual void recv(uint8_t *data, uint8_t *b, int length, int l, bool type){Utils::error("recv not implemented");}


  virtual void setup_send(block128 *in_k0 = nullptr, bool *in_s = nullptr){Utils::error("setup_send not implemented");}
  virtual void setup_send(bool is256, block256 *in_k0 = nullptr, bool *in_s = nullptr){Utils::error("setup_send not implemented");}
  virtual void setup_recv(block128 *in_k0 = nullptr, block128 *in_k1 = nullptr){Utils::error("setup_recv not implemented");}
  virtual void setup_recv(bool is256, block256 *in_k0 = nullptr, block256 *in_k1 = nullptr){Utils::error("setup_recv not implemented");}
};
} // namespace OTPrimitive
