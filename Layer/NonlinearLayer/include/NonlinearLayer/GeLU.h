#include <Datatype/Tensor.h>
#include "../../../Layer/Module.h"
#include <OTProtocol/aux-protocols.h>
#include <OTProtocol/millionaire.h>
#pragma once
using namespace Datatype;
using namespace Utils;

extern int32_t bitlength;
extern int32_t kScale;

#define RING 0
#define OFF_PLACE

namespace NonlinearLayer{

template <typename T, typename IO> class ReLUProtocol {
  public:
  virtual void relu(T *result, T *share, int num_relu,
            uint8_t *msb = nullptr, bool skip_ot = false) = 0;
};


template <typename T, typename IO> 
class ReLURingProtocol : public ReLUProtocol<T, IO> {
public:
  IO *io = nullptr;
  OTPrimitive::OTPack<IO> *otpack;
  OTProtocol::TripleGenerator<IO> *triple_gen = nullptr;
  OTProtocol::MillionaireProtocol<IO> *millionaire = nullptr;
  OTProtocol::AuxProtocols *aux = nullptr;
  int party;
  int algeb_str;
  int l, b;
  int num_cmps;

  // Constructor
  ReLURingProtocol(int party, IO *io, int l, int b,
                   OTPack<IO> *otpack, OT_TYPE ot_type = Datatype::IKNP) {
    this->party = party;
    this->io = io;
    this->l = l;
    this->b = b;
    this->otpack = otpack;
    this->millionaire = new MillionaireProtocol<IO>(party, io, otpack,l,b,ot_type);
    this->triple_gen = this->millionaire->triple_gen;
    this->aux = new AuxProtocols(party, io, otpack);
  }

  // Destructor
  virtual ~ReLURingProtocol() { delete millionaire; }

  void relu(T *result, T *share, int num_relu,
                uint8_t *msb, bool skip_ot) {
        uint8_t *msb_tmp = new uint8_t[num_relu];
        if(msb!=nullptr){
            memcpy(msb_tmp,msb,num_relu*sizeof(uint8_t));
        }
        else{
            this->aux->MSB<T>(share, msb_tmp, num_relu, this->l);
        }
        // std::cout << "MSB done" << std::endl;
        for (int i = 0; i < num_relu; i++) {
            if (this->party == ALICE) {
                msb_tmp[i] = msb_tmp[i] ^ 1;
            }
        }
        this->aux->multiplexer<T>(msb_tmp, share, result, num_relu, this->l,
                        this->l);
        delete[] msb_tmp;
        return;
    }
};

template <typename T, typename IO=Utils::NetIO>
class ReLU : public Module{
    public:
      int bitwidth;
      int num_threads;
      ReLU(ReLUProtocol<T, IO> **reluprotocol,int bitwidth=32, int num_threads=4){
        this->bitwidth = bitwidth;
        this->num_threads = num_threads;
        this->reluProtocol = reluprotocol;
      }

      void operator()(Tensor<T> &x){
        auto shape = x.shape();
        int dim = x.size();
        x.flatten();
        T* x_flatten = x.data().data();
        std::thread relu_threads[num_threads];
        int chunk_size = dim / num_threads;
        for (int i = 0; i < num_threads; ++i) {
            int offset = i * chunk_size;
            int lnum_ops;
            if (i == (num_threads - 1)) {
                lnum_ops = dim - offset;
            } else {
                lnum_ops = chunk_size;
            }
            relu_threads[i] =
                std::thread(relu_thread, reluProtocol[i], x_flatten+offset, x_flatten+offset, lnum_ops);
        }
        for (int i = 0; i < num_threads; ++i) {
            relu_threads[i].join();
        }
        x.reshape(shape);
      }
      
    private:
      ReLUProtocol<T, IO>** reluProtocol=nullptr;
      void static relu_thread(ReLUProtocol<T, IO>* reluProtocol, T* result, T* input, int lnum_ops){
        reluProtocol->relu(result, input, lnum_ops);
      }
};

}