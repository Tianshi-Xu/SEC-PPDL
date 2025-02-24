#include <Datatype/Tensor.h>
#include "../../../Layer/Module.h"
#include <OTProtocol/aux-protocols.h>
#include <OTProtocol/millionaire.h>

using namespace Datatype;
using namespace Utils;

extern int32_t bitlength;
extern int32_t kScale;

#define RING 0
#define OFF_PLACE

namespace NonlinearLayer{

template <typename IO, typename T> class ReLUProtocol {
  public:
  virtual void relu(T *result, T *share, int num_relu,
            uint8_t *msb = nullptr, bool skip_ot = false) = 0;
};


template <typename IO, typename T>
class ReLURingProtocol : public ReLUProtocol<IO, T> {
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
  uint8_t two_small = 1 << 1;
  uint8_t zero_small = 0;
  uint64_t mask_take_32 = -1;
  uint64_t msb_one;
  uint64_t cut_mask;
  uint64_t relu_comparison_rhs;
  T mask_l;
  T relu_comparison_rhs_type;
  T cut_mask_type;
  T msb_one_type;

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
    // configure();
  }

  // Destructor
  virtual ~ReLURingProtocol() { delete millionaire; }

  void configure() {
    if (this->l != 32 && this->l != 64) {
      mask_l = (T)((1ULL << l) - 1);
    } else if (this->l == 32) {
      mask_l = -1;
    } else { // l = 64
      mask_l = -1ULL;
    }
    if (sizeof(T) == sizeof(uint64_t)) {
      msb_one = (1ULL << (this->l - 1));
      relu_comparison_rhs_type = msb_one - 1ULL;
      relu_comparison_rhs = relu_comparison_rhs_type;
      cut_mask_type = relu_comparison_rhs_type;
      cut_mask = cut_mask_type;
    } else {
      msb_one_type = (1 << (this->l - 1));
      relu_comparison_rhs_type = msb_one_type - 1;
      relu_comparison_rhs = relu_comparison_rhs_type + 0ULL;
      cut_mask_type = relu_comparison_rhs_type;
      cut_mask = cut_mask_type + 0ULL;
    }
  }

  void relu(T *result, T *share, int num_relu,
                uint8_t *msb, bool skip_ot) {
        uint8_t *msb_tmp = new uint8_t[num_relu];
        if(msb!=nullptr){
            memcpy(msb_tmp,msb,num_relu*sizeof(uint8_t));
        }
        else{
            this->aux->MSB<T>(share, msb_tmp, num_relu, this->l);
        }
        for (int i = 0; i < num_relu; i++) {
            if (party == ALICE) {
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
      ReLU(ReLUProtocol<IO, T> **reluprotocol,int bitwidth=32, int num_threads=4){
        this->bitwidth = bitwidth;
        this->num_threads = num_threads;
        this->reluProtocol = reluprotocol;
      }

      void operator()(Tensor<T> *x){
        int dim = x->size();
        x->flatten();
        T* x_flatten = x->data().data();
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
      }
      
    private:
      ReLUProtocol<IO, T>** reluProtocol=nullptr;
      void static relu_thread(ReLUProtocol<IO, T>* reluProtocol, T* result, T* input, int lnum_ops){
        reluProtocol->relu(result, input, lnum_ops);
      }
};

}