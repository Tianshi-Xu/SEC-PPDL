#include <Datatype/Tensor.h>
#include "../../../Layer/Module.h"
#include <NonlinearOperator/FixPoint.h>
#include <LinearOperator/Polynomial.h>
#pragma once
using namespace Datatype;
using namespace LinearOperator;
extern int32_t bitlength;
extern int32_t kScale;

#define RING 0
#define OFF_PLACE

namespace NonlinearLayer{

template <typename T, typename IO=Utils::NetIO>
class GeLU : public Module{
    public:
      int bitwidth;
      int scale;
      double coe[5] = {0.020848611754127593, -0.18352506127082727, 0.5410550166368381, -0.03798164612714154, 0.001620808531841547};
      uint64_t coe_fix[5];
      GeLU(FixPointProtocol<T> *fixPoint,HE::HEEvaluator* HE, int bitwidth, int scale){
        this->fixPoint = fixPoint;
        this->bitwidth = bitwidth;
        this->scale = scale;
        this->HE = HE;
        for(int i = 0; i < 5; i++){
          coe_fix[i] = (uint64_t)round(coe[i] * (1ULL << scale));
        }
      }
      // only support ring
      // TODO: support field
      void operator()(Tensor<T> &x){
        Tensor<T> x_ring = x;
        fixPoint->Ring2Field(x, HE->plain_mod, bitwidth);
        auto x_2 = ElementWiseMul(x, x, HE);
        fixPoint->Field2Ring(x_2, HE->plain_mod, bitwidth+scale);
        fixPoint->truncate_reduce(x_2,scale,bitwidth+scale);
        Tensor<T> x_2_ring = x_2;
        fixPoint->Ring2Field(x_2, HE->plain_mod, bitwidth+scale);
        auto x_3 = ElementWiseMul(x_2, x, HE);
        fixPoint->Field2Ring(x_3, HE->plain_mod, bitwidth+scale);
        fixPoint->truncate_reduce(x_3,scale,bitwidth+scale);
        auto x_4 = ElementWiseMul(x_2, x_2, HE);
        fixPoint->Field2Ring(x_4, HE->plain_mod, bitwidth+scale);
        fixPoint->truncate_reduce(x_4,scale,bitwidth+scale);
        Tensor<T> F1(x.shape());
        Tensor<T> F2(x.shape());
        for(size_t i = 0; i < x.size(); i++){
          F1(i) = x_4(i)*coe_fix[0]+x_3(i)*coe_fix[1]+x_2_ring(i)*coe_fix[2]+x_ring(i)*(0.5* (1ULL << scale)-coe_fix[3])+coe_fix[4];
          F2(i) = x_4(i)*coe_fix[0]+x_3(i)*coe_fix[1]+x_2_ring(i)*coe_fix[2]+x_ring(i)*(0.5* (1ULL << scale)+coe_fix[3])+coe_fix[4];
        }
        
      }
      
    private:
      FixPointProtocol<T> *fixPoint;
      HE::HEEvaluator* HE;
};

}