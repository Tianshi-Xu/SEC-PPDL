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
        auto x_2 = ElementWiseMul(x, x, HE);
        fixPoint->truncate_reduce(x_2,scale,bitwidth+scale);
        auto x_3 = ElementWiseMul(x_2, x, HE);
        fixPoint->truncate_reduce(x_3,scale,bitwidth+scale);
        auto x_4 = ElementWiseMul(x_2, x_2, HE);
        fixPoint->truncate_reduce(x_4,scale,bitwidth+scale);
      }
      
    private:
      FixPointProtocol<T> *fixPoint;
      HE::HEEvaluator* HE;
};

}