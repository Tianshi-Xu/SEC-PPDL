#include <Datatype/Tensor.h>
#include "../../../Layer/Module.h"
#include <NonlinearOperator/FixPoint.h>
#include <LinearOperator/Polynomial.h>
#pragma once
using namespace Datatype;
using namespace LinearOperator;
using namespace NonlinearOperator;
extern int32_t bitlength;
extern int32_t kScale;

#define RING 0
#define OFF_PLACE

namespace NonlinearLayer{
template <typename T, typename IO=Utils::NetIO>
class Softmax : public Module{
    public:
      int bitwidth;
      int scale;
      int party;
      double coe[5] = {0.020848611754127593, -0.18352506127082727, 0.5410550166368381, -0.03798164612714154, 0.001620808531841547};
      int64_t coe_fix[5];
      Softmax(FixPoint<T> *fixPoint,HE::HEEvaluator* HE, int bitwidth, int scale){
        this->fixPoint = fixPoint;
        this->bitwidth = bitwidth;
        this->scale = scale;
        this->HE = HE;
        this->party = fixPoint->party;
        for(int i = 0; i < 5; i++){
          coe_fix[i] = (int64_t)round(coe[i] * (1ULL << scale));
        }
      }
      void check_share(Tensor<uint64_t> &x, uint64_t mod, string name){
        if (party == ALICE){
          HE->IO->send_tensor(x);
        }
        else{
          Tensor<uint64_t> x0(x.shape());
          HE->IO->recv_tensor(x0);
          x0 = x0 + x;
          for(size_t i = 0; i < x.size(); i++){
            x0(i) = x0(i) % mod;
          }
          cout << "check " << name << ":" << endl;
          for (size_t i = 2040; i < 2060; i++){
            cout << x0(i) << " ";
          }
        }

      }
      // only support ring
      // TODO: support field
      void operator()(Tensor<T> &x){
        
      }
      
    private:
      NonlinearOperator::FixPoint<T> *fixPoint;
      HE::HEEvaluator* HE;
};

}
