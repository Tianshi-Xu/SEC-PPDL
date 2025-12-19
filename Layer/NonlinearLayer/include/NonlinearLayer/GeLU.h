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
class GeLU : public Module{
    public:
      int bitwidth;
      int scale;
      int party;
      bool signed_arithmetic;
      double coe[5] = {0.020848611754127593, -0.18352506127082727, 0.5410550166368381, -0.03798164612714154, 0.001620808531841547};
      int64_t coe_fix[5];
      GeLU(FixPoint<T> *fixPoint,HE::HEEvaluator* HE, int bitwidth, int scale, bool signed_arithmetic=true){
        this->fixPoint = fixPoint;
        this->bitwidth = bitwidth;
        this->scale = scale;
        this->HE = HE;
        this->party = fixPoint->party;
        this->signed_arithmetic = signed_arithmetic;
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
        cout << "bitwidth, scale, plain_mod:" << bitwidth << " " << scale << " " << HE->plain_mod << endl;
        Tensor<T> x_ring = x;
        // check_share(x, 1ULL << (bitwidth), "x_ring before");
        // cout << "bitwidth, scale, plain_mod:" << bitwidth << " " << scale << " " << HE->plain_mod << endl;
        fixPoint->Ring2Field(x, HE->plain_mod, bitwidth, signed_arithmetic);
        // check_share(x, HE->plain_mod, "x_field");
        auto x_2 = ElementWiseMul(x, x, HE);

        // check_share(x_2, HE->plain_mod, "x_2_field");

        fixPoint->Field2Ring(x_2, HE->plain_mod, bitwidth+scale, signed_arithmetic);

        // check_share(x_2, 1ULL << (bitwidth+scale), "x_2_ring");


        fixPoint->truncate_reduce(x_2,scale,bitwidth+scale);
        // check_share(x_2, 1ULL << (bitwidth), "x_2 after truncate");
        // cout << "OK4" << endl;
        Tensor<T> x_2_ring = x_2;
        fixPoint->Ring2Field(x_2, HE->plain_mod, bitwidth, signed_arithmetic);
        // check_share(x_2, HE->plain_mod, "x_2_field after truncate");

        auto x_3 = ElementWiseMul(x_2, x, HE);
        // check_share(x_3, HE->plain_mod, "x_3");
        fixPoint->Field2Ring(x_3, HE->plain_mod, bitwidth+scale, signed_arithmetic);
        // check_share(x_3, 1ULL << (bitwidth+scale), "x_3_ring");
        fixPoint->truncate_reduce(x_3,scale,bitwidth+scale);
        // check_share(x_3, 1ULL << (bitwidth), "x_3 after truncate");
        auto x_4 = ElementWiseMul(x_2, x_2, HE);
        // check_share(x_4, HE->plain_mod, "x_4");
        fixPoint->Field2Ring(x_4, HE->plain_mod, bitwidth+scale, signed_arithmetic);
        // check_share(x_4, 1ULL << (bitwidth+scale), "x_4_ring");
        fixPoint->truncate_reduce(x_4,scale,bitwidth+scale);
        // check_share(x_4, 1ULL << (bitwidth), "x_4 after truncate");
        Tensor<T> F0(x.shape());
        Tensor<T> F1(x.shape());
        for(size_t i = 0; i < x.size(); i++){
          F0(i) = x_4(i)*coe_fix[0]-x_3(i)*coe_fix[1]+x_2_ring(i)*coe_fix[2]+x_ring(i)*(round(0.5* (1ULL << scale))-coe_fix[3])+coe_fix[4];
          F1(i) = x_4(i)*coe_fix[0]+x_3(i)*coe_fix[1]+x_2_ring(i)*coe_fix[2]+x_ring(i)*(round(0.5* (1ULL << scale))+coe_fix[3])+coe_fix[4];
        }
        // check_share(F1, 1ULL << (bitwidth), "F1");
        fixPoint->truncate(F0,scale,bitwidth);
        fixPoint->truncate(F1,scale,bitwidth);
        // cout << "OK12" << endl;
        Tensor<uint8_t> b0(x.shape()), b1(x.shape()), b2(x.shape());
        fixPoint->less_than_constant(x_ring, -2.7* (1ULL << scale), b0, bitwidth);
        fixPoint->less_than_constant(x_ring, 0.0* (1ULL << scale), b1, bitwidth);
        fixPoint->less_than_constant(2.7* (1ULL << scale), x_ring, b2, bitwidth);
        // cout << "OK13" << endl;
        Tensor<uint8_t> z0(x.shape()), z1(x.shape());
        Tensor<uint8_t> z2 = b2;
        for(size_t i = 0; i < x.size(); i++){
          z0(i) = b0(i) ^ b1(i);
          z1(i) = b1(i) ^ b2(i)^(party-1);
        }
        fixPoint->mux(z0, F0, F0, bitwidth, bitwidth);
        fixPoint->mux(z1, F1, F1, bitwidth, bitwidth);
        fixPoint->mux(z2, x_ring, x_ring, bitwidth, bitwidth);
        // check_share(F0, 1ULL << (bitwidth), "F0 after truncate");
        // check_share(F1, 1ULL << (bitwidth), "F1 after truncate");
        // check_share(x_ring, 1ULL << (bitwidth), "x_ring");
        for(size_t i = 0; i < x.size(); i++){
          x(i) = F0(i) + F1(i) + x_ring(i);
        }
        // check_share(x, 1ULL << (bitwidth), "final result");
      }
      
    private:
      NonlinearOperator::FixPoint<T> *fixPoint;
      HE::HEEvaluator* HE;
};

}
