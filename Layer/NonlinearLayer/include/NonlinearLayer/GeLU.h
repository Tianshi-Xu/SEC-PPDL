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
      // F0 = E + L - O, F1 = E + L + O
      // E = (c0 * x^2 + c2) * x^2 + c4, O = (c1 * x^2 + c3) * x, L(x) = 0.5*x
      double coe[5] = {0.020848611754127593, -0.18352506127082727, 0.5410550166368381, -0.03798164612714154, 0.001620808531841547};
      int64_t coe_fix[5];
      GeLU(FixPoint<T> *fixPoint,HE::HEEvaluator* HE, int bitwidth, int scale){
        this->fixPoint = fixPoint;
        this->bitwidth = bitwidth;
        this->scale = scale;
        this->HE = HE;
        this->party = fixPoint->party;
        for(int i = 0; i < 5; i++){
          coe_fix[i] = (int64_t)round(coe[i] * (1ULL << scale));
        }
      }
      void check_share(const Tensor<T> &x, uint64_t mod, const string &name){
        if (party == ALICE){
          HE->IO->send_tensor(x);
          return;
        }
        Tensor<T> peer_share(x.shape());
        HE->IO->recv_tensor(peer_share);
        if (mod == 0){
          cout << "check " << name << " skipped (mod is zero)" << endl;
          return;
        }
        cout << "check " << name << ":" << endl;
        size_t start = x.size() <= 2040 ? 0 : 2040;
        size_t end = 2041;
        if (end > x.size()){
          end = x.size();
        }
        for (size_t i = start; i < end; i++){
          uint64_t raw = (static_cast<uint64_t>(peer_share(i)) + static_cast<uint64_t>(x(i))) % mod;
          int64_t signed_val = (raw >= (mod >> 1)) ? static_cast<int64_t>(raw - mod) : static_cast<int64_t>(raw);
          cout << signed_val << " ";
          if(i==start){
            cout << "raw: " << raw << " mod: " << mod << endl;
          }
        }
        cout << endl;

      }
      // only support ring
      // TODO: support field
      void operator()(Tensor<T> &x){
        cout << "coe_fix: ";
        for(int i = 0; i < 5; i++){
          cout << coe_fix[i] << " ";
        }
        cout << endl;
        cout << "bitwidth, scale, plain_mod:" << bitwidth << " " << scale << " " << HE->plain_mod << endl;
        Tensor<T> x_ring = x;
        // check_share(x, 1ULL << (bitwidth), "x_ring before");
        // cout << "bitwidth, scale, plain_mod:" << bitwidth << " " << scale << " " << HE->plain_mod << endl;
        fixPoint->Ring2Field(x, HE->plain_mod, bitwidth);
        // check_share(x, HE->plain_mod, "x_field");
        auto x_2 = ElementWiseMul(x, x, HE);

        // check_share(x_2, HE->plain_mod, "x_2_field");

        fixPoint->Field2Ring(x_2, HE->plain_mod, bitwidth+scale);

        // check_share(x_2, 1ULL << (bitwidth+scale), "x_2_ring");


        fixPoint->truncate(x_2,scale,bitwidth+scale,true);
        // check_share(x_2, 1ULL << (bitwidth+scale), "x_2 after truncate");
        // 使用秦九韶折叠的奇偶分解法计算多项式
        // F0 = E + L - O, F1 = E + L + O
        // E = (c0 * x^2 + c2) * x^2 + c4, O = (c1 * x^2 + c3) * x, L(x) = 0.5*x
        
        // 预计算: 同时计算 c0*x^2 和 c1*x^2 (ring上本地计算)
        Tensor<T> tmp_even(x.shape());
        Tensor<T> tmp_odd(x.shape());
        for(size_t i = 0; i < x.size(); i++){
          tmp_even(i) = x_2(i) * coe_fix[0];
          tmp_odd(i) = x_2(i) * coe_fix[1];
        }
        // check_share(tmp_even, 1ULL << (bitwidth+scale), "tmp_even before truncate");
        // check_share(tmp_odd, 1ULL << (bitwidth+scale), "tmp_odd before truncate");
        // 同时truncate
        fixPoint->truncate_reduce(tmp_even, scale, bitwidth+scale);
        fixPoint->truncate_reduce(tmp_odd, scale, bitwidth+scale);
        check_share(tmp_even, 1ULL << (bitwidth), "tmp_even after truncate");
        check_share(tmp_odd, 1ULL << (bitwidth), "tmp_odd after truncate");
        // 分别加上系数
        if(party == ALICE){
          for(size_t i = 0; i < x.size(); i++){
            tmp_even(i) = tmp_even(i) + coe_fix[2];
            tmp_odd(i) = tmp_odd(i) + coe_fix[3];
          }
        }
        check_share(tmp_even, 1ULL << (bitwidth), "tmp_even before mul");
        check_share(tmp_odd, 1ULL << (bitwidth), "tmp_odd before mul");
        
        // 1. 偶数部分用秦九韶折叠: E = (c0 * x^2 + c2) * x^2 + c4
        // Step 1.1: 转到field做安全乘法 tmp_even * x_2
        fixPoint->Ring2Field(tmp_even, HE->plain_mod, bitwidth);
        fixPoint->Ring2Field(x_2, HE->plain_mod, bitwidth);
        
        auto even_part = ElementWiseMul(tmp_even, x_2, HE);
        check_share(even_part, HE->plain_mod, "even_part_field");
        
        // Step 1.2: 转回ring并加上c4
        fixPoint->Field2Ring(even_part, HE->plain_mod, bitwidth+scale);
        fixPoint->truncate_reduce(even_part, scale, bitwidth+scale);
        check_share(even_part, 1ULL << (bitwidth), "even_part after truncate"); 
        
        if (party == ALICE){
          for(size_t i = 0; i < x.size(); i++){
            even_part(i) = even_part(i) + coe_fix[4];
          }
        }
        check_share(even_part, 1ULL << (bitwidth), "even_part after adding c4"); // correct
        // 2. 奇数部分用秦九韶折叠: O = (c1 * x^2 + c3) * x
        // Step 2.1: 转到field做安全乘法 tmp_odd * x
        fixPoint->Ring2Field(tmp_odd, HE->plain_mod, bitwidth);
        
        auto odd_part = ElementWiseMul(tmp_odd, x, HE);
        check_share(odd_part, HE->plain_mod, "odd_part_field");
        
        // Step 2.2: 转回ring
        fixPoint->Field2Ring(odd_part, HE->plain_mod, bitwidth+scale);
        fixPoint->truncate_reduce(odd_part, scale, bitwidth+scale);
        check_share(odd_part, 1ULL << (bitwidth), "odd_part after truncate");
        
        // 3. 线性偏移: L(x) = 0.5 * x (需要先恢复x_ring)
        Tensor<T> linear_offset(x.shape());
        for(size_t i = 0; i < x.size(); i++){
          linear_offset(i) = x_ring(i) * round(0.5 * (1ULL << scale));
        }
        fixPoint->truncate_reduce(linear_offset, scale, bitwidth+scale);
        check_share(linear_offset, 1ULL << (bitwidth), "linear_offset after truncate");
        // 4. 组合结果: F0 = E + L - O, F1 = E + L + O
        Tensor<T> F0(x.shape());
        Tensor<T> F1(x.shape());
        check_share(even_part, 1ULL << (bitwidth), "even_part before final");
        check_share(odd_part, 1ULL << (bitwidth), "odd_part before final");
        check_share(linear_offset, 1ULL << (bitwidth), "linear_offset before final");
        for(size_t i = 0; i < x.size(); i++){
          F0(i) = even_part(i) + linear_offset(i) - odd_part(i);
          F1(i) = even_part(i) + linear_offset(i) + odd_part(i);
        }
        check_share(F0, 1ULL << (bitwidth), "F0");
        check_share(F1, 1ULL << (bitwidth), "F1");
  
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
      
      void operator_old(Tensor<T> &x){
        cout << "bitwidth, scale, plain_mod:" << bitwidth << " " << scale << " " << HE->plain_mod << endl;
        Tensor<T> x_ring = x;
        // check_share(x, 1ULL << (bitwidth), "x_ring before");
        // cout << "bitwidth, scale, plain_mod:" << bitwidth << " " << scale << " " << HE->plain_mod << endl;
        fixPoint->Ring2Field(x, HE->plain_mod, bitwidth);
        // check_share(x, HE->plain_mod, "x_field");
        auto x_2 = ElementWiseMul(x, x, HE);

        // check_share(x_2, HE->plain_mod, "x_2_field");

        fixPoint->Field2Ring(x_2, HE->plain_mod, bitwidth+scale);

        // check_share(x_2, 1ULL << (bitwidth+scale), "x_2_ring");


        fixPoint->truncate_reduce(x_2,scale,bitwidth+scale);
        // check_share(x_2, 1ULL << (bitwidth), "x_2 after truncate");
        // cout << "OK4" << endl;
        Tensor<T> x_2_ring = x_2;
        fixPoint->Ring2Field(x_2, HE->plain_mod, bitwidth);
        // check_share(x_2, HE->plain_mod, "x_2_field after truncate");

        auto x_3 = ElementWiseMul(x_2, x, HE);
        // check_share(x_3, HE->plain_mod, "x_3");
        fixPoint->Field2Ring(x_3, HE->plain_mod, bitwidth+scale);
        // check_share(x_3, 1ULL << (bitwidth+scale), "x_3_ring");
        fixPoint->truncate_reduce(x_3,scale,bitwidth+scale);
        // check_share(x_3, 1ULL << (bitwidth), "x_3 after truncate");
        auto x_4 = ElementWiseMul(x_2, x_2, HE);
        // check_share(x_4, HE->plain_mod, "x_4");
        fixPoint->Field2Ring(x_4, HE->plain_mod, bitwidth+scale);
        // check_share(x_4, 1ULL << (bitwidth+scale), "x_4_ring");
        fixPoint->truncate_reduce(x_4,scale,bitwidth+scale);
        // check_share(x_4, 1ULL << (bitwidth), "x_4 after truncate");
        Tensor<T> F0(x.shape());
        Tensor<T> F1(x.shape());
        for(size_t i = 0; i < x.size(); i++){
          F0(i) = x_4(i)*coe_fix[0]-x_3(i)*coe_fix[1]+x_2_ring(i)*coe_fix[2]+x_ring(i)*(round(0.5* (1ULL << scale))-coe_fix[3])+coe_fix[4];
          F1(i) = x_4(i)*coe_fix[0]+x_3(i)*coe_fix[1]+x_2_ring(i)*coe_fix[2]+x_ring(i)*(round(0.5* (1ULL << scale))+coe_fix[3])+coe_fix[4];
        }
        // check_share(F0, 1ULL << (bitwidth), "F0");
        // check_share(F1, 1ULL << (bitwidth), "F1");
        
        fixPoint->truncate(F0,scale,bitwidth);
        fixPoint->truncate(F1,scale,bitwidth);
        // check_share(F0, 1ULL << (bitwidth), "F0 after truncate");
        // check_share(F1, 1ULL << (bitwidth), "F1 after truncate");
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
