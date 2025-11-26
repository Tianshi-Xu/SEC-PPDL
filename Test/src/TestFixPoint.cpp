#include <NonlinearOperator/FixPoint.h>
#include <seal/util/common.h>
#include <iostream>
#include <limits>
#include <string>
using namespace std;
using namespace NonlinearOperator;
#define MAX_THREADS 4
typedef uint64_t T;
typedef int128_t T128;

int party, port = 8000;
int num_threads = 1;
string address = "127.0.0.1";

int bitlength = 16;
int32_t kScale = 12;
Utils::NetIO *ioArr[MAX_THREADS];
OTPrimitive::OTPack<Utils::NetIO> *otpackArr[MAX_THREADS];
NonlinearOperator::FixPoint<T> *fixpoint;
NonlinearOperator::FixPoint<T128> *fixpoint128;

uint64_t comm_threads[MAX_THREADS];

void test_comapre(){
  /************ Generate Test Data ************/
  /********************************************/
  Tensor<T> input({2,4});  
  Tensor<T> input2({2,4});
  Tensor<uint8_t> result({2,4});
  if (party == ALICE) {
    input.randomize(1ULL<<4);
    input2.randomize(1ULL<<4);
    // input2(0) = 7;
    // input(0)=-1;
  }
  else{
    // input(0)=-1;
  }
  input.print();
  input2.print();
  fixpoint->less_than_zero(input, result, 4);
  result.print();
  fixpoint->less_than_constant(input, 1, result, 4);
  result.print();
  fixpoint->less_than(input, input2, result, 4);
  result.print();
}

void test_ring_field(){
  Tensor<T> input({2,4});
  Tensor<T128> input128({2,4});
  Tensor<T128> result128({2,4});
  constexpr int ring_bw = 10;
  constexpr int ring_bw_128 = 60;
  constexpr int ring2field_ext_bits = 40;

  auto reconstruct_u64 = [&](Tensor<T> &tensor, int stage_bw, const std::string &label) {
    if (party == ALICE) {
      ioArr[0]->send_data(tensor.data().data(), tensor.size() * sizeof(T));
    } else {
      Tensor<T> peer(tensor.shape());
      ioArr[0]->recv_data(peer.data().data(), tensor.size() * sizeof(T));
      Tensor<T> combined(tensor.shape());
      uint64_t mask = stage_bw >= 64 ? std::numeric_limits<uint64_t>::max()
                                     : ((uint64_t(1) << stage_bw) - 1);
      for (int i = 0; i < tensor.size(); i++) {
        combined(i) = (tensor(i) + peer(i)) & mask;
      }
      std::cout << label << std::endl;
      combined.print();
    }
  };

  auto reconstruct_i128 = [&](Tensor<T128> &tensor, int stage_bw, const std::string &label) {
    if (party == ALICE) {
      ioArr[0]->send_data(tensor.data().data(), tensor.size() * sizeof(T128));
    } else {
      Tensor<T128> peer(tensor.shape());
      ioArr[0]->recv_data(peer.data().data(), tensor.size() * sizeof(T128));
      Tensor<T128> combined(tensor.shape());
      int128_t mask = (stage_bw == 128) ? int128_t(-1)
                                         : ((int128_t(1) << stage_bw) - 1);
      for (int i = 0; i < tensor.size(); i++) {
        combined(i) = (tensor(i) + peer(i)) & mask;
      }
      std::cout << label << std::endl;
      combined.print();
    }
  };
  if (party == ALICE) {
    input.randomize(1ULL<<4);
    input(0) = 716;
    input128.randomize(1ULL<<4);
    input128(0) = static_cast<T128>(1) << 45;
  }
  input.print();
  input128.print();
  const uint64_t Q64 = 1099511480321ULL;
  const int128_t Q128 = (int128_t(1) << 80);  // allow values up to 2^80 in tests
  fixpoint->Ring2Field(input, Q64, ring_bw);
  fixpoint128->Ring2Field(input128, Q128, ring_bw_128);
  input.print();
  input128.print();
  
  // Ring2Field reconstructions should match ALICE's plaintext reduced mod Q with the extra 40 guard bits.
  // e.g. the first uint64_t slot should read 716, int128_t slot ~2^45=35184372088832 after reconstruction.
  reconstruct_u64(input, ring_bw + ring2field_ext_bits, "[Ring2Field] uint64_t reconstruction");
  reconstruct_i128(input128, ring_bw_128 + ring2field_ext_bits, "[Ring2Field] int128_t reconstruction");

  fixpoint->Field2Ring(input, Q64, ring_bw);
  fixpoint128->Field2Ring(input128, Q128, ring_bw_128);
  input.print();
  input128.print();
  // Field2Ring reconstructions should return to the original ALICE shares (mod 2^{bitwidth}).
  // So the first entries should again be 716 (uint64) and 2^45 (int128) respectively.
  reconstruct_u64(input, ring_bw, "[Field2Ring] uint64_t reconstruction");
  reconstruct_i128(input128, ring_bw_128, "[Field2Ring] int128_t reconstruction");
}

void test_secure_round(){
  /************ Generate Test Data ************/
  /********************************************/
  Tensor<T> input({2,4});
  int32_t s_fix = 4;    // 定点小数位数
  int32_t bw_fix = 16;  // 定点数位宽
  int32_t bw_acc = 12;  // 结果位宽
  
  if (party == ALICE) {
    // 测试数据: 一些接近舍入边界的值
    // 例如: 15 = 0...01111, 16 = 0...10000, 17 = 0...10001
    // 在s_fix=4时，threshold = 2^3 = 8
    input(0) = 7;   // < 8, 舍入下降
    input(1) = 8;   // = 8, 舍入上升
    input(2) = 15;  // 接近边界
    input(3) = 23;  // > 16
    input(4) = 16;  // = 16
    input(5) = 24;  // 恰好是16+8
    input(6) = 32;  // 2*16
    input(7) = 40;  // 2*16+8
  } else {
    // BOB的shares为0
    for (int i = 0; i < input.size(); i++) {
      input(i) = 0;
    }
  }
  
  cout << "Before secure_round:" << endl;
  input.print();
  
  fixpoint->secure_round(input, s_fix, bw_fix, bw_acc);
  
  cout << "After secure_round:" << endl;
  input.print();
  
  /************** Result Verification ****************/
  /***************************************************/
  if (party == ALICE) {
    ioArr[0]->send_data(input.data().data(), input.size() * sizeof(T));
  } else { // party == BOB
    T *input0 = new T[input.size()];
    ioArr[0]->recv_data(input0, input.size() * sizeof(T));
    
    // 创建 Tensor 来存储重建的结果
    Tensor<T> reconstructed(input.shape());
    for (int i = 0; i < input.size(); i++) {
      reconstructed(i) = (input(i) + input0[i]) & ((1ULL << bw_acc) - 1);
    }
    
    cout << "\n=== Reconstruction of secure_round results ===" << endl;
    cout << "Reconstructed values:" << endl;
    reconstructed.print();
    
    delete[] input0;
  }
}

void test_secure_requant(){
  /************ Generate Test Data ************/
  /********************************************/
  Tensor<T> input({2,4});
  
  // Algorithm 2: From b_acc to b_fix with scale s to scale 2^{s_fix}
  // Using realistic scales < 1.0 in the range (0, 1)
  double scale_in = 0.25;   // scale s = 0.25
  double scale_out = 0.1;   // s' = 0.1 for demonstration
  int32_t bw_in = 12;       // b_acc = 12
  int32_t bw_out = 16;      // b_fix = 16
  int32_t s_fix = 4;        // s_fix = 4
  
  if (party == ALICE) {
    // 测试数据: 整数输入值
    // Algorithm computes: X_f = X_q * s * 2^{s_fix}
    //                    = X_q * 0.25 * 16 = X_q * 4
    input(0) = 10;   // Expected: 10 * 4 = 40
    input(1) = 20;   // Expected: 20 * 4 = 80
    input(2) = 100;  // Expected: 100 * 4 = 400
    input(3) = 500;  // Expected: 500 * 4 = 2000
    input(4) = 1000; // Expected: 1000 * 4 = 4000
    input(5) = 2000; // Expected: 2000 * 4 = 8000
    input(6) = 3000; // Expected: 3000 * 4 = 12000
    input(7) = 4095; // Expected: 4095 * 4 = 16380
  } else {
    // BOB的shares为0
    for (int i = 0; i < input.size(); i++) {
      input(i) = 0;
    }
  }
  
  cout << "\n=== Testing secure_requant: b_acc to b_fix ===" << endl;
  cout << "Input (b_acc=12 bits, scale=" << scale_in << "):" << endl;
  input.print();
  
  fixpoint->secure_requant(input, scale_in, scale_out, bw_in, bw_out, s_fix);
  
  cout << "Output (b_fix=16 bits, scale=2^" << s_fix << "):" << endl;
  // Expected reconstruction (ALICE plaintext × 4): [40, 80, 400, 2000, 4000, 8000, 12000, 16380]
  input.print();
  
  /************** Result Verification ****************/
  /***************************************************/
  if (party == ALICE) {
    ioArr[0]->send_data(input.data().data(), input.size() * sizeof(T));
  } else { // party == BOB
    T *input0 = new T[input.size()];
    ioArr[0]->recv_data(input0, input.size() * sizeof(T));
    
    // 创建 Tensor 来存储重建的结果
    Tensor<T> reconstructed(input.shape());
    uint64_t mask_out = (bw_out == 64 ? -1ULL : ((1ULL << bw_out) - 1));
    for (int i = 0; i < input.size(); i++) {
      reconstructed(i) = (input(i) + input0[i]) & mask_out;
    }
    
    cout << "\n=== Reconstruction of secure_requant results ===" << endl;
    cout << "Reconstructed values:" << endl;
    reconstructed.print();
    
    delete[] input0;
  }
}

void test_extend_128bit() {
  std::cout << "\n=== Testing 128-bit Extend ===" << std::endl;
  
  Tensor<T128> input({4});
  Tensor<T128> result({4});
  
  int32_t bwA = 40;  // Original bitwidth
  int32_t bwB = 72;  // Extended bitwidth
  
  if (party == ALICE) {
    // Initialize with values that require 128-bit representation
    int128_t base = int128_t(1) << 35;  // Large base value
    input(0) = base * 1;
    input(1) = base * 2;
    input(2) = base * 3;
    input(3) = base * 4;
  } else {
    // BOB's shares are zero
    for (int i = 0; i < input.size(); i++) {
      input(i) = 0;
    }
  }
  
  std::cout << "Before extend (bwA=" << bwA << " -> bwB=" << bwB << "):" << std::endl;
  // Note: Tensor<T128>::print() doesn't support int128_t directly, so we print manually
  std::cout << "Input values: ";
  for (int i = 0; i < input.size(); i++) {
    std::cout << static_cast<long long>(input(i)) << " ";
  }
  std::cout << std::endl;
  
  // Test zero extension
  result = input;
  fixpoint128->extend(result, bwA, bwB, false);
  
  std::cout << "After z_extend:" << std::endl;
  std::cout << "Result values: ";
  for (int i = 0; i < result.size(); i++) {
    std::cout << static_cast<long long>(result(i)) << " ";
  }
  std::cout << std::endl;
  
  // Test signed extension
  result = input;
  fixpoint128->extend(result, bwA, bwB, true);
  
  std::cout << "After s_extend:" << std::endl;
  std::cout << "Result values: ";
  for (int i = 0; i < result.size(); i++) {
    std::cout << static_cast<long long>(result(i)) << " ";
  }
  std::cout << std::endl;
  
  // Verify results by reconstructing
  if (party == ALICE) {
    ioArr[0]->send_data(input.data().data(), input.size() * sizeof(T128));
    ioArr[0]->send_data(result.data().data(), result.size() * sizeof(T128));
  } else {
    T128 *input0 = new T128[input.size()];
    T128 *result0 = new T128[result.size()];
    ioArr[0]->recv_data(input0, input.size() * sizeof(T128));
    ioArr[0]->recv_data(result0, result.size() * sizeof(T128));
    
    int128_t mask_bwA = (bwA == 128 ? -1 : ((int128_t(1) << bwA) - 1));
    int128_t mask_bwB = (bwB == 128 ? -1 : ((int128_t(1) << bwB) - 1));
    
    std::cout << "Verification (reconstructed values):" << std::endl;
    for (int i = 0; i < input.size(); i++) {
      int128_t orig_sum = (input(i) + input0[i]) & mask_bwA;
      int128_t result_sum = (result(i) + result0[i]) & mask_bwB;
      // Convert to string for output (int128_t doesn't have direct ostream support)
      auto to_string = [](int128_t val) -> std::string {
        if (val == 0) return "0";
        bool neg = val < 0;
        int128_t abs_val = neg ? -val : val;
        std::string s;
        while (abs_val > 0) {
          s = std::to_string(static_cast<int>(abs_val % 10)) + s;
          abs_val /= 10;
        }
        return neg ? "-" + s : s;
      };
      std::cout << "  [" << i << "] orig=" << to_string(orig_sum)
                << ", extended=" << to_string(result_sum) << std::endl;
    }
    
    delete[] input0;
    delete[] result0;
  }
}

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");

  amap.parse(argc, argv);

  assert(num_threads <= MAX_THREADS);

  /********** Setup IO and Base OTs ***********/
  /********************************************/
  for (int i = 0; i < num_threads; i++) {
    ioArr[i] =
        new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port + i);
    otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], party);
  }
  fixpoint = new NonlinearOperator::FixPoint<T>(party, otpackArr, num_threads);
  fixpoint128 = new NonlinearOperator::FixPoint<T128>(party, otpackArr, num_threads);
  std::cout << "After one-time setup, communication" << std::endl; // TODO: warm up
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    comm_threads[i] = temp;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
  }
  
  // test_comapre();
  test_ring_field();
  // test_secure_round();
  // test_secure_requant();
  // test_extend_128bit();

  uint64_t totalComm = 0;
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
    totalComm += (temp - comm_threads[i]);
  }

  /************** Verification ****************/
  /********************************************/
  // if (party == ALICE) {
  //   ioArr[0]->send_data(input, dim * sizeof(uint64_t));
  //   ioArr[0]->send_data(res, dim * sizeof(uint64_t));
  // } else { // party == BOB
  //   uint64_t *input0 = new uint64_t[dim];
  //   uint64_t *res0 = new uint64_t[dim];
  //   ioArr[0]->recv_data(input0, dim * sizeof(uint64_t));
  //   ioArr[0]->recv_data(res0, dim * sizeof(uint64_t));

  //   for (int i = 0; i < 10; i++) {
  //     uint64_t res_result = (res[i] + res0[i]) & ((1ULL << bitlength) - 1);
  //     cout << endl;
  //     cout <<  "origin_sum:" << ((input[i] + input0[i]) & ((1ULL << bitlength) - 1)) << endl;
  //     cout << "res_sum:" << res_result << "  " << "res_share0:" << res[i] << "  " << "res_share1:" << res0[i] << endl;
  //   //   int64_t X = signed_val(x[i] + x0[i], bw_x);
  //   //   int64_t Y = signed_val(y[i] + y0[i], bw_x);
  //   //   int64_t expectedY = X;
  //   //   if (X < 0)
  //   //     expectedY = 0;
  //   //   if (six != 0) {
  //   //     if (X > int64_t(six))
  //   //       expectedY = six;
  //   //   }
  //   //   // cout << X << "\t" << Y << "\t" << expectedY << endl;
  //   //   assert(Y == expectedY);
  //   }

    // cout << "ReLU" << (six == 0 ? "" : "6") << " Tests Passed" << endl;

    // delete[] input0;
    // delete[] res0;
  // }

  /**** Process & Write Benchmarking Data *****/
  /********************************************/
  // cout << "Number of ring-relu/s:\t" << (double(dim) / t) * 1e6 << std::endl;
  // cout << "one ring-relu cost:\t" << (t / double(dim)) << std::endl;
  // cout << "ring-relu Time\t" << t / (1000.0) << " ms" << endl;
  // cout << "ring-relu Bytes Sent\t" << (totalComm) << " byte" << endl;

  // /******************* Cleanup ****************/
  // /********************************************/
  // delete[] res;
  // delete[] input;
  // for (int i = 0; i < num_threads; i++) {
  //   delete ioArr[i];
  //   delete otpackArr[i];
  // }
}
