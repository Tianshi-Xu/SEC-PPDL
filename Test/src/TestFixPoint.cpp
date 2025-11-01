#include <NonlinearOperator/FixPoint.h>
#include <iostream>
using namespace std;
using namespace NonlinearOperator;
#define MAX_THREADS 4
typedef uint64_t T;

int party, port = 8000;
int num_threads = 1;
string address = "127.0.0.1";

int bitlength = 16;
int32_t kScale = 12;
Utils::NetIO *ioArr[MAX_THREADS];
OTPrimitive::OTPack<Utils::NetIO> *otpackArr[MAX_THREADS];
NonlinearOperator::FixPoint<T> *fixpoint;

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
  Tensor<T> result({2,4});
  if (party == ALICE) {
    input.randomize(1ULL<<4);
    input(0) = 716;
  }
  input.print();
  // fixpoint->Field2Ring(input, 5, 4);
  fixpoint->Ring2Field(input, 1099511480321, 10);
  input.print();
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
  std::cout << "After one-time setup, communication" << std::endl; // TODO: warm up
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    comm_threads[i] = temp;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
  }
  
  // test_comapre();
  // test_ring_field();
  // test_secure_round();
  test_secure_requant();

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
