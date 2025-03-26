#include <NonlinearLayer/GeLU.h>
#include <HE/HE.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace NonlinearLayer;
using namespace HE;
#define MAX_THREADS 4
typedef uint64_t T;
int party, port = 8000;
int num_threads = 4;
string address = "127.0.0.1";

int bitlength = 16;
int32_t kScale = 12;
Utils::NetIO *ioArr[MAX_THREADS];
OTPrimitive::OTPack<Utils::NetIO> *otpackArr[MAX_THREADS];

FixPoint<T> *fixpoint;
HEEvaluator *he;
uint64_t comm_threads[MAX_THREADS];

Tensor<double> gelu_gt(Tensor<double> &input){
  double coe[5] = {0.020848611754127593, -0.18352506127082727, 0.5410550166368381, -0.03798164612714154, 0.001620808531841547};
  Tensor<double> output({8192});
  for(int i=0;i<input.size();i++){
    double x = input(i);
    double abs_x = abs(x);

    if(x<0){
      output(i) = 0;
    }
    else if(abs_x<=2.7){
      output(i) = coe[0] * pow(abs_x, 4) + coe[1] * pow(abs_x, 3) + coe[2] * pow(abs_x, 2) + coe[3] * abs_x + coe[4]+0.5*x;
    }
    else{
      output(i) = x;
    }
  }
  return output;
}

void test_gelu(){
  /************ Generate Test Data ************/
  /********************************************/
  Tensor<T> input({8192});
  Tensor<double> input_real({8192});
  int bitwidth = 22, scale = 20;
  Tensor<double> gt({8192});
  if(party == ALICE){
    input.randomize(1ULL<< (bitwidth-1));
    input(0) = 2.8 * (1ULL << scale);
    input(1) = 1.93021 * (1ULL << scale);
    input.print(8);
    for(size_t i = 0; i < input.size(); i++){
      input_real(i) = double(input(i)) / (1ULL << scale);
    }
    input_real.print(8);
    gt = gelu_gt(input_real);
  }
  
  GeLU<T> gelu(fixpoint, he, 2*bitwidth, scale);
  gelu(input);
  // input.print();
  if (party == ALICE){
    ioArr[0]->send_tensor(input);
  }
  else{
    Tensor<T> input0({8192});
    ioArr[0]->recv_tensor(input0);
    input = input0 + input;
    for(size_t i = 0; i < input.size(); i++){
      input(i) = input(i) & ((1ULL << bitwidth) - 1);
    }
    Tensor<double> output_real({8});
    for(size_t i = 0; i < input.size(); i++){
      output_real(i) = double(input(i)) / (1ULL << scale);
    }
    output_real.print(8);
  }
  cout << "gt:" << endl;
  gt.print(8);
  
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
    if (i & 1) {
      otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], 3 - party); 
    } else {
      otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], party);
    }
  }
  he = new HE::HEEvaluator(ioArr[0], party, 8192,60,Datatype::HOST);
  he->GenerateNewKey();
  fixpoint = new FixPoint<T>(party, otpackArr, num_threads);
  std::cout << "After one-time setup, communication" << std::endl; // TODO: warm up
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    comm_threads[i] = temp;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
  }

  test_gelu();

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
