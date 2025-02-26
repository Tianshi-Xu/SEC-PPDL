#include <NonlinearLayer/ReLU.h>
#include <NonlinearOperator/Truncation.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace NonlinearLayer;
#define MAX_THREADS 4

int party, port = 32000;
int num_threads = 1;
string address = "127.0.0.1";

int bitlength = 16;
int32_t kScale = 12;
Utils::NetIO *ioArr[MAX_THREADS];
OTPrimitive::OTPack<Utils::NetIO> *otpackArr[MAX_THREADS];
ReLUProtocol<Utils::NetIO, int32_t> **reluprotocol = new ReLUProtocol<Utils::NetIO, int32_t>*[MAX_THREADS];
TruncationProtocol **truncationProtocol = new TruncationProtocol*[MAX_THREADS];

uint64_t comm_threads[MAX_THREADS];

void test_relu(){
  uint32_t dim = 64;
  Tensor<int32_t> input({dim});
  Tensor<int32_t> input2({dim});
  if(party == Utils::BOB) {
    input.randomize(4);
  }
  input2 = input;
  input.print();

  ReLU<int32_t> relu(reluprotocol, 4, num_threads);
  relu(&input);
  // input.print();

  uint64_t totalComm = 0;
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
    totalComm += (temp - comm_threads[i]);
  }

  /************** Verification ****************/
  /********************************************/
  if (party == ALICE) {
    ioArr[0]->send_data(input.data().data(), dim * sizeof(int32_t));
  } else { // party == BOB
    int32_t *input0 = new int32_t[dim];
    ioArr[0]->recv_data(input0, dim * sizeof(int32_t));
    int error = 0;
    for (uint32_t i = 0; i < dim; i++) {
      input({i}) = (input({i}) + input0[i]) & ((1ULL << 4) - 1);
      if(input({i})!=(input2({i})>0?input2({i}):0)) {
        error++;
      }
      // cout << "input[" << i << "] = " << input({i}) << endl;
    }
    input.print();
    cout << "error = " << error << endl;
    delete[] input0;
  }
}

void test_truncation(){
  uint32_t dim = 64;
  Tensor<uint64_t> input({dim});
  Tensor<uint64_t> input2({dim});
  if(party == Utils::BOB) {
    input.randomize(4);
  }
  input2 = input;
  input.print();

  NonlinearOperator::Truncation<uint64_t> truncation(truncationProtocol, num_threads);
  truncation(&input, 1, 4, false);
  uint64_t totalComm = 0;
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
    totalComm += (temp - comm_threads[i]);
  }
  if (party == ALICE) {
    ioArr[0]->send_data(input.data().data(), dim * sizeof(uint64_t));
  } else { // party == BOB
    uint64_t *input0 = new uint64_t[dim];
    ioArr[0]->recv_data(input0, dim * sizeof(uint64_t));
    int error = 0;
    for (uint32_t i = 0; i < dim; i++) {
      input({i}) = (input({i}) + input0[i]) & ((1ULL << 4) - 1);
      if(input({i})!=(input2({i})>>1)) {
        cout << "input[" << i << "] = " << input({i}) << " != " << (input2({i})>>1) << endl;
        error++;
      }
      // cout << "input[" << i << "] = " << input({i}) << endl;
    }
    input.print();
    cout << "error = " << error << endl;
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
        new Utils::NetIO(party == Utils::ALICE ? nullptr : address.c_str(), port + i);
    if (i & 1) {
      otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], 3 - party);
    } else {
      otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], party);
    }
    reluprotocol[i] = new ReLURingProtocol<Utils::NetIO, int32_t>(party, ioArr[i], 4, MILL_PARAM, otpackArr[i], Datatype::IKNP);
    truncationProtocol[i] = new TruncationProtocol(party, ioArr[i], otpackArr[i]);
  }
  std::cout << "After one-time setup, communication" << std::endl; // TODO: warm up
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    comm_threads[i] = temp;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
  }
  // test_relu();
  test_truncation();
  /************ Generate Test Data ************/
  /********************************************/
  

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
