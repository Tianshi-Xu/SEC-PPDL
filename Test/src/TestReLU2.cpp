#include <NonlinearLayer/ReLU.h>
#include <NonlinearOperator/Truncation.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace NonlinearLayer;
#define MAX_THREADS 1

int party, port = 32000;
int num_threads = 1;
string address = "127.0.0.1";

int bitlength = 16;
int32_t kScale = 12;
Utils::NetIO *io;
Utils::NetIO *ioArr[MAX_THREADS];
OTPrimitive::OTPack<Utils::NetIO> *otpackArr[MAX_THREADS];
ReLUProtocol<int32_t, Utils::NetIO> *reluprotocol[MAX_THREADS];
TruncationProtocol *truncationProtocol[MAX_THREADS];

uint64_t comm_threads[MAX_THREADS];

void test_relu(){
  
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
    ioArr[i] = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port + i);
    // otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i],party);
    otpackArr[i] = new VOLEOTPack<Utils::NetIO>(ioArr[i], party);
    reluprotocol[i] = new ReLURingProtocol<int32_t, Utils::NetIO>(party, ioArr[i], 4, MILL_PARAM, otpackArr[i], Datatype::VOLE);
    truncationProtocol[i] = new TruncationProtocol(party, ioArr[i], otpackArr[i]);
  }
  io = ioArr[0];
  std::cout << "After one-time setup, communication" << std::endl; // TODO: warm up
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    comm_threads[i] = temp;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
  }
  // test_relu();
  uint32_t dim = 16;
  Tensor<int32_t> input({dim});
  Tensor<int32_t> input2({dim});
  if(party == BOB) {
    input.randomize(4);
  }
  cout << "in main party = " << party << endl;
  input2 = input;
  input.print();

  ReLU<int32_t> relu(reluprotocol, 4, num_threads);
  relu(input);
  input.print();
  uint64_t totalComm = 0;
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
    totalComm += (temp - comm_threads[i]);
  }
  

  /************** Verification ****************/
  /********************************************/
  cout << "Begin Verification" << endl;
  if (party == ALICE) {
    io->send_data(input.data().data(), dim * sizeof(int32_t));
  } else { // party == BOB
    int32_t *input0 = new int32_t[dim];
    io->recv_data(input0, dim * sizeof(int32_t));
    int error = 0;
    for (uint32_t i = 0; i < dim; i++) {
      input({i}) = (input({i}) + input0[i]) & ((1ULL << 4) - 1);
      if(input({i})!=(input2({i})>0?input2({i}):0)) {
        error++;
      }
      // cout << "input[" << i << "] = " << input({i}) << endl;
    }
    // input.print();
    cout << "error = " << error << endl;
    delete[] input0;
  }
}
