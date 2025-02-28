#include <Model/ResNet.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace NonlinearLayer;
using namespace Model;
#define MAX_THREADS 4

int port = 32000;
int num_threads = 1;
string address = "127.0.0.1";

int bitlength = 16;
int32_t kScale = 12;
Utils::NetIO *ioArr[MAX_THREADS];
OTPrimitive::OTPack<Utils::NetIO> *otpackArr[MAX_THREADS];
ReLUProtocol<Utils::NetIO, int32_t> **reluprotocol = new ReLUProtocol<Utils::NetIO, int32_t>*[MAX_THREADS];
TruncationProtocol **truncationProtocol = new TruncationProtocol*[MAX_THREADS];

uint64_t comm_threads[MAX_THREADS];
void test_tensor(Tensor<uint64_t> &x) {
  cout << "test_tensor called" << endl;
  x.print_shape();
  cout << "test_tensor done" << endl;
}
int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  bool party = std::stoi(argv[1]);

  assert(num_threads <= MAX_THREADS);

  /********** Setup IO and Base OTs ***********/
  /********************************************/
  // for (int i = 0; i < num_threads; i++) {
  //   ioArr[i] =
  //       new Utils::NetIO(party == Utils::ALICE ? nullptr : address.c_str(), port + i);
  //   if (i & 1) {
  //     otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], 3 - party);
  //   } else {
  //     otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], party);
  //   }
  //   reluprotocol[i] = new ReLURingProtocol<Utils::NetIO, int32_t>(party, ioArr[i], bitlength, MILL_PARAM, otpackArr[i], Datatype::IKNP);
  //   truncationProtocol[i] = new TruncationProtocol(party, ioArr[i], otpackArr[i]);
  // }
  // std::cout << "After one-time setup, communication" << std::endl; // TODO: warm up
  // for (int i = 0; i < num_threads; i++) {
  //   auto temp = ioArr[i]->counter;
  //   comm_threads[i] = temp;
  //   std::cout << "Thread i = " << i << ", total data sent till now = " << temp
  //             << std::endl;
  // }
  Utils::NetIO netio(party==0?nullptr:address.c_str(), port);
  std::cout << "netio generated" << std::endl;
  HE::HEEvaluator HE(netio, party);
  HE.GenerateNewKey();
  CryptoPrimitive<uint64_t> *cryptoPrimitive = new CryptoPrimitive<uint64_t>(party==0?Utils::ALICE:Utils::BOB, &HE, num_threads, bitlength, Datatype::IKNP, Datatype::CONV_TYPE::Nest, address, port);
  Bottleneck<uint64_t> bottleneck(16, 32, 64, 1, cryptoPrimitive);
  Tensor<uint64_t> input({32, 16, 16});
  if (party == Utils::ALICE) {
    input.randomize(16);
    // input.print();
  }
  // test_tensor(input);
  Tensor<uint64_t> output = bottleneck(input);
  cout << "bottleneck done" << endl;
  output.print_shape();
  // output.print();
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
