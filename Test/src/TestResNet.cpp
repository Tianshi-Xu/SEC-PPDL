#include <Model/ResNet.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace NonlinearLayer;
using namespace Model;
#define MAX_THREADS 8

int party, port = 32000;
int num_threads = 8;
string address = "127.0.0.1";

int bitlength = 16;
int32_t kScale = 12;

uint64_t comm_threads[MAX_THREADS];
void test_tensor(Tensor<uint64_t> &x) {
  cout << "test_tensor called" << endl;
  x.print_shape();
  cout << "test_tensor done" << endl;
}
int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2"); // 1 is server, 2 is client
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  assert(num_threads <= MAX_THREADS);

  CryptoPrimitive<uint64_t, Utils::NetIO> *cryptoPrimitive = new CryptoPrimitive<uint64_t, Utils::NetIO>(party, num_threads, bitlength, VOLE, 8192, 60, Nest, DEVICE, address, port);
  ResNet_3stages<uint64_t> model = resnet_32_c10(cryptoPrimitive);
  Tensor<uint64_t> input({3, 32, 32});
  if (party == ALICE) {
    input.randomize(16);
  }
  Tensor<uint64_t> output = model(input);
  cout << "resnet done" << endl;
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
