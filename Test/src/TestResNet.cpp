#include <Model/ResNet.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace NonlinearLayer;
using namespace Model;
using namespace Datatype;
#define MAX_THREADS 32

int bitlength = 32;
int party, port = 32000;
int num_threads = 32;
string address = "127.0.0.1";

uint64_t comm_threads[MAX_THREADS];
void test_tensor(Tensor<uint64_t> &x) {
  cout << "test_tensor called" << endl;
  x.print_shape();
  cout << "test_tensor done" << endl;
}
int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2"); // 1 is server, 2 is client
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  assert(num_threads <= MAX_THREADS);

  // you can switch IKNP/VOLE; Cheetah/Nested; HOST/DEVICE
  CryptoPrimitive<uint64_t, Utils::NetIO> *cryptoPrimitive = new CryptoPrimitive<uint64_t, Utils::NetIO>(party, num_threads, bitlength, Datatype::VOLE, 8192, 60, Nest, Datatype::DEVICE, address, port);

  // ResNet_3stages<uint64_t> model = resnet_32_c10(cryptoPrimitive);
  ResNet_4stages<uint64_t> model = resnet_50(cryptoPrimitive);
  Tensor<uint64_t> input({3, 224, 224});
  input.randomize(16);
  auto start = high_resolution_clock::now();
  Tensor<uint64_t> output = model(input);
  cout << "resnet done" << endl;
  output.print_shape();

  cout << "time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << " s" << endl;
  uint64_t totalComm = cryptoPrimitive->get_total_comm();
  cout << "totalComm: " << totalComm << " bytes" << endl;
  uint64_t totalRounds = cryptoPrimitive->get_total_rounds();
  cout << "totalRounds: " << totalRounds << endl;

  // output.print();
}
