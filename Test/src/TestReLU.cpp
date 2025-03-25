#include <NonlinearLayer/ReLU.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace NonlinearLayer;
#define MAX_THREADS 4

int party, port = 8000;
int num_threads = 4;
string address = "127.0.0.1";

int bitlength = 16;
int32_t kScale = 12;
Utils::NetIO *ioArr[MAX_THREADS];
OTPrimitive::OTPack<Utils::NetIO> *otpackArr[MAX_THREADS];
ReLUProtocol<int32_t, Utils::NetIO> **reluprotocol = new ReLUProtocol<int32_t, Utils::NetIO>*[MAX_THREADS];

uint64_t comm_threads[MAX_THREADS];

void test_relu(){
  /************ Generate Test Data ************/
  /********************************************/
  Tensor<int32_t> input({8});
  input.randomize(4);
  input.print();

  ReLU<int32_t> relu(reluprotocol, 4, num_threads);
  relu(input);
  input.print();
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
      // 子类对象不能直接赋值给父类对象，因为父类对象不会有子类特有的数据成员，并且可能会丢失子类的数据
      // 但是父类指针可以指向一个子类对象
      // Child 类是从 Parent 类派生的，并且它继承了 Parent 的所有公有成员函数。当创建一个 Child 类对象时，这个对象会包含一个 Parent 类的子对象
      // 当父类指针指向子类对象时，通过虚函数机制（如果父类函数是虚函数）可以实现多态，使得调用的成员函数是子类中的重载版本，而不是父类的版本
    } else {
      otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], party);
    }
    reluprotocol[i] = new ReLURingProtocol<int32_t, Utils::NetIO>(party, 4, MILL_PARAM, otpackArr[i], Datatype::IKNP);
  }
  std::cout << "After one-time setup, communication" << std::endl; // TODO: warm up
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    comm_threads[i] = temp;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
  }

  test_relu();

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
