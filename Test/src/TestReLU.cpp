/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <NonlinearLayer/ReLU.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace NonlinearLayer;
#define MAX_THREADS 4

int party, port = 32000;
int num_threads = 4;
string address = "127.0.0.1";

int bitlength = 16;
int32_t kScale = 12;
NetIO *ioArr[MAX_THREADS];
OTPack<NetIO> *otpackArr[MAX_THREADS];
ReLUProtocol<NetIO, int32_t> **reluprotocol = new ReLUProtocol<NetIO, int32_t>*[MAX_THREADS];

uint64_t comm_threads[MAX_THREADS];

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
        new NetIO(party == ALICE ? nullptr : address.c_str(), port + i);
    if (i & 1) {
      otpackArr[i] = new OTPack<NetIO>(ioArr[i], 3 - party);
    } else {
      otpackArr[i] = new OTPack<NetIO>(ioArr[i], party);
    }
    reluprotocol[i] = new ReLURingProtocol<NetIO, int32_t>(party, ioArr[i], 4, MILL_PARAM, otpackArr[i]);
  }
  std::cout << "After one-time setup, communication" << std::endl; // TODO: warm up
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    comm_threads[i] = temp;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
  }

  /************ Generate Test Data ************/
  /********************************************/
  Tensor<int32_t> input({8});
  input.randomize(4);
  input.print();

  ReLU<int32_t> relu(reluprotocol, 4, num_threads);
  relu(&input);
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
