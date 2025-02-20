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

#include "Operator/truncation.h"
#include "Utils/utils.h"
#include <fstream>
#include <iostream>
#include <thread>

using namespace Utils;
using namespace OT;
using namespace std;

#define MAX_THREADS 8

int party, port = 32000;
int num_threads = 1;
string address = "127.0.0.1";
bool six_comparison = true;

int dim = 10000;//1ULL << 16;
int bw_x = 32;
int s_x = 28;


int bitlength = 16;
int shift = 8;
int32_t kScale = 12;

NetIO *io;
OTPack<NetIO> *otpack;
NetIO *ioArr[MAX_THREADS];
OTPack<NetIO> *otpackArr[MAX_THREADS];

uint64_t comm_threads[MAX_THREADS];

void trunc_thread(int tid, uint64_t *rev, uint64_t *input, int num_ops, uint64_t bitlength) {
    Truncation *trunc_protocol;
  if (tid & 1) {
    trunc_protocol = new Truncation(3 - party, ioArr[tid], otpackArr[tid]);
  } else {
    trunc_protocol = new Truncation(3 - party, ioArr[tid], otpackArr[tid]);
  }
  trunc_protocol->truncate(num_ops, input, rev, shift, bitlength);

  delete trunc_protocol;
}

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", dim, "Number of ReLU operations");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("six", six_comparison, "ReLU6?");
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
  }
  io = ioArr[0];
  otpack = otpackArr[0];
  std::cout << "All Base OTs Done" << std::endl;
  std::cout << "After one-time setup, communication" << std::endl;
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    comm_threads[i] = temp;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
  }

  /************ Generate Test Data ************/
  /********************************************/
  PRG128 prg;

  uint64_t *input = new uint64_t[dim];
  uint64_t *res = new uint64_t[dim];

  prg.random_data(input, dim * sizeof(uint64_t));
  for (int i=0; i<dim; i++){
    input[i]=input[i] & ((1ULL << bitlength) - 1);
  }

  /************** Fork Threads ****************/
  /********************************************/
  // cheetah里silent OT会做warm up影响统计
//   uint64_t total_comm = 0;
//   uint64_t thread_comm[num_threads];
//   for (int i = 0; i < num_threads; i++) {
//     thread_comm[i] = iopackArr[i]->get_comm();
//   }

  auto start = clock_start();
  std::thread trunc_threads[num_threads];
  int chunk_size = dim / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    int offset = i * chunk_size;
    int lnum_ops;
    if (i == (num_threads - 1)) {
      lnum_ops = dim - offset;
    } else {
      lnum_ops = chunk_size;
    }
    trunc_threads[i] =
        std::thread(trunc_thread, i, res, input, lnum_ops, bitlength);
  }
  for (int i = 0; i < num_threads; ++i) {
    trunc_threads[i].join();
  }
  std::cout << "compare success\n";
  long long t = time_from(start);

//   for (int i = 0; i < num_threads; i++) {
//     thread_comm[i] = ioArr[i]->get_comm() - thread_comm[i];
//     total_comm += thread_comm[i];
//   }

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
    ioArr[0]->send_data(input, dim * sizeof(uint64_t));
    ioArr[0]->send_data(res, dim * sizeof(uint64_t));
  } else { // party == BOB
    uint64_t *input0 = new uint64_t[dim];
    uint64_t *res0 = new uint64_t[dim];
    ioArr[0]->recv_data(input0, dim * sizeof(uint64_t));
    ioArr[0]->recv_data(res0, dim * sizeof(uint64_t));

    for (int i = 0; i < 10; i++) {
      uint64_t res_result = (res[i] + res0[i]) & ((1ULL << bitlength) - 1);
      cout << endl;
      cout <<  "origin_sum:" << std::hex << ((input[i] + input0[i]) & ((1ULL << bitlength) - 1)) << endl;
      cout << "res_sum:" << std::hex << res_result << "  " << "res_share0:" << res[i] << "  " << "res_share1:" << res0[i] << endl;
    //   int64_t X = signed_val(x[i] + x0[i], bw_x);
    //   int64_t Y = signed_val(y[i] + y0[i], bw_x);
    //   int64_t expectedY = X;
    //   if (X < 0)
    //     expectedY = 0;
    //   if (six != 0) {
    //     if (X > int64_t(six))
    //       expectedY = six;
    //   }
    //   // cout << X << "\t" << Y << "\t" << expectedY << endl;
    //   assert(Y == expectedY);
    }

    // cout << "ReLU" << (six == 0 ? "" : "6") << " Tests Passed" << endl;

    delete[] input0;
    delete[] res0;
  }

  /**** Process & Write Benchmarking Data *****/
  /********************************************/
  cout << "Number of truncation/s:\t" << (double(dim) / t) * 1e6 << std::endl;
  cout << "one truncation cost:\t" << (t / double(dim)) << std::endl;
  cout << "truncation Time\t" << t / (1000.0) << " ms" << endl;
  cout << "truncation Bytes Sent\t" << (totalComm) << " byte" << endl;

  /******************* Cleanup ****************/
  /********************************************/
  delete[] res;
  delete[] input;
  for (int i = 0; i < num_threads; i++) {
    delete ioArr[i];
    delete otpackArr[i];
  }
}
