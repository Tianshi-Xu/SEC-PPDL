#include <NonlinearOperator/FixPoint.h>
#include <seal/util/common.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
using namespace std;
using namespace NonlinearOperator;
#define MAX_THREADS 4
typedef int64_t T;
typedef int128_t T128;

int party, port = 8000;
int num_threads = 4;
string address = "127.0.0.1";

int bitlength = 16;
int32_t kScale = 12;
Utils::NetIO *ioArr[MAX_THREADS];
OTPrimitive::OTPack<Utils::NetIO> *otpackArr[MAX_THREADS];
NonlinearOperator::FixPoint<T> *fixpoint;
NonlinearOperator::FixPoint<T128> *fixpoint128;

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
  constexpr size_t slot_count = 8192;
  constexpr int ring_bw = 16;
  constexpr int scale = 10;  // convert floats with 10 fractional bits
  const uint64_t Q = 65537ULL;  // small prime > 2^16 to keep reductions simple
  const uint64_t ring_mask = (ring_bw >= 64)
                                 ? std::numeric_limits<uint64_t>::max()
                                 : ((uint64_t(1) << ring_bw) - 1);
  const double min_val = -5.0;
  const double max_val = 5.0;

  // Create float tensor with test values
  Tensor<double> float_input({slot_count});
  for (size_t i = 0; i < slot_count; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(slot_count - 1);
    float_input(i) = min_val + (max_val - min_val) * t;
  }

  // Convert to fixed-point using Tensor's built-in method
  Tensor<T> input = Tensor<T>::FromFloatTensorToFixed(float_input, scale);
  
  // Apply ring mask to ensure values are in range
  if (party == ALICE) {
    for (size_t i = 0; i < slot_count; ++i) {
      input(i) = input(i) & ring_mask;
    }
  } else {
    for (size_t i = 0; i < slot_count; ++i) {
      input(i) = 0;
    }
  }

  fixpoint->Ring2Field(input, Q, ring_bw);

  if (party == ALICE) {
    ioArr[0]->send_data(input.data().data(), input.size() * sizeof(T));
  } else {
    Tensor<T> peer(input.shape());
    ioArr[0]->recv_data(peer.data().data(), input.size() * sizeof(T));

    size_t mismatches = 0;
    const size_t sample = 8;
    std::vector<size_t> mismatch_indices;
    mismatch_indices.reserve(sample);

    // Reconstruct original float values for comparison
    Tensor<double> expected_floats({slot_count});
    for (size_t i = 0; i < slot_count; ++i) {
      double t = static_cast<double>(i) / static_cast<double>(slot_count - 1);
      expected_floats(i) = min_val + (max_val - min_val) * t;
    }
    Tensor<T> expected_ring = Tensor<T>::FromFloatTensorToFixed(expected_floats, scale);
    
    for (size_t i = 0; i < slot_count; ++i) {
      // Reconstruct in field domain: (alice_share + bob_share) mod Q
      uint64_t field_recon = (input(i) + peer(i)) % Q;
      // Expected: original ring value mod Q
      uint64_t ring_original = expected_ring(i) & ring_mask;
      uint64_t expected_field = ring_original % Q;
      
      if (field_recon != expected_field) {
        if (mismatch_indices.size() < sample) {
          mismatch_indices.push_back(i);
        }
        ++mismatches;
      }
      if (i < sample) {
        std::cout << "slot " << i << ": field_recon=" << field_recon
                  << ", expected=" << expected_field
                  << " (from " << expected_floats(i) << ")" << std::endl;
      }
    }

    if (mismatches == 0) {
      std::cout << "[test_ring_field] PASS: all " << slot_count
                << " slots match after Ring2Field conversion" << std::endl;
    } else {
      std::cout << "[test_ring_field] FAIL: " << mismatches
                << " mismatches detected" << std::endl;
      for (size_t idx : mismatch_indices) {
        uint64_t field_recon = (input(idx) + peer(idx)) % Q;
        uint64_t ring_original = expected_ring(idx) & ring_mask;
        uint64_t expected_field = ring_original % Q;
        std::cout << "  slot " << idx << ": field_recon=" << field_recon
                  << ", expected=" << expected_field << std::endl;
      }
    }
  }
}

void test_field_ring(){
  constexpr size_t slot_count = 8192;
  constexpr int ring_bw = 16;
  constexpr int scale = 10;
  const uint64_t Q = 65537ULL;
  const uint64_t ring_mask = (ring_bw >= 64)
                                 ? std::numeric_limits<uint64_t>::max()
                                 : ((uint64_t(1) << ring_bw) - 1);
  const double min_val = -5.0;
  const double max_val = 5.0;

  // Create float tensor with test values
  Tensor<double> float_values({slot_count});
  for (size_t i = 0; i < slot_count; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(slot_count - 1);
    float_values(i) = min_val + (max_val - min_val) * t;
  }

  // Convert to fixed-point using Tensor's built-in method
  Tensor<T> original_ring = Tensor<T>::FromFloatTensorToFixed(float_values, scale);
  
  // Store original ring values for verification
  std::vector<uint64_t> original_ring_values(slot_count);
  
  Tensor<T> field_input({slot_count});
  if (party == ALICE) {
    for (size_t i = 0; i < slot_count; ++i) {
      // Encode as ring value, then take mod Q to simulate field input
      uint64_t ring_val = original_ring(i) & ring_mask;
      original_ring_values[i] = ring_val;
      field_input(i) = ring_val % Q;
    }
  } else {
    for (size_t i = 0; i < slot_count; ++i) {
      field_input(i) = 0;
    }
  }

  fixpoint->Field2Ring(field_input, Q, ring_bw);

  if (party == ALICE) {
    ioArr[0]->send_data(field_input.data().data(), field_input.size() * sizeof(T));
    // Send original values for verification
    ioArr[0]->send_data(original_ring_values.data(), original_ring_values.size() * sizeof(uint64_t));
  } else {
    Tensor<T> peer(field_input.shape());
    ioArr[0]->recv_data(peer.data().data(), field_input.size() * sizeof(T));
    
    std::vector<uint64_t> original_from_alice(slot_count);
    ioArr[0]->recv_data(original_from_alice.data(), original_from_alice.size() * sizeof(uint64_t));

    size_t mismatches = 0;
    const size_t sample = 8;
    std::vector<size_t> mismatch_indices;
    mismatch_indices.reserve(sample);

    for (size_t i = 0; i < slot_count; ++i) {
      // Reconstruct in ring domain
      uint64_t ring_recon = (field_input(i) + peer(i)) & ring_mask;
      // Expected: original ring value (mod ring_mask)
      uint64_t expected_ring = original_from_alice[i] & ring_mask;
      
      if (ring_recon != expected_ring) {
        if (mismatch_indices.size() < sample) {
          mismatch_indices.push_back(i);
        }
        ++mismatches;
      }
      if (i < sample) {
        std::cout << "slot " << i << ": ring_recon=" << ring_recon
                  << ", expected=" << expected_ring
                  << " (from " << float_values(i) << ")" << std::endl;
      }
    }

    if (mismatches == 0) {
      std::cout << "[test_field_ring] PASS: all " << slot_count
                << " slots match after Field2Ring conversion" << std::endl;
    } else {
      std::cout << "[test_field_ring] FAIL: " << mismatches
                << " mismatches detected" << std::endl;
      for (size_t idx : mismatch_indices) {
        uint64_t ring_recon = (field_input(idx) + peer(idx)) & ring_mask;
        uint64_t expected_ring = original_from_alice[idx] & ring_mask;
        std::cout << "  slot " << idx << ": ring_recon=" << ring_recon
                  << ", expected=" << expected_ring << std::endl;
      }
    }
  }
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
  // Expected reconstruction (ALICE plaintext × 4): [40, 80, 400, 2000, 4000, 8000, 12000, 16380]
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

void test_extend_u64() {
  constexpr size_t slot_count = 8192;
  constexpr int32_t bwA = 38;
  constexpr int32_t bwB = 60;
  Tensor<T> input({slot_count});

  if (party == ALICE) {
    for (size_t i = 0; i < slot_count; ++i) {
      input(i) = i;
    }
  } else {
    for (size_t i = 0; i < slot_count; ++i) {
      input(i) = 0;
    }
  }

  fixpoint->extend(input, bwA, bwB, true);

  if (party == ALICE) {
    ioArr[0]->send_data(input.data().data(), input.size() * sizeof(T));
  } else {
    Tensor<T> peer(input.shape());
    ioArr[0]->recv_data(peer.data().data(), input.size() * sizeof(T));

    const uint64_t maskA = (bwA == 64 ? std::numeric_limits<uint64_t>::max()
                                      : ((uint64_t(1) << bwA) - 1));
    const uint64_t maskB = (bwB == 64 ? std::numeric_limits<uint64_t>::max()
                                      : ((uint64_t(1) << bwB) - 1));

    size_t mismatches = 0;
    for (size_t i = 0; i < slot_count; ++i) {
      uint64_t combined = (input(i) + peer(i)) & maskB;
      uint64_t expected = i & maskA;
      if (combined != expected) {
        ++mismatches;
      }
      if (i>=2040 && i<2060) {
        std::cout << "slot " << i << ": got " << combined
                  << ", expected " << expected << std::endl;
      }
      if (i>=4090 && i<4110) {
        std::cout << "slot " << i << ": got " << combined
                  << ", expected " << expected << std::endl;
      }
    }

    if (mismatches == 0) {
      std::cout << "[test_extend_u64] PASS: zero-extend preserved all "
                << slot_count << " slots" << std::endl;
    } else {
      std::cout << "[test_extend_u64] FAIL: " << mismatches
                << " mismatches detected" << std::endl;
    }
  }
}

// Validate less_than_constant by reconstructing and comparing with cleartext predicate.
void test_less_than_constant() {
  constexpr int32_t bw = 16;
  constexpr size_t n = 32;
  const T constant = 3;

  Tensor<T> input({n});
  Tensor<uint8_t> result({n});

  if (party == ALICE) {
    input.randomize(1ULL << bw);
  } else {
    for (size_t i = 0; i < n; ++i) input(i) = 0;
  }

  fixpoint->less_than_constant(input, constant, result, bw);

  if (party == ALICE) {
    ioArr[0]->send_data(input.data().data(), input.size() * sizeof(T));
    ioArr[0]->send_data(result.data().data(), result.size() * sizeof(uint8_t));
  } else {
    Tensor<T> other_input({n});
    Tensor<uint8_t> other_result({n});
    ioArr[0]->recv_data(other_input.data().data(), other_input.size() * sizeof(T));
    ioArr[0]->recv_data(other_result.data().data(), other_result.size() * sizeof(uint8_t));

    const uint64_t ring_mask = (bw == 64 ? std::numeric_limits<uint64_t>::max()
                                       : ((1ULL << bw) - 1));
    const uint64_t sign_bit = (bw == 64 ? (1ULL << 63) : (1ULL << (bw - 1)));
    const uint64_t modulus = (bw == 64 ? 0ULL : (1ULL << bw));

    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
      uint64_t combined_u = (static_cast<uint64_t>(input(i)) + static_cast<uint64_t>(other_input(i))) & ring_mask;
      int64_t combined_s;
      if (bw == 64) {
        combined_s = static_cast<int64_t>(combined_u);
      } else if (combined_u >= sign_bit) {
        combined_s = static_cast<int64_t>(combined_u) - static_cast<int64_t>(modulus);
      } else {
        combined_s = static_cast<int64_t>(combined_u);
      }

      uint8_t pred = result(i) ^ other_result(i);  // MSB output is XOR-shared
      uint8_t expected = (combined_s < constant) ? 1 : 0;
      if (pred != expected && mismatches < 8) {
        std::cout << "[less_than_constant] mismatch idx=" << i
                  << " x=" << combined_s
                  << " const=" << constant
                  << " pred=" << static_cast<int>(pred)
                  << " expected=" << static_cast<int>(expected) << std::endl;
      }
      mismatches += (pred != expected);
    }

    if (mismatches == 0) {
      std::cout << "[less_than_constant] all " << n << " cases matched." << std::endl;
    } else {
      std::cout << "[less_than_constant] mismatches=" << mismatches << " out of " << n << std::endl;
    }
  }
}

void test_truncate_and_truncate_reduce() {
  constexpr size_t slot_count = 8192;
  constexpr int32_t scale_bits = 12;
  constexpr int32_t bitwidth = 32;  // Input bitwidth
  constexpr int32_t shift = 6;      // Truncate shift amount
  const double min_val = -5.0;
  const double max_val = 5.0;

  std::cout << "\n=== Testing truncate and truncate_reduce ===" << std::endl;
  std::cout << "Input range: [" << min_val << ", " << max_val << "]" << std::endl;
  std::cout << "scale_bits=" << scale_bits << ", bitwidth=" << bitwidth << ", shift=" << shift << std::endl;

  // Create float tensor with test values in range [-5, 5]
  Tensor<double> float_values({slot_count});
  for (size_t i = 0; i < slot_count; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(slot_count - 1);
    float_values(i) = min_val + (max_val - min_val) * t;
  }

  // Convert to fixed-point
  Tensor<T> fixed_values = Tensor<T>::FromFloatTensorToFixed(float_values, scale_bits);
  
  // Create two copies for testing two operations
  Tensor<T> input_truncate_reduce({slot_count});
  Tensor<T> input_truncate({slot_count});
  
  if (party == ALICE) {
    for (size_t i = 0; i < slot_count; ++i) {
      input_truncate_reduce(i) = fixed_values(i);
      input_truncate(i) = fixed_values(i);
    }
  } else {
    for (size_t i = 0; i < slot_count; ++i) {
      input_truncate_reduce(i) = 0;
      input_truncate(i) = 0;
    }
  }

  // Test 1: truncate_reduce - shift right AND reduce bitwidth by shift bits
  std::cout << "\n--- Testing truncate_reduce ---" << std::endl;
  std::cout << "Input bitwidth=" << bitwidth << ", Output bitwidth=" << (bitwidth - shift) << std::endl;
  
  fixpoint->truncate_reduce(input_truncate_reduce, shift, bitwidth);

  // Test 2: truncate - shift right but keep same bitwidth
  std::cout << "\n--- Testing truncate ---" << std::endl;
  std::cout << "Shift=" << shift << ", bitwidth=" << bitwidth << " (unchanged)" << std::endl;
  
  fixpoint->truncate(input_truncate, shift, bitwidth);

  // Verify both results
  if (party == ALICE) {
    ioArr[0]->send_data(input_truncate_reduce.data().data(), input_truncate_reduce.size() * sizeof(T));
    ioArr[0]->send_data(input_truncate.data().data(), input_truncate.size() * sizeof(T));
  } else {
    Tensor<T> peer_truncate_reduce(input_truncate_reduce.shape());
    Tensor<T> peer_truncate(input_truncate.shape());
    ioArr[0]->recv_data(peer_truncate_reduce.data().data(), peer_truncate_reduce.size() * sizeof(T));
    ioArr[0]->recv_data(peer_truncate.data().data(), peer_truncate.size() * sizeof(T));

    const int32_t bitwidth_reduced = bitwidth - shift;  // truncate_reduce reduces bitwidth by shift
    const uint64_t mask_reduced = (bitwidth_reduced == 64 ? std::numeric_limits<uint64_t>::max()
                                                           : ((1ULL << bitwidth_reduced) - 1));
    const uint64_t mask_original = (bitwidth == 64 ? std::numeric_limits<uint64_t>::max()
                                                    : ((1ULL << bitwidth) - 1));

    // Verify truncate_reduce
    std::cout << "\n=== Verification of truncate_reduce ===" << std::endl;
    std::cout << "Output bitwidth: " << bitwidth_reduced << " (reduced by " << shift << " bits)" << std::endl;
    size_t mismatches_tr = 0;
    for (size_t i = 0; i < slot_count; ++i) {
      uint64_t reconstructed = (input_truncate_reduce(i) + peer_truncate_reduce(i)) & mask_reduced;
      
      // Expected: original value >> shift, masked to reduced bitwidth
      double original_val = float_values(i);
      int64_t original_fixed = static_cast<int64_t>(std::llround(original_val * (1LL << scale_bits)));
      int64_t expected_fixed = original_fixed >> shift;
      uint64_t expected = static_cast<uint64_t>(expected_fixed) & mask_reduced;
      
      if (reconstructed != expected) {
        ++mismatches_tr;
      }
      
      if (i < 8) {
        int64_t signed_recon = static_cast<int64_t>(reconstructed & mask_reduced);
        if (bitwidth_reduced < 64 && (reconstructed & (1ULL << (bitwidth_reduced - 1)))) {
          signed_recon -= (1LL << bitwidth_reduced);
        }
        double recon_float = static_cast<double>(signed_recon) / (1LL << (scale_bits - shift));
        
        std::cout << "slot " << i << ": input=" << original_val 
                  << ", reconstructed=" << recon_float
                  << " (fixed: reconstructed=" << reconstructed << ", expected=" << expected << ")"
                  << std::endl;
      }
    }

    if (mismatches_tr == 0) {
      std::cout << "[truncate_reduce] PASS: all " << slot_count << " values correct" << std::endl;
    } else {
      std::cout << "[truncate_reduce] FAIL: " << mismatches_tr << "/" << slot_count << " mismatches" << std::endl;
    }

    // Verify truncate
    std::cout << "\n=== Verification of truncate ===" << std::endl;
    std::cout << "Output bitwidth: " << bitwidth << " (unchanged)" << std::endl;
    size_t mismatches_t = 0;
    for (size_t i = 0; i < slot_count; ++i) {
      uint64_t reconstructed = (input_truncate(i) + peer_truncate(i)) & mask_original;
      
      // Expected: original value >> shift, masked to original bitwidth
      double original_val = float_values(i);
      int64_t original_fixed = static_cast<int64_t>(std::llround(original_val * (1LL << scale_bits)));
      int64_t expected_fixed = original_fixed >> shift;
      uint64_t expected = static_cast<uint64_t>(expected_fixed) & mask_original;
      
      if (reconstructed != expected) {
        ++mismatches_t;
      }
      
      if (i < 8) {
        int64_t signed_recon = static_cast<int64_t>(reconstructed & mask_original);
        if (bitwidth < 64 && (reconstructed & (1ULL << (bitwidth - 1)))) {
          signed_recon -= (1LL << bitwidth);
        }
        double recon_float = static_cast<double>(signed_recon) / (1LL << (scale_bits - shift));
        
        std::cout << "slot " << i << ": input=" << original_val 
                  << ", reconstructed=" << recon_float
                  << " (fixed: reconstructed=" << reconstructed << ", expected=" << expected << ")"
                  << std::endl;
      }
    }

    if (mismatches_t == 0) {
      std::cout << "[truncate] PASS: all " << slot_count << " values correct" << std::endl;
    } else {
      std::cout << "[truncate] FAIL: " << mismatches_t << "/" << slot_count << " mismatches" << std::endl;
    }
  }
}

void test_extend_128bit() {
  std::cout << "\n=== Testing 128-bit Extend ===" << std::endl;
  
  Tensor<T128> input({4});
  Tensor<T128> result({4});
  
  int32_t bwA = 40;  // Original bitwidth
  int32_t bwB = 72;  // Extended bitwidth
  
  if (party == ALICE) {
    // Initialize with values that require 128-bit representation
    int128_t base = int128_t(1) << 35;  // Large base value
    input(0) = base * 1;
    input(1) = base * 2;
    input(2) = base * 3;
    input(3) = base * 4;
  } else {
    // BOB's shares are zero
    for (int i = 0; i < input.size(); i++) {
      input(i) = 0;
    }
  }
  
  std::cout << "Before extend (bwA=" << bwA << " -> bwB=" << bwB << "):" << std::endl;
  // Note: Tensor<T128>::print() doesn't support int128_t directly, so we print manually
  std::cout << "Input values: ";
  for (int i = 0; i < input.size(); i++) {
    std::cout << static_cast<long long>(input(i)) << " ";
  }
  std::cout << std::endl;
  
  // Test zero extension
  result = input;
  fixpoint128->extend(result, bwA, bwB, false);
  
  std::cout << "After z_extend:" << std::endl;
  std::cout << "Result values: ";
  for (int i = 0; i < result.size(); i++) {
    std::cout << static_cast<long long>(result(i)) << " ";
  }
  std::cout << std::endl;
  
  // Test signed extension
  result = input;
  fixpoint128->extend(result, bwA, bwB, true);
  
  std::cout << "After s_extend:" << std::endl;
  std::cout << "Result values: ";
  for (int i = 0; i < result.size(); i++) {
    std::cout << static_cast<long long>(result(i)) << " ";
  }
  std::cout << std::endl;
  
  // Verify results by reconstructing
  if (party == ALICE) {
    ioArr[0]->send_data(input.data().data(), input.size() * sizeof(T128));
    ioArr[0]->send_data(result.data().data(), result.size() * sizeof(T128));
  } else {
    T128 *input0 = new T128[input.size()];
    T128 *result0 = new T128[result.size()];
    ioArr[0]->recv_data(input0, input.size() * sizeof(T128));
    ioArr[0]->recv_data(result0, result.size() * sizeof(T128));
    
    int128_t mask_bwA = (bwA == 128 ? -1 : ((int128_t(1) << bwA) - 1));
    int128_t mask_bwB = (bwB == 128 ? -1 : ((int128_t(1) << bwB) - 1));
    
    std::cout << "Verification (reconstructed values):" << std::endl;
    for (int i = 0; i < input.size(); i++) {
      int128_t orig_sum = (input(i) + input0[i]) & mask_bwA;
      int128_t result_sum = (result(i) + result0[i]) & mask_bwB;
      // Convert to string for output (int128_t doesn't have direct ostream support)
      auto to_string = [](int128_t val) -> std::string {
        if (val == 0) return "0";
        bool neg = val < 0;
        int128_t abs_val = neg ? -val : val;
        std::string s;
        while (abs_val > 0) {
          s = std::to_string(static_cast<int>(abs_val % 10)) + s;
          abs_val /= 10;
        }
        return neg ? "-" + s : s;
      };
      std::cout << "  [" << i << "] orig=" << to_string(orig_sum)
                << ", extended=" << to_string(result_sum) << std::endl;
    }
    
    delete[] input0;
    delete[] result0;
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
  fixpoint128 = new NonlinearOperator::FixPoint<T128>(party, otpackArr, num_threads);
  std::cout << "After one-time setup, communication" << std::endl; // TODO: warm up
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    comm_threads[i] = temp;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
  }
  
  // test_comapre();
  // test_ring_field();
  // test_field_ring();
  // test_extend_u64();
  test_less_than_constant();
  // test_secure_round();
  // test_secure_requant();
  // test_extend_128bit();
  // test_truncate_and_truncate_reduce();

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
