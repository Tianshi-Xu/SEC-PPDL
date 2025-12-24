#include <NonlinearLayer/GeLU.h>
#include <HE/HE.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <vector>
#include <limits>
using namespace std;
using namespace NonlinearLayer;
using namespace HE;
#define MAX_THREADS 4
typedef int64_t T;
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

// 将float转换为bf16 (brain floating point 16)
// BF16格式: 1位符号 + 8位指数 + 7位尾数
uint16_t float_to_bf16(float value) {
    uint32_t f32;
    memcpy(&f32, &value, sizeof(float));
    
    // 处理NaN和Inf
    uint32_t exp = (f32 >> 23) & 0xFF;
    if (exp == 0xFF) {
        // NaN或Inf,直接截断
        return (uint16_t)(f32 >> 16);
    }
    
    // 舍入到最近偶数 (Round to Nearest Even)
    // 检查被截断部分的最高位(第16位)
    uint32_t rounding_bias = 0x7FFF + ((f32 >> 16) & 1);
    f32 += rounding_bias;
    
    // 取高16位
    return (uint16_t)(f32 >> 16);
}

// 将bf16转换回float
float bf16_to_float(uint16_t bf16_value) {
    uint32_t f32 = ((uint32_t)bf16_value) << 16;
    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

// 打印bf16值的二进制表示(用于调试)
void print_bf16_bits(uint16_t bf16_val, float original) {
    cout << "原始float: " << original << " -> BF16: ";
    for(int i = 15; i >= 0; i--) {
        cout << ((bf16_val >> i) & 1);
        if(i == 15 || i == 7) cout << " ";  // 分隔符号位和指数位
    }
    cout << " -> 还原float: " << bf16_to_float(bf16_val) << endl;
}

// bf16版本的gelu ground truth
// 使用标准GeLU公式: GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x/√2))
// 或者使用tanh近似: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
Tensor<double> gelu_gt(Tensor<double> &input){
  
  Tensor<double> output(input.shape());
  
  // 常数 1/√2 用于erf计算
  const double SQRT2_INV = 0.7071067811865476;  // 1/√2
  
  for(int i = 0; i < input.size(); i++){
    // 输入转换为bf16精度
    float x_f32 = (float)input(i);
    uint16_t x_bf16 = float_to_bf16(x_f32);
    float x = bf16_to_float(x_bf16);
    
    // 标准GeLU公式: GELU(x) = x * Φ(x)
    // 其中 Φ(x) = 0.5 * (1 + erf(x/√2))
    
    // 计算 x/√2
    float x_scaled_f = x * SQRT2_INV;
    x_scaled_f = bf16_to_float(float_to_bf16(x_scaled_f));
    
    // 计算 erf(x/√2)
    float erf_val_f = std::erf(x_scaled_f);
    erf_val_f = bf16_to_float(float_to_bf16(erf_val_f));
    
    // 计算 0.5 * (1 + erf(x/√2))
    float phi_f = 0.5f * (1.0f + erf_val_f);
    phi_f = bf16_to_float(float_to_bf16(phi_f));
    
    // 计算 x * Φ(x)
    float result_f = x * phi_f;
    result_f = bf16_to_float(float_to_bf16(result_f));
    
    output(i) = (double)result_f;
  }
  
  return output;
}

void test_gelu(){
  const size_t n = 8192;
  Tensor<T> input({n});
  Tensor<double> input_real({n});
  Tensor<double> gt({n});
  const int bitwidth = 20;
  const int scale = 15;
  he = new HE::HEEvaluator(ioArr[0], party, 8192,bitwidth*2,Datatype::HOST,{});
  he->GenerateNewKey();
  he->print_parameters();
  if (party == ALICE){
    for (size_t i = 0; i < n; ++i){
      double v = -4 + 8.0 * static_cast<double>(i) / static_cast<double>(n - 1); // [-4,4]均匀分布
      // double v = -2;
      input_real(i) = v;
      input(i) = static_cast<T>(llround(v * static_cast<double>(1ULL << scale)));
    }
    printf("input_real(0): ", input_real(0));
    printf("input(0): ", input(0));
    // input(0) = 4.00049 * (1ULL << scale);
    // input_real(0) = 4.00049;
    gt = gelu_gt(input_real);
  }
  
  GeLU<T> gelu(fixpoint, he, bitwidth, scale);
  gelu(input);  // in-place share of HE GeLU output

  if (party == ALICE){
    Tensor<T> other_output({n});
    ioArr[0]->recv_tensor(other_output);

    Tensor<double> recon({n});
    const uint64_t ring_mask = ((bitwidth >= 64) ? std::numeric_limits<uint64_t>::max() : ((1ULL << bitwidth) - 1));
    const uint64_t sign_bit = (bitwidth >= 64) ? (1ULL << 63) : (1ULL << (bitwidth - 1));
    const uint64_t modulus = (bitwidth >= 64) ? 0ULL : (1ULL << bitwidth);
    const double inv_scale = 1.0 / static_cast<double>(1ULL << scale);

    for (size_t i = 0; i < n; ++i) {
      uint64_t combined_u = (static_cast<uint64_t>(input(i)) + static_cast<uint64_t>(other_output(i))) & ring_mask;
      int64_t combined_s = (combined_u >= sign_bit && bitwidth < 64)
                               ? static_cast<int64_t>(combined_u) - static_cast<int64_t>(modulus)
                               : static_cast<int64_t>(combined_u);
      recon(i) = static_cast<double>(combined_s) * inv_scale;
    }

    double mae = 0.0, max_err = 0.0;
    struct ErrSample { double err; size_t idx; double input; double gt; double recon; };
    std::vector<ErrSample> err_rank;
    err_rank.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      double err = std::abs(recon(i) - gt(i));
      mae += err;
      max_err = std::max(max_err, err);
      err_rank.push_back({err, i, input_real(i), gt(i), recon(i)});
    }
    mae /= static_cast<double>(n);

    std::sort(err_rank.begin(), err_rank.end(), [](const ErrSample &a, const ErrSample &b){
      return a.err > b.err;
    });

    std::cout << "[GeLU] top mismatches (idx, in, gt, he, err):" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(8, err_rank.size()); ++i) {
      const auto &s = err_rank[i];
      std::cout << "  [" << s.idx << "] " << s.input << ", " << s.gt
                << ", " << s.recon << ", err=" << s.err << std::endl;
    }
    std::cout << "[GeLU] sample recon[0]=" << recon(0) << std::endl;
    std::cout << "[GeLU] mae=" << mae << ", max_err=" << max_err << std::endl;
    std::ofstream ofs("gelu_eval.csv");
    ofs << "input,gt,he\n";
    for (size_t i = 0; i < n; ++i){
      ofs << input_real(i) << "," << gt(i) << "," << recon(i) << "\n";
    }
    ofs.close();
    std::cout << "[GeLU] saved inputs/gt/he outputs to gelu_eval.csv" << std::endl;
  } else {
    ioArr[0]->send_tensor(input);
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
  fixpoint = new FixPoint<T>(party, otpackArr, num_threads);
  // std::cout << "After one-time setup, communication" << std::endl; // TODO: warm up
  // for (int i = 0; i < num_threads; i++) {
  //   auto temp = ioArr[i]->counter;
  //   comm_threads[i] = temp;
  //   std::cout << "Thread i = " << i << ", total data sent till now = " << temp
  //             << std::endl;
  // }

  // 测试不同密文位宽
  // test_gelu_bitwidth();
  
  // 原始测试
  test_gelu();

  // uint64_t totalComm = 0;
  // for (int i = 0; i < num_threads; i++) {
  //   auto temp = ioArr[i]->counter;
  //   std::cout << "Thread i = " << i << ", total data sent till now = " << temp
  //             << std::endl;
  //   totalComm += (temp - comm_threads[i]);
  // }
}
