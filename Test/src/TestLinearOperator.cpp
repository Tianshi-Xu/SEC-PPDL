#include <LinearOperator/Polynomial.h>
#include <Utils/ArgMapping/ArgMapping.h>
#include <iostream>

int party, port = 32000;
int num_threads = 2;
std::string address = "127.0.0.1";

Utils::NetIO* netio;
HE::HEEvaluator* he;
using namespace std;
using namespace LinearOperator;
using namespace HE;

namespace {
void test_poly(HE::HEEvaluator* he){;
    Tensor<uint64_t> x({8192});
    Tensor<uint64_t> y({8192});
    Tensor<uint64_t> z({8192});
    if(party == ALICE){
        for(uint32_t i = 0; i < 8192; i++){
            x(i) = i;
            y(i) = i;
        }
    }
    x.print(10);
    y.print(10);
    z = LinearOperator::ElementWiseMul(x, x, he);
    if (party == ALICE){
        netio->send_tensor(z);
    }else{
        Tensor<uint64_t> z0({8192});
        netio->recv_tensor(z0);
        auto z_result = z + z0;
        for(uint32_t i = 0; i < 8192; i++){
            z_result(i) = z_result(i) % he->plain_mod;
        }
        z_result.print(10);
    }
}

std::string Int128ToString(int128_t value) {
    if (value == 0) {
        return "0";
    }
    bool negative = value < 0;
    int128_t abs_value = negative ? -value : value;
    std::string digits;
    while (abs_value > 0) {
        auto remainder = static_cast<int32_t>(abs_value % 10);
        digits.push_back(static_cast<char>('0' + remainder));
        abs_value /= 10;
    }
    if (negative) {
        digits.push_back('-');
    }
    std::reverse(digits.begin(), digits.end());
    return digits;
}

void test_ckks_inverse_fft() {
    const std::size_t degree = 8;
    const int fft_scale = 40;              // keep transform scaling moderate to avoid overflow
    const int extra_shift = 32;            // boosts payload magnitude while staying within int128 range
    const int128_t scale = static_cast<int128_t>(int64_t{1}) << fft_scale;
    const int128_t payload_shift = static_cast<int128_t>(1) << extra_shift;
    const int128_t base = scale * payload_shift; // stored fixed-point unit for real value 1 << extra_shift

    std::vector<LinearOperator::Complex128> inputs(degree);
    std::cout << "CKKSInverseFFT inputs (fixed-point, scale=2^" << fft_scale
              << ", payload shift=2^" << extra_shift << "): ";
    for (std::size_t i = 0; i < degree; ++i) {
        int128_t magnitude = static_cast<int128_t>(static_cast<long long>(i + 1)) * base;
        inputs[i] = LinearOperator::Complex128(magnitude, static_cast<int128_t>(0));
        std::cout << Int128ToString(inputs[i].real());
        if (i + 1 != degree) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    auto outputs = LinearOperator::CKKSInverseFFT(inputs, degree, fft_scale);

    std::cout << "CKKSInverseFFT outputs (real parts): ";
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        std::cout << Int128ToString(outputs[i].real());
        if (i + 1 != outputs.size()) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    const int128_t numerator = static_cast<int128_t>(degree + 1);
    const int128_t denominator = static_cast<int128_t>(2);
    const int128_t average_scaled = (numerator * base) / denominator;
    std::cout << "Expected first coefficient (average * base): "
              << Int128ToString(average_scaled) << std::endl;
}
} // namespace

int main(int argc, char **argv){
    // ArgMapping amap;
    // amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2"); // 1 is server, 2 is client
    // amap.arg("p", port, "Port Number");
    // amap.arg("ip", address, "IP Address of server (ALICE)");
    // amap.parse(argc, argv);
    
    // netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
    // std::cout << "netio generated" << std::endl;
    // he = new HE::HEEvaluator(netio, party, 8192,32,Datatype::HOST);
    // he->GenerateNewKey();
    
    // test_poly(he);
    test_ckks_inverse_fft();

    return 0;
}