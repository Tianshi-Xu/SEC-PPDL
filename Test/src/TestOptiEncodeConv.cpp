#include <LinearLayer/ConvOptiEncode.h>
#include <Utils/ArgMapping/ArgMapping.h>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
using Datatype::Tensor;
uint64_t Residue(int64_t value, uint64_t modulus) {
    return value < 0 ? modulus - static_cast<uint64_t>(-value) % modulus : static_cast<uint64_t>(value) % modulus;
}
struct Case { size_t ci, co, h, k, stride, padding, channels, tile; bool zero, dense; size_t baby = 0; };
Tensor<uint64_t> Reference(const Tensor<uint64_t> &input, const Tensor<uint64_t> &weight,
    const Tensor<uint64_t> &bias, const Case &tc, uint64_t modulus) {
    const size_t out_size = (tc.h + 2 * tc.padding - tc.k) / tc.stride + 1;
    Tensor<uint64_t> output({tc.co, out_size, out_size});
    for (size_t o = 0; o < tc.co; ++o) for (size_t i = 0; i < out_size; ++i)
        for (size_t j = 0; j < out_size; ++j) {
            uint64_t sum = bias(o);
            for (size_t c = 0; c < tc.ci; ++c) for (size_t a = 0; a < tc.k; ++a)
                for (size_t b = 0; b < tc.k; ++b) {
                    const int64_t row = static_cast<int64_t>(i * tc.stride + a) - tc.padding;
                    const int64_t col = static_cast<int64_t>(j * tc.stride + b) - tc.padding;
                    if (row < 0 || col < 0 || row >= static_cast<int64_t>(tc.h) || col >= static_cast<int64_t>(tc.h)) continue;
                    sum = static_cast<uint64_t>((static_cast<__uint128_t>(input({c, size_t(row), size_t(col)})) *
                        weight({o, c, a, b}) + sum) % modulus);
                }
            output({o, i, j}) = sum;
        }
    return output;
}
void RunCase(const Case &tc, size_t index, HE::HEEvaluator &he, int threads, int repeat, bool benchmark) {
    const uint64_t modulus = he.plain_mod;
    std::mt19937_64 rng(1777 + index);
    Tensor<uint64_t> input({tc.ci, tc.h, tc.h}), plain_input(input.shape());
    Tensor<uint64_t> weights({tc.co, tc.ci, tc.k, tc.k}), plain_weights(weights.shape());
    Tensor<uint64_t> bias({tc.co}), plain_bias({tc.co});
    for (size_t i = 0; i < input.size(); ++i) {
        const uint64_t value = tc.dense ? rng() % modulus : Residue(int64_t(rng() % 19) - 9, modulus);
        const uint64_t share = rng() % modulus;
        plain_input(i) = value;
        input(i) = he.server ? share : (value + modulus - share) % modulus;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        plain_weights(i) = tc.zero ? 0 : (tc.dense ? rng() % modulus : Residue(int64_t(rng() % 9) - 4, modulus));
        weights(i) = he.server ? plain_weights(i) : 0;
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        plain_bias(i) = Residue(int64_t(rng() % 15) - 7, modulus);
        bias(i) = he.server ? plain_bias(i) : 0;
    }
    LinearLayer::OptiEncodeOptions options;
    options.threads = threads;
    options.channels_per_block = tc.channels;
    options.tile_output_size = tc.tile;
    options.baby_steps = tc.baby;
    LinearLayer::Conv2DOptiEncode conv(tc.h, tc.stride, tc.padding, weights, bias, &he, options);
    const auto reference = Reference(plain_input, plain_weights, plain_bias, tc, modulus);
    for (int call = 0; call < repeat; ++call) {
        he.IO->sync();
        const auto before = he.IO->counter;
        const auto start = std::chrono::steady_clock::now();
        auto output = conv(input);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto bytes = he.IO->counter - before;
        Tensor<uint64_t> peer(output.shape());
        if (he.server) he.IO->recv_data(peer.data().data(), peer.size() * sizeof(uint64_t));
        else he.IO->send_data(output.data().data(), output.size() * sizeof(uint64_t));
        uint64_t mismatches = 0;
        if (he.server) for (size_t i = 0; i < output.size(); ++i)
            mismatches += (output(i) + peer(i)) % modulus != reference(i);
        he.IO->sync();
        if (he.server) he.IO->send_data(&mismatches, sizeof(mismatches));
        else he.IO->recv_data(&mismatches, sizeof(mismatches));
        if (mismatches) throw std::runtime_error("OptiEncode numerical mismatch: " + std::to_string(mismatches));
        if (he.server) {
            const auto &layout = conv.Layout();
            const auto &stats = conv.Stats();
            std::cout << "PASS case=" << index << " call=" << call << " ci=" << tc.ci << " co=" << tc.co
                << " h=" << tc.h << " k=" << tc.k << " stride=" << tc.stride << " padding=" << tc.padding
                << " threads=" << threads << " C=" << layout.channels << " tiles=" << layout.TileCount()
                << " in_ct=" << stats.input_ciphertexts << " out_ct=" << stats.output_ciphertexts
                << " seconds=" << seconds << " compute=" << stats.compute_seconds
                << " offline=" << stats.offline_seconds << " server_bytes=" << bytes << std::endl;
        }
    }
    if (benchmark) {
        LinearLayer::Conv2DCheetah baseline(tc.h, tc.stride, tc.padding, weights, bias, &he,
            LinearLayer::Conv2DCheetah::PartitionCost::TotalCiphertexts, threads);
        he.IO->sync();
        const auto bytes_before = he.IO->counter;
        const auto start = std::chrono::steady_clock::now();
        auto output = baseline(input);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto baseline_bytes = he.IO->counter - bytes_before;
        if (he.server) for (size_t i = 0; i < output.size(); ++i)
            output(i) = (output(i) + bias(i / (output.shape()[1] * output.shape()[2]))) % modulus;
        Tensor<uint64_t> peer(output.shape());
        if (he.server) he.IO->recv_data(peer.data().data(), peer.size() * sizeof(uint64_t));
        else he.IO->send_data(output.data().data(), output.size() * sizeof(uint64_t));
        uint64_t failures = 0;
        if (he.server) for (size_t i = 0; i < output.size(); ++i)
            failures += (output(i) + peer(i)) % modulus != reference(i);
        if (he.server) he.IO->send_data(&failures, sizeof(failures));
        else he.IO->recv_data(&failures, sizeof(failures));
        if (failures) throw std::runtime_error("Cheetah comparison mismatch");
        if (he.server) std::cout << "BASELINE case=" << index << " seconds=" << seconds
            << " in_ct=" << baseline.dC * baseline.dH * baseline.dW
            << " out_ct=" << baseline.dM * baseline.dH * baseline.dW
            << " server_bytes=" << baseline_bytes << std::endl;
    }
}
}
int main(int argc, char **argv) {
    try {
        int party = 1, port = 32000, threads = 1, degree = 8192, bits = 40, repeat = 2, benchmark = 0, defaults = 0, gpu = 0, custom = 0;
        std::string address = "127.0.0.1";
        ArgMapping args;
        args.arg("r", party, "Party"); args.arg("p", port, "Port"); args.arg("ip", address, "Server IP");
        args.arg("nt", threads, "Worker threads"); args.arg("n", degree, "Polynomial degree");
        args.arg("bits", bits, "Plaintext modulus bits"); args.arg("repeat", repeat, "Repeated calls");
        args.arg("bench", benchmark, "Compare Cheetah"); args.arg("default_keys", defaults, "Use legacy rotation keys");
        args.arg("gpu", gpu, "Use GPU HE evaluation");
        args.arg("custom", custom, "Use a custom RNS prime chain");
        args.parse(argc, argv);
        if ((party != 1 && party != 2) || threads < 1 || repeat < 1) throw std::invalid_argument("Invalid test arguments");
        Utils::NetIO io(party == 1 ? nullptr : address.c_str(), port);
        if (custom && degree != 8192) throw std::invalid_argument("Custom test chain requires N=8192");
        HE::HEEvaluator he(&io, party, degree, bits, gpu ? Datatype::DEVICE : Datatype::HOST,
            custom ? std::vector<int>{43,43,43,43,43} : std::vector<int>{});
        he.GenerateNewKey(defaults ? std::vector<uint32_t>{} : LinearLayer::Conv2DOptiEncode::GaloisElements(degree));
        std::vector<Case> cases{
            {3,5,4,3,1,1,0,0,false,false}, {9,7,7,3,2,1,4,2,false,false},
            {5,3,3,1,1,0,2,0,false,false}, {2,3,2,3,1,2,2,1,false,false},
            {3,5,6,2,2,0,4,2,false,false}, {2,3,4,3,1,1,2,2,true,false},
            {3,5,3,1,1,0,4,0,false,true}, {1,1,1,1,1,0,1,0,false,false},
            {3,5,100,3,2,1,0,0,false,false}, {16,32,14,3,1,1,0,0,false,false},
            {3,5,6,3,1,1,64,0,false,false}};
        if (degree >= 8192) cases.push_back({3,5,3,3,1,1,128,0,false,false});
        const size_t boundary_c = degree >= 8192 ? 128 : 64;
        cases.push_back({boundary_c + 1, boundary_c + 3, 8, 1, 1, 0, boundary_c, 0, false, false});
        cases.push_back({5,7,4,3,1,1,8,0,false,false,1});
        cases.push_back({5,7,4,3,1,1,8,0,false,false,8});
        cases.push_back({3,5,7,1,3,0,4,0,false,false});
        cases.push_back({64,128,14,3,1,1,0,0,false,false});
        for (size_t i = 0; i < cases.size(); ++i) RunCase(cases[i], i, he, threads, repeat, benchmark && i == cases.size() - 1);
        he.FreeKey();
        std::cout << "ALL OPTIENCODE CONVOLUTION CHECKS PASSED" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
