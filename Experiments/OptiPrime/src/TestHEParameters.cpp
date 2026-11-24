#include <LinearLayer/ConvOptiEncode.h>
#include <Utils/ArgMapping/ArgMapping.h>
#include <iostream>
#include <random>
#include <sstream>

namespace {
using Datatype::Tensor;
uint64_t Field(int64_t v, uint64_t t) {
    return v < 0 ? (t - uint64_t(-v) % t) % t : uint64_t(v) % t;
}
void Check(HE::HEEvaluator &he, size_t ci, size_t co, size_t h, size_t k, size_t c, int pattern) {
    const uint64_t t = he.plain_mod;
    Tensor<int64_t> x({ci, h, h}), w({co, ci, k, k});
    std::mt19937_64 rng(91 + pattern);
    for (auto &v : x.data())
        v = (pattern == 0 || pattern == 1)
                ? 255
                : (pattern >= 3 ? int64_t(rng() % 256) : int64_t(rng() % 256) - 128);
    for (auto &v : w.data())
        v = pattern == 0 ? 7 : (pattern == 1 ? -8 : (pattern == 4 ? 0 : int64_t(rng() % 16) - 8));
    Tensor<uint64_t> input(x.shape()), weight(w.shape()), bias({co});
    for (size_t i = 0; i < input.size(); ++i) {
        auto r = rng() % t;
        input(i) = he.server ? r : (Field(x(i), t) + t - r) % t;
    }
    if (he.server)
        for (size_t i = 0; i < weight.size(); ++i)
            weight(i) = Field(w(i), t);
    LinearLayer::OptiEncodeOptions options;
    options.threads = 4;
    options.ciphertext_inverse_normalization = !(t & 1);
    options.channels_per_block = c;
    LinearLayer::Conv2DOptiEncode conv(h, 1, k / 2, weight, bias, &he, options);
    auto out = conv(input);
    Tensor<uint64_t> peer(out.shape());
    if (he.server)
        he.IO->recv_data(peer.data().data(), peer.size() * 8);
    else {
        he.IO->send_data(out.data().data(), out.size() * 8);
        he.IO->flush();
    }
    uint64_t errors = 0;
    if (he.server)
        for (size_t o = 0; o < co; ++o)
            for (size_t r = 0; r < h; ++r)
                for (size_t col = 0; col < h; ++col) {
                    int64_t sum = 0;
                    for (size_t i = 0; i < ci; ++i)
                        for (size_t kr = 0; kr < k; ++kr)
                            for (size_t kc = 0; kc < k; ++kc) {
                                const int64_t row = int64_t(r + kr) - int64_t(k / 2),
                                              column = int64_t(col + kc) - int64_t(k / 2);
                                if (row >= 0 && column >= 0 && row < int64_t(h) &&
                                    column < int64_t(h))
                                    sum += x({i, size_t(row), size_t(column)}) * w({o, i, kr, kc});
                            }
                    const size_t index = (o * h + r) * h + col;
                    errors += (out(index) + peer(index)) % t != Field(sum, t);
                }
    if (he.server) {
        he.IO->send_data(&errors, 8);
        he.IO->flush();
    } else
        he.IO->recv_data(&errors, 8);
    if (errors)
        throw std::runtime_error("Insufficient HE parameters or incorrect convolution: " +
                                 std::to_string(errors));
    if (he.server)
        std::cout << "PASS low-HE ci=" << ci << " co=" << co << " h=" << h << " k=" << k
                  << " C=" << conv.Layout().channels << " pattern=" << pattern << std::endl;
}
} // namespace
int main(int argc, char **argv) {
    try {
        int party = 1, port = 22000, gpu = 0, bits = 26, po2 = 0;
        std::string qbits = "44,44,44";
        ArgMapping args;
        args.arg("r", party);
        args.arg("p", port);
        args.arg("gpu", gpu);
        args.arg("bits", bits);
        args.arg("qbits", qbits);
        args.arg("po2", po2);
        args.parse(argc, argv);
        std::vector<int> q;
        std::stringstream text(qbits);
        std::string token;
        while (std::getline(text, token, ','))
            q.push_back(std::stoi(token));
        Utils::NetIO io(party == 1 ? nullptr : "127.0.0.1", port);
        HE::HEEvaluator he(&io, party, 8192, bits, gpu ? Datatype::DEVICE : Datatype::HOST, q,
                           po2 ? uint64_t(1) << bits : 0);
        he.GenerateNewKey(LinearLayer::Conv2DOptiEncode::GaloisElements(8192));
        for (int pattern = 0; pattern < 3; ++pattern)
            Check(he, 513, 13, 7, 3, 128, pattern);
        Check(he, 513, 131, 5, 3, 128, 3);
        Check(he, 64, 128, 14, 3, 0, 2);
        Check(he, 3, 64, 17, 7, 0, 2);
        Check(he, 1, 2, 7, 3, 1, 1);
        Check(he, 3, 5, 7, 3, 2, 4);
        he.FreeKey();
        std::cout << "PROTOCOL CHECK COMPLETE" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
