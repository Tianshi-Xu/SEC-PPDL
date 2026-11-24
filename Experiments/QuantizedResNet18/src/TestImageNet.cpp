// Accuracy harness for public test images only. The observer reconstructs activations.
#include <QuantExperiment/Bundle.h>
#include <Utils/ArgMapping/ArgMapping.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

namespace {
using Datatype::Tensor;
using namespace QuantExperiment;
namespace fs = std::filesystem;

template <typename T> std::vector<T> Read(std::istream &in, size_t n) {
    std::vector<T> v(n);
    if (!in.read(reinterpret_cast<char *>(v.data()), n * sizeof(T)))
        throw std::runtime_error("Incomplete model or test vector");
    return v;
}
void Magic(std::istream &in, const std::string &wanted) {
    const auto bytes = Read<char>(in, wanted.size());
    if (std::string(bytes.begin(), bytes.end()) != wanted)
        throw std::runtime_error("Unexpected bundle format");
}
std::vector<int32_t> OpenAtServer(Context &ctx, const Tensor<uint64_t> &shares, int bits) {
    auto &io = *ctx.he->IO;
    if (ctx.config.party == 2) {
        io.send_data(shares.data().data(), shares.size() * sizeof(uint64_t));
        io.flush();
        return {};
    }
    std::vector<uint64_t> peer(shares.size());
    io.recv_data(peer.data(), peer.size() * sizeof(uint64_t));
    std::vector<int32_t> result(shares.size());
    const int64_t mod = int64_t(1) << bits;
    for (size_t i = 0; i < result.size(); ++i) {
        const int64_t value = (shares(i) + peer[i]) & (mod - 1);
        const auto signed_value = value >= mod / 2 ? value - mod : value;
        if (signed_value < INT32_MIN || signed_value > INT32_MAX)
            throw std::runtime_error("Opened value exceeds the signed test-vector format");
        result[i] = static_cast<int32_t>(signed_value);
    }
    return result;
}
} // namespace

int main(int argc, char **argv) {
    try {
        using namespace QuantExperiment;
        Config cfg;
        std::string bundle, output, qbits;
        int gpu = 0, approximate = 0, count = 1, trace_count = 0, cheetah = 0;
        int ferret = 1, preprocess = 1;
        ArgMapping args;
        args.arg("r", cfg.party);
        args.arg("p", cfg.port);
        args.arg("nt", cfg.threads);
        args.arg("bundle", bundle);
        args.arg("results", output);
        args.arg("gpu", gpu);
        args.arg("ferret", ferret);
        args.arg("prep", preprocess);
        args.arg("approx_relu", approximate);
        args.arg("count", count);
        args.arg("trace", trace_count);
        args.arg("qbits", qbits);
        args.arg("cheetah", cheetah);
        args.arg("address", cfg.address);
        args.parse(argc, argv);
        const fs::path root(bundle), results(output);
        ImageNetBundle trained(root);
        const int party = cfg.party, port = cfg.port, threads = cfg.threads;
        const auto address = cfg.address;
        cfg = trained.InferenceConfig();
        cfg.party = party;
        cfg.port = port;
        cfg.threads = threads;
        cfg.address = address;
        cfg.linear = gpu ? LinearBackend::GPU : LinearBackend::CPU;
        cfg.approximate_relu = approximate;
        cfg.ot = ferret ? OTBackend::Ferret : OTBackend::IKNP;
        cfg.preprocess = preprocess;
        if (!qbits.empty()) {
            cfg.ciphertext_prime_bits.clear();
            std::istringstream primes(qbits);
            for (std::string prime; std::getline(primes, prime, ',');)
                cfg.ciphertext_prime_bits.push_back(std::stoi(prime));
        }
        if (cheetah) {
            if (approximate || gpu)
                throw std::runtime_error("Cheetah baseline requires CPU exact mode");
            cfg.linear = LinearBackend::Cheetah;
            cfg.preprocess = cfg.power_of_two = cfg.ciphertext_inverse = false;
            cfg.legacy_conversion = true;
            cfg.activation_ring_bits = cfg.accumulator_bits = 60;
            cfg.plaintext_bits = 50;
            cfg.ciphertext_prime_bits.clear();
            cfg.ot = OTBackend::IKNP;
        }
        auto weights = cfg.party == 1 ? trained.LoadWeights() : Weights{};
        std::ifstream cases(root / "cases.bin", std::ios::binary);
        Magic(cases, cfg.activation_bits > 8 ? "QIC2" : "QIC1");
        if (count < 1 || uint32_t(count) > Read<uint32_t>(cases, 1)[0] || trace_count < 0 ||
            trace_count > count)
            throw std::runtime_error("Invalid image or trace count");
        std::ofstream predictions;
        if (cfg.party == 1) {
            predictions.exceptions(std::ios::failbit | std::ios::badbit);
            predictions.open(results / "predictions.jsonl");
        }
        Context context(cfg);
        QuantizedImageNetModel model(context, cfg.party == 1 ? &weights : nullptr, trained.metadata,
                                     trained.steps);
        weights.clear();
        if (cfg.preprocess)
            QuantizedImageNetModel::Plan(context, trained.metadata, trained.steps);
        for (int i = 0; i < count; ++i) {
            const auto label = Read<uint32_t>(cases, 1)[0];
            Tensor<uint64_t> input({3, 224, 224});
            auto load_input = [&](const auto &x) {
                if (cfg.party == 2)
                    std::copy(x.begin(), x.end(), input.data().begin());
            };
            if (cfg.activation_bits > 8)
                load_input(Read<uint16_t>(cases, input.size()));
            else
                load_input(Read<uint8_t>(cases, input.size()));
            const auto expected = Read<int32_t>(cases, 1000);
            Profile profile;
            std::ofstream trace;
            if (i < trace_count) {
                if (cfg.party == 1) {
                    trace.exceptions(std::ios::failbit | std::ios::badbit);
                    trace.open(results / ("trace-" + std::to_string(i) + ".bin"), std::ios::binary);
                    trace.write("QTR2", 4);
                }
                profile.trace_layout_steps = true;
                profile.observe_activation = [&](const Tensor<uint64_t> &shares) {
                    const auto values = OpenAtServer(context, shares, cfg.activation_ring_bits);
                    if (cfg.party == 1) {
                        const uint32_t size = values.size();
                        trace.write(reinterpret_cast<const char *>(&size), sizeof(size));
                        trace.write(reinterpret_cast<const char *>(values.data()),
                                    size * sizeof(int32_t));
                    }
                };
            }
            using Clock = std::chrono::steady_clock;
            context.Sync();
            const auto start = Clock::now();
            const auto start_bytes = context.Bytes();
            context.Preprocess();
            context.Sync();
            const auto online = Clock::now();
            const auto online_bytes = context.Bytes();
            const auto logits = model.Forward(input, profile);
            const auto index = Classify(context, logits);
            const auto finish = Clock::now();
            const auto finish_bytes = context.Bytes();
            double linear_seconds = 0;
            uint64_t linear_bytes = 0;
            for (auto component :
                 {profile.ring_to_field, profile.convolution, profile.fully_connected,
                  profile.field_to_ring, profile.classifier_tail}) {
                linear_seconds += component.seconds;
                linear_bytes += component.bytes;
            }
            std::cout << "INFERENCE {\"image\":" << i << ",\"online_seconds\":"
                      << std::chrono::duration<double>(finish - online).count()
                      << ",\"online_bytes\":" << finish_bytes - online_bytes
                      << ",\"offline_seconds\":"
                      << std::chrono::duration<double>(online - start).count()
                      << ",\"offline_bytes\":" << online_bytes - start_bytes
                      << ",\"linear_seconds\":" << linear_seconds
                      << ",\"linear_bytes\":" << linear_bytes << "}" << std::endl;
            const auto winner = OpenAtServer(context, index, cfg.ArgmaxBits());
            const auto actual = OpenAtServer(context, logits, cfg.ClassifierBits());
            context.CheckConsumed();
            if (cfg.party == 1) {
                if (winner.at(0) != std::max_element(actual.begin(), actual.end()) - actual.begin())
                    throw std::runtime_error("Secure argmax mismatch at image " +
                                             std::to_string(i));
                if (!approximate && actual != expected)
                    throw std::runtime_error("Exact logit mismatch at image " + std::to_string(i));
                predictions << "{\"index\":" << i << ",\"label\":" << label
                            << ",\"winner\":" << winner[0] << ",\"logits\":[";
                for (size_t j = 0; j < actual.size(); ++j)
                    predictions << (j ? "," : "") << actual[j];
                predictions << "]}" << std::endl;
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
