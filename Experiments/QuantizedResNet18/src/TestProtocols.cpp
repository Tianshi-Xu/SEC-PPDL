#include <QuantExperiment/Protocols.h>
#include <Utils/ArgMapping/ArgMapping.h>
#include <iostream>
#include <random>

namespace {
using Datatype::Tensor;
Tensor<uint64_t> Share(const Tensor<int64_t> &plain, int party, int bits) {
    std::mt19937_64 rng(42);
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    Tensor<uint64_t> result(plain.shape());
    for (size_t i = 0; i < plain.size(); ++i) {
        const auto r = rng() & mask;
        result(i) = party == 1 ? r : (uint64_t(plain(i)) - r) & mask;
    }
    return result;
}
Tensor<int64_t> Open(QuantExperiment::Context &context, const Tensor<uint64_t> &input, int bits,
                     bool field = false) {
    Tensor<uint64_t> peer(input.shape());
    auto &io = *context.he->IO;
    if (context.config.party == 1) {
        io.send_data(input.data().data(), input.size() * 8);
        io.flush();
        io.recv_data(peer.data().data(), peer.size() * 8);
    } else {
        io.recv_data(peer.data().data(), peer.size() * 8);
        io.send_data(input.data().data(), input.size() * 8);
        io.flush();
    }
    Tensor<int64_t> out(input.shape());
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    for (size_t i = 0; i < input.size(); ++i) {
        uint64_t v =
            field ? (input(i) + peer(i)) % context.he->plain_mod : (input(i) + peer(i)) & mask;
        out(i) = field
                     ? (v > context.he->plain_mod / 2 ? int64_t(v) - int64_t(context.he->plain_mod)
                                                      : int64_t(v))
                     : (v & (uint64_t(1) << (bits - 1)) ? int64_t(v) - (int64_t(1) << bits)
                                                        : int64_t(v));
    }
    return out;
}
void Check(const Tensor<int64_t> &a, const Tensor<int64_t> &b, const char *label) {
    if (a.shape() != b.shape() || a.data() != b.data())
        throw std::runtime_error(label);
}
void Suite(QuantExperiment::Context &context, bool verify) {
    using namespace QuantExperiment;
    Tensor<int64_t> plain({517});
    for (size_t i = 0; i < plain.size(); ++i)
        plain(i) = int64_t(i % 511) - 255;
    auto shares = Share(plain, context.config.party, context.config.activation_ring_bits);
    context.RingToField(shares);
    if (verify)
        Check(Open(context, shares, 32, true), plain, "Parallel ring to field mismatch");
    context.FieldToRing(shares);
    if (verify)
        Check(Open(context, shares, context.config.accumulator_bits), plain,
              "Parallel field to ring mismatch");
    auto signed_wide = shares;
    context.ExtendSigned(signed_wide, context.config.accumulator_bits,
                         context.config.ClassifierBits());
    if (verify)
        Check(Open(context, signed_wide, context.config.ClassifierBits()), plain,
              "Signed classifier extension mismatch");
    Tensor<int64_t> expected(plain.shape());
    for (size_t i = 0; i < plain.size(); ++i)
        expected(i) = std::clamp(plain(i), int64_t(0), int64_t(255));
    auto quantized = Requantize(context, shares, 0, 0);
    if (verify)
        Check(Open(context, quantized, context.config.activation_ring_bits), expected,
              "Quantization clamp mismatch");
    context.ExtendUnsigned(quantized, context.config.activation_ring_bits,
                           context.config.accumulator_bits);
    if (verify)
        Check(Open(context, quantized, context.config.accumulator_bits), expected,
              "Unsigned extension tail mismatch");
    // Check signed floor truncation on both sides of quantization boundaries.
    Tensor<int64_t> boundaries({517}), requantized_reference({517});
    for (size_t i = 0; i < boundaries.size(); ++i) {
        boundaries(i) = int64_t((i * 79) % 40001) - 20000;
        const auto v = boundaries(i);
        requantized_reference(i) = std::clamp(v / 8 - (v % 8 < 0) + 128, int64_t(0), int64_t(255));
    }
    auto shifted = Requantize(
        context, Share(boundaries, context.config.party, context.config.accumulator_bits), 3, 128,
        context.config.power_of_two ? 16 : std::max(16, context.config.activation_ring_bits));
    if (verify)
        Check(Open(context, shifted, context.config.activation_ring_bits), requantized_reference,
              "Signed exact requantization mismatch");
    if (context.config.power_of_two) {
        const int bits = context.config.accumulator_bits;
        context.config.approximate_relu = true;
        for (const int zero_point : {0, 128}) {
            for (const int shift : {5, 6, 7, 8, 9}) {
                const int64_t divisor = int64_t(1) << shift;
                const int64_t edges[] = {-2, -1, 0, 1, 254, 255, 256, 511};
                for (size_t i = 0; i < boundaries.size(); ++i)
                    boundaries(i) =
                        (edges[i % 8] - zero_point) * divisor + int64_t((i / 8) % 3) - 1;
                auto approximate = Requantize(
                    context, Share(boundaries, context.config.party, bits), shift, zero_point);
                if (verify) {
                    const auto opened = Open(context, approximate, bits);
                    for (size_t i = 0; i < opened.size(); ++i) {
                        const auto positive =
                            std::max<int64_t>(0, boundaries(i) + zero_point * divisor);
                        if (opened(i) != positive / divisor &&
                            opened(i) != (positive + divisor - 1) / divisor)
                            throw std::runtime_error(
                                "Known-zero approximate shift exceeded one LSB");
                    }
                }
            }
        }
        context.config.approximate_relu = false;
    }
    Tensor<int64_t> plane({512, 7, 7}), average({512, 1, 1});
    for (size_t c = 0; c < 512; ++c)
        for (size_t j = 0; j < 49; ++j)
            plane(c * 49 + j) = (c * 7 + j * 11) % 256;
    const int totals[] = {0, 12495, 48, 49, 50, 12494};
    for (size_t c = 0; c < 6; ++c) {
        int left = totals[c];
        for (size_t j = 0; j < 49; ++j) {
            plane(c * 49 + j) = std::min(left, 255);
            left -= plane(c * 49 + j);
        }
    }
    for (size_t c = 0; c < 512; ++c) {
        for (size_t j = 0; j < 49; ++j)
            average(c) += plane(c * 49 + j);
        average(c) /= 49;
    }
    auto pooled = GlobalAverage(
        context, Share(plane, context.config.party, context.config.activation_ring_bits));
    if (verify)
        Check(Open(context, pooled, context.config.activation_ring_bits), average,
              "Exact global mean mismatch");
    if (context.config.power_of_two) {
        const int bits = context.config.accumulator_bits;
        Tensor<int64_t> large({8, 7, 7}), reference({8, 1, 1});
        for (size_t i = 0; i < large.size(); ++i) {
            large(i) = (int64_t(1) << (bits - 1)) - 1 - int64_t(i % 97);
            reference(i / 49) += large(i);
        }
        for (auto &v : reference.data())
            v /= 49;
        auto wide = GlobalAverage(context, Share(large, context.config.party, bits), bits + 5);
        if (verify)
            Check(Open(context, wide, bits), reference, "Widened 7x7 mean mismatch");
        const int scores_bits=context.config.ClassifierBits();
        const int64_t limit=int64_t(1) << (scores_bits-1);
        Tensor<int64_t> scores({4}, {-limit,limit-1,limit-1,0});
        auto winner=Classify(context, Share(scores,context.config.party,scores_bits));
        if (verify && Open(context,winner,context.config.ArgmaxBits())(0)!=1)
            throw std::runtime_error("Signed argmax failed at opposite ring endpoints");
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        QuantExperiment::Config cfg;
        int ot = 0, prep = 1, parallel = 1, legacy = 0, po2 = 0, po2bits = 26;
        cfg.linear = QuantExperiment::LinearBackend::CPU;
        cfg.threads = 4;
        ArgMapping args;
        args.arg("r", cfg.party);
        args.arg("p", cfg.port);
        args.arg("nt", cfg.threads);
        args.arg("accbits", cfg.accumulator_bits);
        args.arg("bits", cfg.plaintext_bits);
        args.arg("ot", ot);
        args.arg("prep", prep);
        args.arg("parallel", parallel);
        args.arg("legacy", legacy);
        args.arg("po2", po2);
        args.arg("po2bits", po2bits);
        args.parse(argc, argv);
        cfg.ot = ot ? QuantExperiment::OTBackend::Ferret : QuantExperiment::OTBackend::IKNP;
        cfg.preprocess = prep;
        cfg.parallel_conversions = parallel;
        cfg.legacy_conversion = legacy;
        cfg.power_of_two = po2;
        cfg.ciphertext_inverse = po2;
        if (legacy) {
            cfg.activation_ring_bits = 60;
            cfg.accumulator_bits = 60;
            cfg.plaintext_bits = 50;
            cfg.ciphertext_prime_bits = {};
        }
        if (po2) {
            cfg.activation_ring_bits = po2bits;
            cfg.accumulator_bits = po2bits;
            cfg.plaintext_bits = po2bits + 1;
        }
        QuantExperiment::Context context(cfg);
        context.BeginPlanning();
        Suite(context, false);
        context.EndPlanning();
        context.Sync();
        for (int round = 0; round < 2; ++round) {
            context.Preprocess();
            context.Sync();
            Suite(context, true);
            context.CheckConsumed();
            context.Sync();
            if (cfg.party == 1)
                std::cout << "PASS quant-protocols nt=" << cfg.threads << " ot=" << ot
                          << " prep=" << prep << " parallel=" << parallel << " legacy=" << legacy
                          << " round=" << round << std::endl;
        }
        std::cout << "PROTOCOL CHECK COMPLETE" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
