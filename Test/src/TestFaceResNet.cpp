#include <Model/FaceResNet.h>
#include <LinearLayer/CirLinear.h>
#include <array>
#include <chrono>

using Model::FaceResNet64;
using Datatype::Tensor;

namespace {

constexpr int64_t unit = int64_t(1) << FaceResNet64::scale;
using Clock = std::chrono::steady_clock;

double ElapsedSeconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

template<typename T>
T Exchange(Utils::NetIO &io, int party, T share) {
    T peer{};
    if (party == ALICE) {
        io.send_data(&share, sizeof(share));
        io.flush();
        io.recv_data(&peer, sizeof(peer));
    } else {
        io.recv_data(&peer, sizeof(peer));
        io.send_data(&share, sizeof(share));
        io.flush();
    }
    return peer;
}

uint64_t Share(int64_t value, size_t index, int party) {
    const uint64_t mask = 0x9e3779b97f4a7c15ULL * (index + 1);
    return party == ALICE ? mask : static_cast<uint64_t>(value) - mask;
}

void CheckBoundedConversion(Model::CryptoPrimitive<uint64_t> &crypto) {
    constexpr uint64_t modulus = 251, mask = 255;
    constexpr int shift = 3;
    constexpr std::array<int64_t, 7> values{-62, -17, -1, 0, 1, 17, 63};
    constexpr size_t count = values.size() * (mask + 1);
    auto &fixpoint = *crypto.fixpoint;
    auto &io = *crypto.HE->IO;
    Tensor<uint64_t> ring({count}), field({count});
    for (size_t i = 0; i < count; ++i) {
        const auto value = values[i / (mask + 1)];
        const uint64_t share = i & mask;
        ring(i) = crypto.party == ALICE ? share : (static_cast<uint64_t>(value) - share) & mask;
        const uint64_t field_share = share % modulus;
        field(i) = crypto.party == ALICE ? field_share : (2 * modulus + value - field_share) % modulus;
    }
    fixpoint.BoundedRing2Field(ring, modulus, 8);
    auto truncated = field;
    fixpoint.BoundedField2Ring(field, modulus, 8);
    fixpoint.BoundedField2Ring(truncated, modulus, 8, shift);

    std::array<uint64_t, 3 * count> local{};
    for (size_t i = 0; i < count; ++i) {
        local[i] = ring(i);
        local[count + i] = field(i);
        local[2 * count + i] = truncated(i);
    }
    const auto peer = Exchange(io, crypto.party, local);
    for (size_t i = 0; i < count; ++i) {
        const auto value = values[i / (mask + 1)];
        const auto expected_field = (modulus + value) % modulus;
        if ((ring(i) + peer[i]) % modulus != expected_field ||
            ((field(i) + peer[count + i]) & mask) != (static_cast<uint64_t>(value) & mask)) {
            throw std::runtime_error("Bounded modulus conversion failed");
        }
        const int64_t divisor = int64_t(1) << shift;
        const int64_t exact = value / divisor - (value % divisor < 0);
        const uint64_t error = (truncated(i) + peer[2 * count + i] - static_cast<uint64_t>(exact)) & mask;
        if (error != 0 && error != 1 && error != mask) {
            throw std::runtime_error("Fused truncation exceeded one output unit of error");
        }
    }
}

void CheckProtocols(Model::CryptoPrimitive<uint64_t> &crypto) {
    constexpr size_t dim = FaceResNet64::embedding_size;
    constexpr uint64_t ring_mask = (uint64_t(1) << FaceResNet64::bitwidth) - 1;
    const Tensor<uint64_t> plain_input({1, 1, 1}, {(uint64_t(1) << 62) + 4});
    const NonlinearLayer::AvgPool2D<uint64_t> padded_pool(2, 1, 1);
    const auto plain_output = padded_pool(plain_input);
    if (plain_output.shape() != std::vector<size_t>{1, 2, 2} ||
        !std::all_of(plain_output.data().begin(), plain_output.data().end(),
            [](uint64_t value) { return value == (uint64_t(1) << 60) + 1; })) {
        throw std::runtime_error("Padded integer average pooling failed");
    }

    auto &io = *crypto.HE->IO;
    auto &fixpoint = *crypto.fixpoint;
    const uint64_t modulus = crypto.HE->plain_mod;
    Tensor<uint64_t> field({3});
    const std::array<int64_t, 3> values{-unit, 0, unit};
    for (size_t i = 0; i < values.size(); ++i) {
        field(i) = Share(values[i], i, crypto.party);
    }
    fixpoint.SignedRing2Field(field, modulus, FaceResNet64::bitwidth);
    for (size_t i = 0; i < values.size(); ++i) {
        const uint64_t expected = static_cast<uint64_t>(static_cast<int64_t>(modulus) + values[i]) % modulus;
        if ((field(i) + Exchange(io, crypto.party, field(i))) % modulus != expected) {
            throw std::runtime_error("Signed ring-to-field conversion failed");
        }
    }
    Tensor<uint64_t> input({1, 2, 2});
    for (size_t i = 0; i < input.size(); ++i) {
        input(i) = Share(-static_cast<int64_t>(i) - 1, i, crypto.party);
    }
    NonlinearLayer::AvgPool2D<uint64_t> pool(2);
    const auto pooled = pool.ForwardShared(input, fixpoint, FaceResNet64::bitwidth);
    const auto peer = Exchange(io, crypto.party, pooled(0));
    if (((pooled(0) + peer) & ring_mask) != (uint64_t(-3) & ring_mask)) {
        throw std::runtime_error("Signed shared average pooling failed");
    }

    Tensor<uint64_t> x({dim});
    Tensor<uint64_t> y({dim});
    const std::array<int64_t, 4> second{-unit, 0, unit / 2, unit};
    for (const int64_t component : second) {
        for (size_t i = 0; i < dim; ++i) {
            x(i) = Share(i == 0 ? unit : 0, i, crypto.party);
            y(i) = Share(i == 0 ? component : 0, i + dim, crypto.party);
        }
        const uint8_t share = FaceResNet64::MatchEmbeddings(x, y, 1.0, fixpoint);
        if ((share ^ Exchange(io, crypto.party, share)) != (component > 0)) {
            throw std::runtime_error("Private cosine match failed");
        }
    }
    for (size_t i = 0; i < dim; ++i) {
        x(i) = Share(i == 0 ? unit : 0, i, crypto.party);
        y(i) = Share(i < 2 ? unit : 0, i + dim, crypto.party);
    }
    for (double threshold : {0.5, 0.75}) {
        const uint8_t share = FaceResNet64::MatchEmbeddings(x, y, threshold, fixpoint);
        if ((share ^ Exchange(io, crypto.party, share)) != (threshold == 0.5)) {
            throw std::runtime_error("Private cosine threshold failed");
        }
    }
    for (size_t i = 0; i < dim; ++i) {
        x(i) = Share(unit, i, crypto.party);
        y(i) = Share(unit, i + dim, crypto.party);
    }
    const uint8_t share = FaceResNet64::MatchEmbeddings(x, y, 1.0, fixpoint);
    if ((share ^ Exchange(io, crypto.party, share)) != 1) {
        throw std::runtime_error("Private cosine match failed at the Q12 norm limit");
    }
}

FaceResNet64::Weights MakeWeights(FaceResNet64::Backbone backbone, bool dense) {
    const auto shapes = FaceResNet64::ConvolutionShapes(backbone);
    FaceResNet64::FloatWeights weights(shapes.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        const auto &shape = shapes[i];
        const size_t center = shape.kernel / 2;
        weights[i].weight = Tensor<float>({shape.out_channels, shape.in_channels, shape.kernel, shape.kernel});
        weights[i].bias = Tensor<float>({shape.out_channels});
        if (dense) {
            for (size_t j = 0; j < weights[i].weight.size(); ++j) {
                weights[i].weight(j) = (j % 2 == 0 ? 1.0f : -1.0f) / unit;
            }
        }
        if (i == 0) {
            weights[i].weight({0, 0, center, center}) = 1.0f;
            weights[i].weight({1, 0, center, center}) = -1.0f;
        } else {
            const float coefficient = shape.kernel == 1 ? 1.0f : 0.25f;
            weights[i].weight({0, 0, center, center}) = coefficient;
            weights[i].weight({1, 1, center, center}) = coefficient;
        }
    }
    return FaceResNet64::EncodeWeights(weights);
}

void CheckProjection(Model::CryptoPrimitive<uint64_t> &crypto, size_t he_threads) {
    constexpr size_t in_channels = 64, out_channels = FaceResNet64::embedding_size;
    const uint64_t modulus = crypto.HE->plain_mod;
    auto &io = *crypto.HE->IO;
    Tensor<uint64_t> input({in_channels, 1, 1});
    Tensor<uint64_t> kernel({out_channels, in_channels, 1, 1});
    Tensor<uint64_t> matrix({in_channels, out_channels});
    Tensor<uint64_t> conv_bias({out_channels});
    Tensor<uint64_t> linear_bias({1, out_channels});
    std::array<int64_t, out_channels> expected{};
    for (size_t in = 0; in < in_channels; ++in) {
        const int64_t value = static_cast<int64_t>(in % 11) - 5;
        const uint64_t mask = Share(0, in, ALICE) % modulus;
        input(in) = crypto.party == ALICE ? mask : (2 * modulus + value - mask) % modulus;
        for (size_t out = 0; out < out_channels; ++out) {
            const int64_t coeff = static_cast<int64_t>((3 * in + 5 * out) % 7) - 3;
            kernel({out, in, 0, 0}) = matrix({in, out}) = (modulus + coeff) % modulus;
            expected[out] += value * coeff;
        }
    }

    LinearLayer::Conv2DCheetah cheetah(1, 1, 0, kernel, conv_bias, crypto.HE,
        LinearLayer::Conv2DCheetah::PartitionCost::InputCiphertexts, he_threads);
    LinearLayer::CirConv2D circonv(1, 1, 0, 1, kernel, conv_bias, crypto.HE, he_threads);
    auto cheetah_output = cheetah(input);
    auto conv_output = circonv(input);
    input.reshape({1, in_channels});
    auto linear = crypto.party == ALICE
        ? LinearLayer::CirLinearNest(1, 1, matrix, linear_bias, crypto.HE)
        : LinearLayer::CirLinearNest(1, in_channels, out_channels, 1, crypto.HE);
    auto linear_output = linear(input);
    if (conv_output.shape() != std::vector<size_t>{out_channels, 1, 1} ||
        linear_output.shape() != std::vector<size_t>{1, out_channels}) {
        throw std::runtime_error("Unexpected pointwise projection shape");
    }
    for (size_t out = 0; out < out_channels; ++out) {
        const uint64_t expected_residue = (modulus + expected[out]) % modulus;
        for (uint64_t share : {cheetah_output(out), conv_output(out), linear_output(out)}) {
            if ((share + Exchange(io, crypto.party, share)) % modulus != expected_residue) {
                throw std::runtime_error("Pointwise projection differs from the matrix product");
            }
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    int optiencode = 0;
    int party = 0, port = 32000, num_threads = 3, full = 0, mm = 0, baseline = 0, vole = 0, checks = 1;
    int backbone_arg = 0, dense = 0, he_threads = FaceResNet64::default_he_threads;
    std::string address = "127.0.0.1";
    ArgMapping arguments;
    arguments.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    arguments.arg("p", port, "Port number");
    arguments.arg("ip", address, "IP address of ALICE");
    arguments.arg("full", full, "Also verify two 64x64 images");
    arguments.arg("mm", mm, "Enable M&M conversion and fused truncation");
    arguments.arg("opti", optiencode, "Use OptiEncode for all convolutions");
    arguments.arg("baseline", baseline, "Use Cheetah for all convolutions with the original partition cost");
    arguments.arg("vole", vole, "Use Ferret VOLE-style OT instead of IKNP");
    arguments.arg("checks", checks, "Run operator checks before the full benchmark");
    arguments.arg("backbone", backbone_arg, "Backbone: compact = 0, otherwise 18, 20, 32, or 34");
    arguments.arg("dense", dense, "Use synthetic weights with all kernel coefficients nonzero");
    arguments.arg("he_threads", he_threads, "CPU workers for server-side hybrid convolutions");
    arguments.parse(argc, argv);
    if (party != ALICE && party != BOB) {
        return 1;
    }
    if (he_threads < 1) {
        throw std::invalid_argument("he_threads must be positive");
    }
    using Backbone = FaceResNet64::Backbone;
    Backbone backbone = Backbone::Compact;
    switch (backbone_arg) {
        case 0: break;
        case 18: backbone = Backbone::ResNet18; break;
        case 20: backbone = Backbone::ResNet20; break;
        case 32: backbone = Backbone::ResNet32; break;
        case 34: backbone = Backbone::ResNet34; break;
        default: throw std::invalid_argument("backbone must be 0, 18, 20, 32, or 34");
    }

    if (baseline && optiencode) throw std::invalid_argument("Choose one convolution backend");
    const auto ot_type = vole ? Datatype::VOLE : Datatype::IKNP;
    const auto setup_start = Clock::now();
    Model::CryptoPrimitive<uint64_t> crypto(party, num_threads, FaceResNet64::bitwidth, ot_type,
        8192, 50, optiencode ? Datatype::OptiEncode : Datatype::Cheetah, Datatype::HOST, address, port);
    const auto setup_end = Clock::now();
    const uint64_t setup_bytes = crypto.get_total_comm();
    const uint64_t setup_preprocessing_bytes = crypto.get_preprocessing_comm();
    std::cout << "Session setup seconds: " << ElapsedSeconds(setup_start, setup_end) << std::endl;
    if (checks) {
        const auto checks_start = Clock::now();
        CheckProtocols(crypto);
        CheckBoundedConversion(crypto);
        CheckProjection(crypto, he_threads);
        std::cout << "Operator checks seconds: " << ElapsedSeconds(checks_start, Clock::now()) << std::endl;
    }
    if (full) {
        const auto mode = optiencode ? FaceResNet64::ConvolutionMode::OptiEncode
                        : baseline ? FaceResNet64::ConvolutionMode::CheetahBaseline
                                   : FaceResNet64::ConvolutionMode::Hybrid;
        const auto packing_start = Clock::now();
        const auto weights = party == ALICE ? MakeWeights(backbone, dense != 0) : FaceResNet64::Weights{};
        FaceResNet64 model(crypto, party == ALICE ? &weights : nullptr, mm != 0, mode, backbone, he_threads);
        const auto packing_end = Clock::now();
        Tensor<uint64_t> x({3, 64, 64}), y({3, 64, 64});
        for (size_t i = 0; i < x.size(); ++i) {
            const int64_t value = i < 64 * 64 ? unit : 0;
            x(i) = Share(value, i, party);
            y(i) = Share(-value, i + x.size(), party);
        }
        // Start both timers after each party has prepared its model and inputs.
        Exchange(*crypto.HE->IO, party, uint8_t(0));
        FaceResNet64::CommunicationStats communication;
        const uint64_t before_verify = crypto.get_total_comm();
        const uint64_t before_preprocessing = crypto.get_preprocessing_comm();
        const auto verification_start = Clock::now();
        if (model.Verify(x, y, 0.5, &communication)) {
            throw std::runtime_error("End-to-end face verification accepted orthogonal embeddings");
        }
        const auto first_pair_end = Clock::now();
        if (!model.Verify(x, x, 0.5, &communication)) {
            throw std::runtime_error("End-to-end face verification rejected identical embeddings");
        }
        const auto second_pair_end = Clock::now();
        std::cout << "Local model packing seconds: " << ElapsedSeconds(packing_start, packing_end) << std::endl;
        std::cout << "First verification seconds: " << ElapsedSeconds(verification_start, first_pair_end) << std::endl;
        std::cout << "Second verification seconds: " << ElapsedSeconds(first_pair_end, second_pair_end) << std::endl;
        std::cout << "Average forward seconds per image: " << communication.forward_seconds / 4 << std::endl;
        std::cout << "Convolution bytes sent (two pairs): " << communication.convolution_bytes << std::endl;
        std::cout << "Linear-layer bytes sent (two pairs): " << communication.linear_layer_bytes << std::endl;
        const uint64_t verification_bytes = crypto.get_total_comm() - before_verify;
        const uint64_t verification_preprocessing = crypto.get_preprocessing_comm() - before_preprocessing;
        std::cout << "Verification bytes sent (two pairs): " << verification_bytes << std::endl;
        if (vole) {
            std::cout << "OT setup and warm-up bytes sent (one session): " << setup_preprocessing_bytes << std::endl;
            std::cout << "Other setup bytes sent (one session): " << setup_bytes - setup_preprocessing_bytes << std::endl;
            std::cout << "OT refill bytes sent (two pairs): " << verification_preprocessing << std::endl;
            std::cout << "Online linear-layer bytes sent (two pairs): "
                      << communication.linear_layer_bytes - communication.linear_preprocessing_bytes << std::endl;
            std::cout << "Online verification bytes sent (two pairs): "
                      << verification_bytes - verification_preprocessing << std::endl;
        }
    }
    std::cout << "FaceResNet64 checks passed" << std::endl;
    return 0;
}
