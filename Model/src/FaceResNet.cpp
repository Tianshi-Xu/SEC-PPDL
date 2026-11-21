#include <Model/FaceResNet.h>
#include <cmath>
#include <stdexcept>

namespace Model {

namespace {

struct ConvShape {
    size_t size;
    size_t in_channels;
    size_t out_channels;
    size_t kernel;
    size_t stride;
};

constexpr std::array<ConvShape, 10> conv_shapes{{
    {64, 3, 16, 3, 1},
    {64, 16, 16, 3, 1}, {64, 16, 16, 3, 1},
    {64, 16, 32, 3, 2}, {32, 32, 32, 3, 1}, {64, 16, 32, 1, 2},
    {32, 32, 64, 3, 2}, {16, 64, 64, 3, 1}, {32, 32, 64, 1, 2},
    {1, 64, FaceResNet64::embedding_size, 1, 1},
}};

constexpr uint64_t unit = uint64_t(1) << FaceResNet64::scale;
constexpr uint64_t threshold_scale = uint64_t(1) << 12;

uint64_t QuantizeThreshold(double threshold) {
    if (!std::isfinite(threshold) || threshold <= 0.0 || threshold > 1.0) {
        throw std::invalid_argument("Face verification threshold must be in (0, 1]");
    }
    return std::max(uint64_t(1), static_cast<uint64_t>(std::ceil(threshold * threshold * threshold_scale)));
}

bool InRange(const Datatype::Tensor<uint64_t> &tensor, uint64_t limit) {
    for (const uint64_t value : tensor.data()) {
        if (value > limit && uint64_t(0) - value > limit) {
            return false;
        }
    }
    return true;
}

void CheckImage(const Datatype::Tensor<uint64_t> &image) {
    if (image.shape() != std::vector<size_t>{3, FaceResNet64::image_size, FaceResNet64::image_size}) {
        throw std::invalid_argument("FaceResNet64 expects a 3x64x64 crop");
    }
}

uint64_t FieldResidue(uint64_t value, uint64_t modulus) {
    return value >> 63 ? modulus - (uint64_t(0) - value) : value;
}

} // namespace

FaceResNet64::FaceResNet64(CryptoPrimitive<uint64_t> &crypto, const Weights *weights,
                           bool use_mm_conversion, ConvolutionMode mode)
    : crypto(crypto), use_mm_conversion(use_mm_conversion) {
    if (!crypto.HE || !crypto.fixpoint || !crypto.relu || crypto.relu->bitwidth != bitwidth ||
        crypto.HE->plain_mod <= (uint64_t(1) << 33) || crypto.fixpoint->party != crypto.party ||
        crypto.HE->server != (crypto.party == ALICE)) {
        throw std::invalid_argument("FaceResNet64 requires a 60-bit ring and a BFV plaintext modulus greater than 2^33");
    }
    if (crypto.HE->polyModulusDegree < 8192) {
        throw std::invalid_argument("FaceResNet64 requires a BFV polynomial degree of at least 8192");
    }
    if (crypto.HE->server != (weights != nullptr)) {
        throw std::invalid_argument("FaceResNet64 weights must be supplied by the server");
    }
    using PartitionCost = LinearLayer::Conv2DCheetah::PartitionCost;
    const bool baseline = mode == ConvolutionMode::CheetahBaseline;
    const auto cheetah_cost = baseline ? PartitionCost::InputCiphertexts : PartitionCost::TotalCiphertexts;
    for (size_t i = 0; i < conv_shapes.size(); ++i) {
        const auto &shape = conv_shapes[i];
        const bool use_cheetah = baseline || shape.size == image_size;
        const std::vector<size_t> weight_shape{shape.out_channels, shape.in_channels, shape.kernel, shape.kernel};
        const std::vector<size_t> bias_shape{shape.out_channels};
        if (weights) {
            const auto &parameters = (*weights)[i];
            if (parameters.weight.shape() != weight_shape || parameters.bias.shape() != bias_shape ||
                !InRange(parameters.weight, 4 * unit) || !InRange(parameters.bias, 4 * unit * unit)) {
                throw std::invalid_argument("Invalid FaceResNet64 weights or bias");
            }
            if (use_cheetah) {
                convolutions[i] = std::make_unique<LinearLayer::Conv2DCheetah>(
                    shape.size, shape.stride, shape.kernel / 2, parameters.weight, parameters.bias, crypto.HE, cheetah_cost);
            } else {
                auto field_weight = parameters.weight;
                for (auto &value : field_weight.data()) {
                    value = FieldResidue(value, crypto.HE->plain_mod);
                }
                convolutions[i] = std::make_unique<LinearLayer::CirConv2D>(
                    shape.size, shape.stride, shape.kernel / 2, 1, field_weight, parameters.bias, crypto.HE);
            }
        } else if (use_cheetah) {
            convolutions[i] = std::make_unique<LinearLayer::Conv2DCheetah>(
                shape.size, shape.in_channels, shape.out_channels, shape.kernel, shape.stride, crypto.HE, cheetah_cost);
        } else {
            convolutions[i] = std::make_unique<LinearLayer::CirConv2D>(
                shape.size, shape.in_channels, shape.out_channels, shape.kernel, shape.stride, 1, crypto.HE);
        }
    }
}

Datatype::Tensor<uint64_t> FaceResNet64::Convolve(size_t index, Datatype::Tensor<uint64_t> input,
                                                CommunicationStats *communication) {
    const uint64_t before_layer = communication ? crypto.get_total_comm() : 0;
    const uint64_t modulus = crypto.HE->plain_mod;
    // Every activation entering a convolution is within [-8, 8] at Q8.
    if (use_mm_conversion) {
        crypto.fixpoint->BoundedRing2Field(input, modulus, 16);
    } else {
        crypto.fixpoint->SignedRing2Field(input, modulus, 16);
    }
    const uint64_t before_conv = communication ? crypto.get_total_comm() : 0;
    auto output = (*convolutions[index])(input);
    if (communication) {
        communication->convolution_bytes += crypto.get_total_comm() - before_conv;
    }
    if (crypto.HE->server) {
        const auto &bias = convolutions[index]->bias;
        const size_t plane = output.shape()[1] * output.shape()[2];
        const uint64_t offset = use_mm_conversion ? 0 : modulus / 2;
        for (size_t i = 0; i < output.size(); ++i) {
            const uint64_t residue = FieldResidue(bias(i / plane), modulus);
            output(i) = (output(i) + residue + offset) % modulus;
        }
    }
    if (use_mm_conversion) {
        crypto.fixpoint->BoundedField2Ring(output, modulus, bitwidth, scale);
    } else {
        crypto.fixpoint->Field2Ring(output, modulus, bitwidth);
        if (crypto.HE->server) {
            for (auto &value : output.data()) {
                value -= modulus / 2;
            }
        }
        crypto.fixpoint->truncate(output, scale, bitwidth, false, true);
    }
    output.bitwidth = bitwidth;
    output.scale = scale;
    if (communication) {
        communication->linear_layer_bytes += crypto.get_total_comm() - before_layer;
    }
    return output;
}

Datatype::Tensor<uint64_t> FaceResNet64::Clamp(const Datatype::Tensor<uint64_t> &input, int64_t low, int64_t high) {
    auto above_low = input;
    auto above_high = input;
    if (crypto.HE->server) {
        for (size_t i = 0; i < input.size(); ++i) {
            above_low(i) -= static_cast<uint64_t>(low);
            above_high(i) -= static_cast<uint64_t>(high);
        }
    }
    (*crypto.relu)(above_low);
    (*crypto.relu)(above_high);
    auto result = above_low - above_high;
    if (crypto.HE->server) {
        for (auto &value : result.data()) {
            value += static_cast<uint64_t>(low);
        }
    }
    result.bitwidth = bitwidth;
    result.scale = scale;
    return result;
}

Datatype::Tensor<uint64_t> FaceResNet64::ResidualBlock(size_t index, const Datatype::Tensor<uint64_t> &input,
                                                     bool shortcut, CommunicationStats *communication) {
    auto x = Clamp(Convolve(index, input, communication), 0, 8 * unit);
    x = Convolve(index + 1, std::move(x), communication);
    auto skip = shortcut ? Convolve(index + 2, input, communication) : input;
    return Clamp(x + skip, 0, 8 * unit);
}

Datatype::Tensor<uint64_t> FaceResNet64::Forward(const Datatype::Tensor<uint64_t> &image,
                                               CommunicationStats *communication) {
    CheckImage(image);
    auto x = Clamp(Convolve(0, image, communication), 0, 8 * unit);
    x = ResidualBlock(1, x, false, communication);
    x = ResidualBlock(3, x, true, communication);
    x = ResidualBlock(6, x, true, communication);
    x = pool.ForwardShared(x, *crypto.fixpoint, bitwidth);
    x = Clamp(Convolve(9, std::move(x), communication), -static_cast<int64_t>(unit), unit);
    x.reshape({embedding_size});
    return x;
}

uint8_t FaceResNet64::MatchEmbeddings(const Datatype::Tensor<uint64_t> &x, const Datatype::Tensor<uint64_t> &y,
                                     double threshold, NonlinearOperator::FixPoint<uint64_t> &fixpoint) {
    const uint64_t threshold_squared = QuantizeThreshold(threshold);
    const std::vector<size_t> shape{embedding_size};
    if (x.shape() != shape || y.shape() != shape) {
        throw std::invalid_argument("Face verification expects two 128-dimensional embeddings");
    }
    Datatype::Tensor<uint64_t> lhs({3, embedding_size});
    Datatype::Tensor<uint64_t> rhs({3, embedding_size});
    for (size_t i = 0; i < embedding_size; ++i) {
        lhs({0, i}) = lhs({1, i}) = x(i);
        rhs({0, i}) = lhs({2, i}) = rhs({2, i}) = y(i);
        rhs({1, i}) = x(i);
    }
    const auto products = fixpoint.Multiply(lhs, rhs, bitwidth);
    uint64_t dot = 0, norm_x = 0, norm_y = 0;
    for (size_t i = 0; i < embedding_size; ++i) {
        dot += products({0, i});
        norm_x += products({1, i});
        norm_y += products({2, i});
    }
    Datatype::Tensor<uint64_t> left({2}, {dot, norm_x});
    Datatype::Tensor<uint64_t> right({2}, {dot, norm_y});
    const auto squared = fixpoint.Multiply(left, right, bitwidth);
    // The largest magnitude in either comparison is 2^58 in the 60-bit ring.
    Datatype::Tensor<uint64_t> checks({2}, {dot, squared(0) * threshold_scale - squared(1) * threshold_squared});
    if (fixpoint.party == ALICE) {
        checks(0) -= 1;
    }
    Datatype::Tensor<uint8_t> negative({2});
    fixpoint.less_than_zero(checks, negative, bitwidth);
    const uint8_t flip = fixpoint.party == ALICE;
    return fixpoint.BooleanAnd(negative(0) ^ flip, negative(1) ^ flip);
}

bool FaceResNet64::Verify(const Datatype::Tensor<uint64_t> &first,
                          const Datatype::Tensor<uint64_t> &second, double threshold,
                          CommunicationStats *communication) {
    CheckImage(first);
    CheckImage(second);
    QuantizeThreshold(threshold);
    const auto first_embedding = Forward(first, communication);
    const auto second_embedding = Forward(second, communication);
    const uint8_t share = MatchEmbeddings(first_embedding, second_embedding, threshold, *crypto.fixpoint);
    uint8_t peer_share = 0;
    auto *io = crypto.HE->IO;
    if (crypto.HE->server) {
        io->send_data(&share, sizeof(share));
        io->flush();
        io->recv_data(&peer_share, sizeof(peer_share));
    } else {
        io->recv_data(&peer_share, sizeof(peer_share));
        io->send_data(&share, sizeof(share));
        io->flush();
    }
    return (share ^ peer_share) == 1;
}

} // namespace Model
