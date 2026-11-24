#include <Model/FaceResNet.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace Model {

namespace {

constexpr uint64_t unit = uint64_t(1) << FaceResNet64::scale;
constexpr uint64_t threshold_scale = uint64_t(1) << 12;
constexpr uint64_t ring_half = uint64_t(1) << (FaceResNet64::bitwidth - 1);

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
    return value >> 63 ? (modulus - (uint64_t(0) - value) % modulus) % modulus : value % modulus;
}

} // namespace

FaceResNet64::Layout FaceResNet64::MakeLayout(Backbone backbone) {
    std::vector<size_t> depths;
    switch (backbone) {
        case Backbone::Compact: depths = {1, 1, 1}; break;
        case Backbone::ResNet18: depths = {2, 2, 2, 2}; break;
        case Backbone::ResNet20: depths = {3, 3, 3}; break;
        case Backbone::ResNet32: depths = {5, 5, 5}; break;
        case Backbone::ResNet34: depths = {3, 4, 6, 3}; break;
    }
    const bool use_stem_max_pool = depths.size() == 4;
    const size_t stem_channels = use_stem_max_pool ? 64 : 16;
    const size_t stem_stride = use_stem_max_pool ? 2 : 1;
    const size_t stem_kernel = use_stem_max_pool ? 7 : 3;
    Layout result;
    result.stem_max_pool = use_stem_max_pool;
    result.convolutions.push_back({image_size, 3, stem_channels, stem_kernel, stem_stride});
    size_t spatial = image_size / stem_stride / (use_stem_max_pool ? 2 : 1);
    size_t in_channels = stem_channels;
    for (size_t stage = 0; stage < depths.size(); ++stage) {
        const size_t out_channels = stem_channels << stage;
        for (size_t block = 0; block < depths[stage]; ++block) {
            const size_t stride = stage != 0 && block == 0 ? 2 : 1;
            const bool shortcut = stride != 1;
            result.blocks.push_back({result.convolutions.size(), shortcut});
            result.convolutions.push_back({spatial, in_channels, out_channels, 3, stride});
            result.convolutions.push_back({spatial / stride, out_channels, out_channels, 3, 1});
            if (shortcut) {
                result.convolutions.push_back({spatial, in_channels, out_channels, 1, stride});
            }
            spatial /= stride;
            in_channels = out_channels;
        }
    }
    result.pool_size = spatial;
    result.convolutions.push_back({1, in_channels, embedding_size, 1, 1});
    return result;
}

std::vector<FaceResNet64::ConvShape> FaceResNet64::ConvolutionShapes(Backbone backbone) {
    return MakeLayout(backbone).convolutions;
}

FaceResNet64::Weights FaceResNet64::EncodeWeights(const FloatWeights &weights) {
    const auto encode = [](const Datatype::Tensor<float> &input, int fractional_bits) {
        const double limit = std::ldexp(static_cast<double>(ring_half - 1), -fractional_bits);
        for (const float value : input.data()) {
            if (!std::isfinite(value) || std::abs(static_cast<double>(value)) > limit) {
                throw std::invalid_argument("FaceResNet64 FP32 parameter exceeds the fixed-point range");
            }
        }
        return Datatype::Tensor<uint64_t>::FromFloatTensorToFixed(input, fractional_bits);
    };
    Weights encoded(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        encoded[i] = {encode(weights[i].weight, scale), encode(weights[i].bias, 2 * scale)};
    }
    return encoded;
}

FaceResNet64::FaceResNet64(CryptoPrimitive<uint64_t> &crypto, const Weights *weights,
                           bool use_mm_conversion, ConvolutionMode mode, Backbone backbone, size_t he_threads)
    : crypto(crypto), use_mm_conversion(use_mm_conversion), layout(MakeLayout(backbone)),
      pool(layout.pool_size), convolutions(layout.convolutions.size()) {
    if (!crypto.HE || !crypto.fixpoint || !crypto.relu || crypto.relu->bitwidth != bitwidth ||
        crypto.HE->plain_mod <= (uint64_t(1) << 33) || crypto.fixpoint->party != crypto.party ||
        crypto.HE->server != (crypto.party == ALICE)) {
        throw std::invalid_argument("FaceResNet64 requires the OpenCheetah 37-bit ring and a BFV plaintext modulus greater than 2^33");
    }
    if (crypto.HE->polyModulusDegree < 8192) {
        throw std::invalid_argument("FaceResNet64 requires a BFV polynomial degree of at least 8192");
    }
    if (crypto.HE->server != (weights != nullptr)) {
        throw std::invalid_argument("FaceResNet64 weights must be supplied by the server");
    }
    if (weights) {
        if (weights->size() != convolutions.size()) {
            throw std::invalid_argument("FaceResNet64 weight count does not match the selected backbone");
        }
        for (size_t i = 0; i < convolutions.size(); ++i) {
            const auto &shape = layout.convolutions[i];
            const auto &parameters = (*weights)[i];
            if (parameters.weight.shape() != std::vector<size_t>{shape.out_channels, shape.in_channels, shape.kernel, shape.kernel} ||
                parameters.bias.shape() != std::vector<size_t>{shape.out_channels} ||
                !InRange(parameters.weight, ring_half - 1) || !InRange(parameters.bias, ring_half - 1)) {
                throw std::invalid_argument("Invalid FaceResNet64 weights or bias");
            }
        }
    }
    using PartitionCost = LinearLayer::Conv2DCheetah::PartitionCost;
    const bool baseline = mode == ConvolutionMode::CheetahBaseline;
    const size_t convolution_threads = baseline ? 1 : he_threads;
    const auto cheetah_cost = baseline ? PartitionCost::InputCiphertexts : PartitionCost::TotalCiphertexts;
    for (size_t i = 0; i < convolutions.size(); ++i) {
        const auto &shape = layout.convolutions[i];
        if (mode == ConvolutionMode::OptiEncode) {
            LinearLayer::OptiEncodeOptions options;
            options.threads = he_threads;
            options.add_bias = false; // Convolve adds the signed bias during share conversion.
            if (weights) {
                auto field_weight = (*weights)[i].weight;
                for (auto &value : field_weight.data())
                    value = FieldResidue(value, crypto.HE->plain_mod);
                convolutions[i] = std::make_unique<LinearLayer::Conv2DOptiEncode>(
                    shape.size, shape.stride, shape.kernel / 2, field_weight, (*weights)[i].bias,
                    crypto.HE, options);
            } else {
                convolutions[i] = std::make_unique<LinearLayer::Conv2DOptiEncode>(
                    shape.size, shape.in_channels, shape.out_channels, shape.kernel, shape.stride,
                    crypto.HE, options);
            }
            continue;
        }
        const bool use_cheetah = baseline || shape.size == image_size;
        if (weights) {
            const auto &parameters = (*weights)[i];
            if (use_cheetah) {
                convolutions[i] = std::make_unique<LinearLayer::Conv2DCheetah>(
                    shape.size, shape.stride, shape.kernel / 2, parameters.weight, parameters.bias, crypto.HE,
                    cheetah_cost, convolution_threads);
            } else {
                auto field_weight = parameters.weight;
                for (auto &value : field_weight.data()) {
                    value = FieldResidue(value, crypto.HE->plain_mod);
                }
                convolutions[i] = std::make_unique<LinearLayer::CirConv2D>(
                    shape.size, shape.stride, shape.kernel / 2, 1, field_weight, parameters.bias, crypto.HE,
                    convolution_threads);
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
    const uint64_t before_preprocessing = communication ? crypto.get_preprocessing_comm() : 0;
    const uint64_t modulus = crypto.HE->plain_mod;
    if (use_mm_conversion) {
        crypto.fixpoint->BoundedRing2Field(input, modulus, bitwidth);
    } else {
        crypto.fixpoint->SignedRing2Field(input, modulus, bitwidth);
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
        communication->linear_preprocessing_bytes += crypto.get_preprocessing_comm() - before_preprocessing;
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
    auto x = Convolve(index, input, communication);
    Activate(x);
    x = Convolve(index + 1, std::move(x), communication);
    auto skip = shortcut ? Convolve(index + 2, input, communication) : input;
    x = x + skip;
    Activate(x);
    return x;
}

void FaceResNet64::Activate(Datatype::Tensor<uint64_t> &input) {
    (*crypto.relu)(input);
    input.bitwidth = bitwidth;
    input.scale = scale;
}

Datatype::Tensor<uint64_t> FaceResNet64::StemMaxPool(const Datatype::Tensor<uint64_t> &input) {
    const size_t channels = input.shape()[0];
    const size_t height = input.shape()[1];
    const size_t size = height / 2;
    const size_t plane = channels * size * size;
    size_t window = 9;
    Datatype::Tensor<uint64_t> values({window, channels, size, size});
    // Padding by zero is equivalent to negative infinity after ReLU.
    for (size_t kr = 0; kr < 3; ++kr) {
        for (size_t kc = 0; kc < 3; ++kc) {
            for (size_t r = 0; r < size; ++r) {
                const size_t rr = 2 * r + kr;
                if (rr == 0 || rr > height) continue;
                for (size_t c = 0; c < size; ++c) {
                    const size_t cc = 2 * c + kc;
                    if (cc == 0 || cc > height) continue;
                    for (size_t ch = 0; ch < channels; ++ch) {
                        values({3 * kr + kc, ch, r, c}) = input({ch, rr - 1, cc - 1});
                    }
                }
            }
        }
    }
    while (window > 1) {
        const size_t pairs = window / 2;
        Datatype::Tensor<uint64_t> differences({pairs, plane});
        for (size_t pair = 0; pair < pairs; ++pair) {
            const size_t left = 2 * pair * plane;
            for (size_t i = 0; i < plane; ++i) {
                differences(pair * plane + i) = values(left + i) - values(left + plane + i);
            }
        }
        (*crypto.relu)(differences);
        for (size_t pair = 0; pair < pairs; ++pair) {
            const size_t right = (2 * pair + 1) * plane;
            for (size_t i = 0; i < plane; ++i) {
                values(pair * plane + i) = differences(pair * plane + i) + values(right + i);
            }
        }
        if (window % 2 != 0) {
            std::copy_n(values.data().begin() + (window - 1) * plane, plane, values.data().begin() + pairs * plane);
        }
        window = (window + 1) / 2;
    }
    Datatype::Tensor<uint64_t> output({channels, size, size}, bitwidth, scale);
    std::copy_n(values.data().begin(), plane, output.data().begin());
    return output;
}

Datatype::Tensor<uint64_t> FaceResNet64::Forward(const Datatype::Tensor<uint64_t> &image,
                                               CommunicationStats *communication) {
    using Clock = std::chrono::steady_clock;
    const auto start = communication ? Clock::now() : Clock::time_point{};
    CheckImage(image);
    auto x = Convolve(0, image, communication);
    Activate(x);
    if (layout.stem_max_pool) {
        x = StemMaxPool(x);
    }
    for (const auto &block : layout.blocks) {
        x = ResidualBlock(block.index, x, block.shortcut, communication);
    }
    x = pool.ForwardShared(x, *crypto.fixpoint, bitwidth);
    x = Clamp(Convolve(convolutions.size() - 1, std::move(x), communication), -static_cast<int64_t>(unit), unit);
    x.reshape({embedding_size});
    if (communication) {
        communication->forward_seconds += std::chrono::duration<double>(Clock::now() - start).count();
    }
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
    // Q12 dot products and squared norms fit in the 37-bit model ring.
    // Lift only the three scalars before the degree-four comparison.
    constexpr int comparison_bitwidth = 64, threshold_bits = 12;
    Datatype::Tensor<uint64_t> wide({3}, {dot, norm_x, norm_y});
    if (fixpoint.party == ALICE) {
        for (auto &value : wide.data()) {
            value += ring_half;
        }
    }
    fixpoint.extend(wide, bitwidth, comparison_bitwidth);
    if (fixpoint.party == ALICE) {
        for (auto &value : wide.data()) {
            value -= ring_half;
        }
    }
    Datatype::Tensor<uint64_t> left({2}, {wide(0), wide(1)});
    Datatype::Tensor<uint64_t> right({2}, {wide(0), wide(2)});
    const auto squared = fixpoint.Multiply(left, right, comparison_bitwidth);

    // Split the norm product before multiplying by the Q12 threshold.
    // This gives ceil(norm_product * threshold_squared / threshold_scale)
    // exactly, while all reconstructed intermediates fit in 64 bits.
    Datatype::Tensor<uint64_t> high({1}, {squared(1)});
    fixpoint.truncate(high, threshold_bits, comparison_bitwidth, false, false);
    Datatype::Tensor<uint64_t> low({1},
        {(squared(1) - high(0) * threshold_scale) * threshold_squared});
    if (fixpoint.party == ALICE) {
        low(0) += threshold_scale - 1;
    }
    fixpoint.truncate(low, threshold_bits, comparison_bitwidth, false, false);
    Datatype::Tensor<uint64_t> checks({2},
        {wide(0), squared(0) - high(0) * threshold_squared - low(0)});
    if (fixpoint.party == ALICE) {
        checks(0) -= 1;
    }
    Datatype::Tensor<uint8_t> negative({2});
    fixpoint.less_than_zero(checks, negative, comparison_bitwidth);
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
