#include <QuantExperiment/Model.h>
#include <QuantExperiment/SharedSelection.h>

namespace QuantExperiment {
using Datatype::Tensor;
namespace {
void ValidateTopology(const std::vector<LayerMetadata> &layers, const std::vector<Step> &steps) {
    size_t count = 0;
    std::vector<std::pair<size_t, size_t>> shapes{{3, 224}};
    for (const auto &step : steps) {
        auto [channels, size] = shapes.at(step.input);
        if (step.op == Operation::Conv) {
            const auto &s = layers.at(count++).shape;
            if (s.size != size || s.input_channels != channels || !s.output_channels || !s.stride ||
                !s.kernel || size + 2 * s.padding < s.kernel)
                throw std::invalid_argument("Invalid convolution dimensions in model layout");
            size = (size + 2 * s.padding - s.kernel) / s.stride + 1;
            channels = s.output_channels;
        } else if (step.op == Operation::Residual) {
            if (shapes.at(step.skip) != shapes.at(step.input))
                throw std::invalid_argument("Mismatched residual branch");
        } else if (step.op == Operation::Flatten) {
            channels *= size * size;
            size = 1;
        } else if (step.op == Operation::GlobalAverage) {
            if (size != 7)
                throw std::invalid_argument("Expected 7x7 global average");
            size = 1;
        } else {
            if (step.op == Operation::MaxPool3 && size != 112)
                throw std::invalid_argument("Expected ResNet stem pool");
            size = step.op == Operation::MaxPool3 ? (size + 1) / 2 : size / 2;
        }
        shapes.emplace_back(channels, size);
    }
    if (count != layers.size() || steps.empty() || steps.back().op != Operation::Conv ||
        shapes.back() != std::make_pair(size_t(1000), size_t(1)))
        throw std::invalid_argument("Expected a 1000-logit classifier");
}
uint64_t Field(uint64_t v, uint64_t t) {
    return v >> 63 ? (t - (uint64_t(0) - v) % t) % t : v % t;
}
template <typename Conv>
Tensor<uint64_t> Graph(Context &context, Tensor<uint64_t> input, Conv convolution, Profile &profile,
                       const std::vector<Step> &steps, const std::vector<LayerMetadata> &metadata) {
    std::vector<Tensor<uint64_t>> values;
    values.reserve(steps.size() + 1);
    values.push_back(std::move(input));
    // Release each feature map after its last consumer, including residual shortcuts.
    std::vector<size_t> uses(steps.size() + 1, 0);
    for (const auto &step : steps) {
        ++uses[step.input];
        if (step.op == Operation::Residual)
            ++uses[step.skip];
    }
    auto take = [&](size_t id) -> Tensor<uint64_t> {
        if (--uses[id] == 0)
            return std::move(values[id]);
        return values[id];
    };
    std::vector<uint64_t> upper{uint64_t(context.config.ActivationMax())};
    size_t layer = 0;
    for (const auto &step : steps) {
        auto x = take(step.input);
        uint64_t bound = upper[step.input];
        switch (step.op) {
        case Operation::Conv:
            bound = context.config.approximate_relu
                        ? uint64_t(1)
                              << (context.config.accumulator_bits - 1 - metadata[layer].shift)
                        : uint64_t(context.config.ActivationMax());
            x = convolution(layer++, std::move(x));
            break;
        case Operation::MaxPool3:
        case Operation::MaxPool2: {
            Timer timer(context, profile.maxpool);
            x = step.op == Operation::MaxPool3
                    ? MaxPoolQuantized(context, x)
                    : MaxPoolShared(x, 2, 2, 0, *context.fix,
                                                    context.config.activation_ring_bits);
            break;
        }
        case Operation::Residual: {
            Timer timer(context, profile.residual);
            if (context.config.approximate_relu)
                bound = std::min(bound + upper[step.skip],
                                 (uint64_t(1) << (context.config.accumulator_bits - 1)) - 1);
            x = x + take(step.skip);
            if (context.config.party == 1)
                for (auto &v : x.data())
                    v -= context.config.ActivationMidpoint() * (1 + int(step.projection));
            x = context.config.approximate_relu
                    ? LowerRelu(context, x)
                    : Clamp(context, x, 0, context.config.ActivationMax(),
                            context.config.power_of_two ? context.config.activation_bits + 2
                                                        : context.config.activation_ring_bits,
                            context.config.activation_ring_bits);
            Reduce(x, context.config.activation_ring_bits);
            if (profile.observe_activation)
                profile.observe_activation(x);
            break;
        }
        case Operation::GlobalAverage: {
            Timer timer(context, profile.globalpool);
            int sum_bits = 0;
            for (uint64_t limit = 49 * bound; limit; limit >>= 1)
                ++sum_bits;
            x = GlobalAverage(context, std::move(x), sum_bits);
            break;
        }
        case Operation::Flatten:
            x.reshape({x.size(), 1, 1});
            break;
        }
        if (profile.trace_layout_steps && profile.observe_activation &&
            step.op != Operation::Conv && step.op != Operation::Residual)
            profile.observe_activation(x);
        upper.push_back(bound);
        values.push_back(std::move(x));
    }
    auto logits = std::move(values.back());
    logits.reshape({1000});
    return logits;
}
void BeforeConvolution(Context &context, const LayerMetadata &meta, Tensor<uint64_t> &input,
                       Profile &profile) {
    Timer timer(context, profile.ring_to_field);
    if (context.config.party == 1)
        for (auto &v : input.data())
            v -= meta.input_zero_point;
    context.RingToField(input);
}
void AfterConvolution(Context &context, const LayerMetadata &meta, Tensor<uint64_t> &output,
                      bool final, Profile &profile) {
    {
        Timer timer(context, profile.field_to_ring);
        context.FieldToRing(output);
    }
    if (!final) {
        Timer timer(context, profile.requantization);
        output = Requantize(context, std::move(output), meta.shift, meta.output_zero_point,
                            context.config.legacy_conversion ? context.config.accumulator_bits
                                                             : meta.clamp_bits,
                            &profile.requantization_detail);
        if (profile.observe_activation)
            profile.observe_activation(output);
    }
}
} // namespace
QuantizedImageNetModel::QuantizedImageNetModel(Context &context, const Weights *weights,
                                               std::vector<LayerMetadata> metadata,
                                               std::vector<Step> steps)
    : context_(context), metadata_(std::move(metadata)), steps_(std::move(steps)) {
    ValidateTopology(metadata_, steps_);
    if ((context.config.party == 1) != (weights != nullptr) ||
        (weights && weights->size() != metadata_.size()))
        throw std::invalid_argument("Only the server supplies the quantized weights");
    LinearLayer::OptiEncodeOptions options;
    options.threads = context.config.threads;
    options.add_bias = false;
    options.output_mod_switches = context.config.output_mod_switches;
    options.ciphertext_inverse_normalization = context.config.ciphertext_inverse;
    for (size_t index = 0; index < metadata_.size(); ++index) {
        const auto &s = metadata_[index].shape;
        const auto &m = metadata_[index];
        if ((!m.multipliers.empty() &&
             (index + 1 == metadata_.size() || m.multipliers.size() != s.output_channels)) ||
            m.weight_bits < 2 || m.weight_bits > 8)
            throw std::invalid_argument("Invalid imported quantization parameters");
        Tensor<uint64_t> w({s.output_channels, s.input_channels, s.kernel, s.kernel}),
            b({s.output_channels});
        if (weights) {
            const auto &p = (*weights)[index];
            if (p.weight.shape() != w.shape() || p.bias.shape() != b.shape())
                throw std::invalid_argument("Quantized layer shape mismatch");
            w = p.weight;
            b = p.bias;
            for (auto &v : w.data()) {
                const int bound = 1 << (m.weight_bits - 1);
                if (static_cast<int64_t>(v) < -bound || static_cast<int64_t>(v) >= bound)
                    throw std::invalid_argument("Weight is outside declared quantization range");
                v = Field(v, context.he->plain_mod);
            }
            const size_t width = w.size() / s.output_channels;
            for (size_t o = 0; o < s.output_channels; ++o) {
                __uint128_t bound = b(o) >> 63 ? uint64_t(0) - b(o) : b(o);
                bound += __uint128_t(width) * (1 << (m.weight_bits - 1)) *
                         context.config.ActivationMax() *
                         (m.multipliers.empty() ? 1 : m.multipliers[o]);
                if (!context.config.empirical_range && bound >= context.he->plain_mod / 4)
                    throw std::invalid_argument(
                        "Plaintext modulus does not cover the W4A8 accumulator");
                if (!context.config.empirical_range &&
                    bound >= (__uint128_t(1) << (context.config.accumulator_bits - 2)))
                    throw std::invalid_argument(
                        "Accumulator ring does not cover the certified signed range");
                const uint64_t bias_magnitude = b(o) >> 63 ? uint64_t(0) - b(o) : b(o);
                if (m.multipliers.empty() && index + 1 < metadata_.size() &&
                    bias_magnitude > (uint64_t(8) << m.shift))
                    throw std::invalid_argument("Bias exceeds the certified dyadic clamp bound");
            }
        }
        if (context.config.linear == LinearBackend::Cheetah ||
            (s.size == 1 && s.input_channels >= 4096)) {
            auto layer = std::make_unique<LinearLayer::Conv2DCheetah>(
                s.size, s.stride, s.padding, w, b, context.he.get(),
                LinearLayer::Conv2DCheetah::PartitionCost::InputCiphertexts, context.config.threads);
            layers_.emplace_back(std::move(layer));
        } else
            layers_.emplace_back(std::make_unique<LinearLayer::Conv2DOptiEncode>(
                s.size, s.stride, s.padding, w, b, context.he.get(), options));
    }
}
Tensor<uint64_t> QuantizedImageNetModel::Convolve(size_t index, Tensor<uint64_t> input,
                                                  Profile &profile) {
    const auto &meta = metadata_[index];
    BeforeConvolution(context_, meta, input, profile);
    Tensor<uint64_t> output;
    const bool final = index + 1 == metadata_.size();
    {
        Timer timer(context_, final ? profile.fully_connected : profile.convolution);
        output = (*layers_[index])(input);
    }
    {
        Timer timer(context_, final ? profile.fully_connected : profile.convolution);
        if (!meta.multipliers.empty()) {
            const size_t plane = output.size() / meta.multipliers.size();
            for (size_t i = 0; i < output.size(); ++i)
                output(i) =
                    context_.config.power_of_two
                        ? (output(i) * meta.multipliers[i / plane]) & (context_.he->plain_mod - 1)
                        : uint64_t(__uint128_t(output(i)) * meta.multipliers[i / plane] %
                                   context_.he->plain_mod);
        }
        if (context_.config.party == 1 && !final) {
            const auto &bias = layers_[index]->bias;
            const size_t plane = output.size() / bias.size();
            for (size_t i = 0; i < output.size(); ++i)
                output(i) = (output(i) + Field(bias(i / plane), context_.he->plain_mod)) %
                            context_.he->plain_mod;
        }
    }
    AfterConvolution(context_, meta, output, final, profile);
    if (final) {
        Timer timer(context_, profile.classifier_tail);
        context_.ExtendSigned(output, context_.config.accumulator_bits,
                              context_.config.ClassifierBits());
        if (context_.config.party == 1)
            for (size_t i = 0; i < output.size(); ++i)
                output(i) += layers_[index]->bias(i);
        Reduce(output, context_.config.ClassifierBits());
    }
    return output;
}
Tensor<uint64_t> QuantizedImageNetModel::Forward(const Tensor<uint64_t> &input, Profile &profile) {
    if (input.shape() != std::vector<size_t>{3, 224, 224})
        throw std::invalid_argument("Expected 3x224x224 integer activation input");
    return Graph(
        context_, input,
        [&](size_t index, Tensor<uint64_t> x) { return Convolve(index, std::move(x), profile); },
        profile, steps_, metadata_);
}
void QuantizedImageNetModel::Plan(Context &context, const std::vector<LayerMetadata> &metadata,
                                  const std::vector<Step> &steps) {
    ValidateTopology(metadata, steps);
    Profile profile;
    context.BeginPlanning();
    Tensor<uint64_t> public_zero_input({3, 224, 224});
    auto logits = Graph(
        context, public_zero_input,
        [&](size_t index, Tensor<uint64_t> input) {
            const auto &meta = metadata[index];
            const auto &s = meta.shape;
            BeforeConvolution(context, meta, input, profile);
            const size_t size = (s.size + 2 * s.padding - s.kernel) / s.stride + 1;
            Tensor<uint64_t> public_zero_result({s.output_channels, size, size});
            AfterConvolution(context, meta, public_zero_result, index + 1 == metadata.size(),
                             profile);
            if (index + 1 == metadata.size())
                context.ExtendSigned(public_zero_result, context.config.accumulator_bits,
                                     context.config.ClassifierBits());
            return public_zero_result;
        },
        profile, steps, metadata);
    Classify(context, std::move(logits));
    context.EndPlanning();
}
} // namespace QuantExperiment
