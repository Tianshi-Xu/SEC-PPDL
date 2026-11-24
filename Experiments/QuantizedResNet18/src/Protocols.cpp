#include <QuantExperiment/Protocols.h>
#include <QuantExperiment/SharedSelection.h>

namespace QuantExperiment {
using Datatype::Tensor;
Tensor<uint64_t> Classify(Context &context, Tensor<uint64_t> logits) {
    const int width = context.config.ArgmaxBits();
    if (width > context.config.ClassifierBits())
        context.ExtendSigned(logits, context.config.ClassifierBits(), width);
    else
        Reduce(logits, width);
    return ArgMaxShared(logits, *context.fix, width);
}
void Reduce(Tensor<uint64_t> &input, int bits) {
    if (bits < 1 || bits > 64)
        throw std::invalid_argument("Invalid ring width");
    const auto mask = bits == 64 ? UINT64_MAX : (uint64_t(1) << bits) - 1;
    for (auto &v : input.data())
        v &= mask;
    input.bitwidth = bits;
}
Tensor<uint64_t> Clamp(Context &context, const Tensor<uint64_t> &input, int64_t low, int64_t high,
                       int bits, int payload_bits) {
    if (!payload_bits)
        payload_bits = bits;
    if (bits < 2 || bits > payload_bits || payload_bits > 60)
        throw std::invalid_argument("Clamp comparison width must fit its payload ring");
    if (low > high)
        throw std::invalid_argument("Invalid quantization clamp");
    Tensor<uint64_t> comparisons({2 * input.size()}), selected(comparisons.shape());
    for (size_t i = 0; i < input.size(); ++i) {
        comparisons(i) = input(i) - (context.config.party == 1 ? uint64_t(low) : 0);
        comparisons(i + input.size()) = input(i) - (context.config.party == 1 ? uint64_t(high) : 0);
    }
    Tensor<uint8_t> positive(comparisons.shape());
    context.fix->less_than_zero(comparisons, positive, bits);
    if (context.config.party == 1)
        for (auto &v : positive.data())
            v ^= 1;
    context.fix->mux(positive, comparisons, selected, payload_bits, payload_bits);
    Tensor<uint64_t> result(input.shape());
    for (size_t i = 0; i < input.size(); ++i)
        result(i) = selected(i) - selected(i + input.size()) +
                    (context.config.party == 1 ? uint64_t(low) : 0);
    result.bitwidth = payload_bits;
    return result;
}
Tensor<uint64_t> LowerRelu(Context &context, Tensor<uint64_t> &input,
                           RequantizationProfile *profile) {
    RequantizationProfile unused;
    auto &times = profile ? *profile : unused;
    const int bits = context.config.accumulator_bits;
    Tensor<uint8_t> keep(input.shape());
    {
        Timer timer(context, times.sign);
        context.fix->less_than_zero(input, keep, bits);
    }
    if (context.config.party == 1)
        for (auto &v : keep.data())
            v ^= 1;
    Tensor<uint64_t> output(input.shape());
    {
        Timer timer(context, times.mux);
        context.fix->mux(keep, input, output, bits, bits);
    }
    Reduce(output, bits);
    return output;
}
Tensor<uint64_t> Requantize(Context &context, Tensor<uint64_t> input, int shift, int zero_point,
                            int clamp_bits, RequantizationProfile *profile) {
    if (shift < 0 || shift >= context.config.accumulator_bits || zero_point < 0 ||
        zero_point > context.config.ActivationMax())
        throw std::invalid_argument("Invalid dyadic requantization parameters");
    if (context.config.approximate_relu) {
        if (context.config.party == 1)
            for (auto &v : input.data())
                v += uint64_t(zero_point) << shift;
        input = LowerRelu(context, input, profile);
        if (shift) {
            Component unused;
            Timer timer(context, profile ? profile->shift : unused);
            context.ApproximateShiftNonnegative(input, shift);
        }
        return input;
    }
    if (shift)
        context.fix->truncate(input, shift, context.config.accumulator_bits, false, true);
    if (context.config.party == 1)
        for (auto &v : input.data())
            v += zero_point;
    if (!clamp_bits)
        clamp_bits = context.config.accumulator_bits;
    if ((!context.config.power_of_two && clamp_bits < context.config.activation_ring_bits) ||
        clamp_bits > context.config.accumulator_bits)
        throw std::invalid_argument("Invalid certified clamp width");
    if (!context.config.power_of_two)
        Reduce(input, clamp_bits);
    auto result = Clamp(context, input, 0, context.config.ActivationMax(), clamp_bits,
                        context.config.power_of_two ? context.config.accumulator_bits : clamp_bits);
    Reduce(result, context.config.activation_ring_bits);
    return result;
}
Tensor<uint64_t> MaxPoolQuantized(Context &context, const Tensor<uint64_t> &input) {
    if (!context.config.power_of_two)
        return MaxPoolShared(input, 3, 2, 1, *context.fix,
                                             context.config.activation_ring_bits);
    const auto shape = input.shape();
    if (shape.size() != 3 || shape[1] != 112 || shape[2] != 112)
        throw std::invalid_argument("PO2 RN18 maxpool expects 112x112 activations");
    const size_t count = shape[0] * 56 * 56;
    size_t width = 9;
    Tensor<uint64_t> candidates({count, width});
    for (size_t c = 0; c < shape[0]; ++c)
        for (size_t r = 0; r < 56; ++r)
            for (size_t col = 0; col < 56; ++col)
                for (size_t k = 0; k < 9; ++k) {
                    const int row = int(2 * r + k / 3) - 1, column = int(2 * col + k % 3) - 1;
                    candidates(((c * 56 + r) * 56 + col) * 9 + k) =
                        row >= 0 && row < 112 && column >= 0 && column < 112
                            ? input({c, size_t(row), size_t(column)})
                            : 0; // Pooling follows ReLU, so zero is a valid padding sentinel.
                }
    while (width > 1) {
        const size_t pairs = width / 2, next_width = (width + 1) / 2;
        Tensor<uint64_t> left({count * pairs}), delta(left.shape()), selected(left.shape());
        for (size_t i = 0; i < count; ++i)
            for (size_t p = 0; p < pairs; ++p) {
                left(i * pairs + p) = candidates(i * width + 2 * p);
                delta(i * pairs + p) = left(i * pairs + p) - candidates(i * width + 2 * p + 1);
            }
        Tensor<uint8_t> choose_right(delta.shape());
        // A8/sentinel differences fit signed 10 bits; retain the HE work ring.
        context.fix->less_than_zero(delta, choose_right,
                                    context.config.approximate_relu
                                        ? context.config.accumulator_bits
                                        : context.config.activation_bits + 2);
        for (auto &v : delta.data())
            v = uint64_t(0) - v;
        context.fix->mux(choose_right, delta, selected, context.config.activation_ring_bits,
                         context.config.activation_ring_bits);
        Tensor<uint64_t> next({count, next_width});
        for (size_t i = 0; i < count; ++i) {
            for (size_t p = 0; p < pairs; ++p)
                next(i * next_width + p) = left(i * pairs + p) + selected(i * pairs + p);
            if (width & 1)
                next(i * next_width + pairs) = candidates(i * width + width - 1);
        }
        candidates = std::move(next);
        width = next_width;
    }
    Reduce(candidates, context.config.activation_ring_bits);
    candidates.reshape({shape[0], 56, 56});
    return candidates;
}
namespace {
Tensor<uint64_t> Divide49(Context &context, const Tensor<uint64_t> &sums, int precision,
                          int work_bits) {
    const auto channels = sums.size();
    auto quotient = sums;
    const uint64_t reciprocal = (uint64_t(1) << precision) / 49;
    for (auto &v : quotient.data())
        v *= reciprocal;
    context.fix->truncate(quotient, precision, work_bits, false, true);
    Tensor<uint64_t> remainder({channels}), one({channels}), increment({channels});
    for (size_t i = 0; i < channels; ++i) {
        remainder(i) = sums(i) - quotient(i) * 49 - (context.config.party == 1 ? 49 : 0);
        one(i) = context.config.party == 1 ? 1 : 0;
    }
    Tensor<uint8_t> correct({channels});
    context.fix->less_than_zero(remainder, correct, work_bits);
    if (context.config.party == 1)
        for (auto &v : correct.data())
            v ^= 1;
    context.fix->mux(correct, one, increment, work_bits, work_bits);
    quotient = quotient + increment;
    return quotient;
}
} // namespace
Tensor<uint64_t> GlobalAverage(Context &context, Tensor<uint64_t> input, int sum_bits) {
    if (input.shape().size() != 3 || input.shape()[1] != 7 || input.shape()[2] != 7)
        throw std::invalid_argument("Expected Cx7x7 activation");
    const int ring = context.config.accumulator_bits;
    const int width = context.config.power_of_two ? std::max(ring, sum_bits) : ring;
    const bool split = context.config.power_of_two && width > 32;
    const int precision =
        split ? 22
              : (context.config.power_of_two ? width
                                             : std::max(16, context.config.activation_bits + 6));
    const int work = std::max(width + int(split), 2 * precision - 4);
    context.ExtendUnsigned(input, context.config.activation_ring_bits, width);
    const size_t channels = input.shape()[0];
    Tensor<uint64_t> sums({channels});
    for (size_t c = 0; c < channels; ++c)
        for (size_t j = 0; j < 49; ++j)
            sums(c) += input(c * 49 + j);
    Reduce(sums, width);
    context.ExtendUnsigned(sums, width, work);
    Tensor<uint64_t> quotient;
    if (split) {
        // A sum of 49 signed-positive 32-bit values needs up to 37 bits.
        // Divide its high part first so each reciprocal product fits in 40 bits.
        auto high = sums;
        context.fix->truncate(high, 16, work, false, true);
        high = Divide49(context, high, precision, work);
        for (size_t i = 0; i < channels; ++i)
            sums(i) -= (high(i) * 49) << 16;
        quotient = Divide49(context, sums, precision, work);
        for (size_t i = 0; i < channels; ++i)
            quotient(i) += high(i) << 16;
    } else {
        quotient = Divide49(context, sums, precision, work);
    }
    Reduce(quotient, context.config.activation_ring_bits);
    quotient.reshape({channels, 1, 1});
    return quotient;
}
} // namespace QuantExperiment
