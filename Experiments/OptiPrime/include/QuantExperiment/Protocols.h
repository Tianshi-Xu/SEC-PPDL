#pragma once
#include <QuantExperiment/Timing.h>

namespace QuantExperiment {
struct RequantizationProfile {
    Component sign, mux, shift;
};
Datatype::Tensor<uint64_t> LowerRelu(Context &context, Datatype::Tensor<uint64_t> &input,
                                     RequantizationProfile *profile = nullptr);
Datatype::Tensor<uint64_t> Clamp(Context &context, const Datatype::Tensor<uint64_t> &input,
                                 int64_t low, int64_t high, int bits, int payload_bits = 0);
Datatype::Tensor<uint64_t> MaxPoolQuantized(Context &context,
                                            const Datatype::Tensor<uint64_t> &input);
Datatype::Tensor<uint64_t> Requantize(Context &context, Datatype::Tensor<uint64_t> input, int shift,
                                      int zero_point, int clamp_bits = 0,
                                      RequantizationProfile *profile = nullptr);
Datatype::Tensor<uint64_t> GlobalAverage(Context &context, Datatype::Tensor<uint64_t> input,
                                         int sum_bits = 0);
void Reduce(Datatype::Tensor<uint64_t> &input, int bits);
Datatype::Tensor<uint64_t> Classify(Context &context, Datatype::Tensor<uint64_t> logits);
} // namespace QuantExperiment
