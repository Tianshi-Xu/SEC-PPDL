#pragma once
#include <QuantExperiment/Protocols.h>
#include <LinearLayer/ConvOptiEncode.h>
#include <memory>

namespace QuantExperiment {
struct LayerShape {
    size_t size, input_channels, output_channels, kernel, stride, padding;
};
struct LayerMetadata {
    LayerShape shape;
    int shift = 0, input_zero_point = 0, output_zero_point = 0, clamp_bits = 16;
    int weight_bits = 4;
    // Public per-output-channel fixed-point gains, before bias and the dyadic shift.
    std::vector<uint32_t> multipliers;
};
struct Parameters {
    Datatype::Tensor<uint64_t> weight, bias;
};
enum class Operation { Conv, MaxPool3, MaxPool2, Residual, GlobalAverage, Flatten };
struct Step {
    Operation op;
    size_t input, skip = 0;
    bool projection = false;
};
using Weights = std::vector<Parameters>;
struct Profile {
    RequantizationProfile requantization_detail;
    // Optional observer for public-input accuracy audits.
    std::function<void(const Datatype::Tensor<uint64_t> &)> observe_activation;
    bool trace_layout_steps = false;
    Component ring_to_field, convolution, field_to_ring, requantization;
    Component residual, maxpool, globalpool, argmax;
    Component fully_connected, classifier_tail;
};
class QuantizedImageNetModel {
  public:
    QuantizedImageNetModel(Context &context, const Weights *weights,
                           std::vector<LayerMetadata> metadata,
                           std::vector<Step> steps);
    Datatype::Tensor<uint64_t> Forward(const Datatype::Tensor<uint64_t> &input, Profile &profile);
    // Public-layout-only planning. No image or model weights are accepted.
    static void Plan(Context &context, const std::vector<LayerMetadata> &metadata,
                     const std::vector<Step> &steps);

  private:
    Context &context_;
    std::vector<LayerMetadata> metadata_;
    std::vector<Step> steps_;
    std::vector<std::unique_ptr<LinearLayer::Conv2D>> layers_;
    Datatype::Tensor<uint64_t> Convolve(size_t index, Datatype::Tensor<uint64_t> input,
                                        Profile &profile);
};
} // namespace QuantExperiment
