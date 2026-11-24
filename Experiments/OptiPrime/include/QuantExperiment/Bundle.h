#pragma once
#include <QuantExperiment/Model.h>
#include <filesystem>

namespace QuantExperiment {
// Layout and scales are public. Only the model owner needs weights.bin.
class ImageNetBundle {
  public:
    explicit ImageNetBundle(const std::filesystem::path &directory);
    Config InferenceConfig() const;
    Weights LoadWeights() const;
    std::vector<LayerMetadata> metadata;
    std::vector<Step> steps;

  private:
    std::filesystem::path root_;
    int work_bits_, activation_bits_ = 8;
};
} // namespace QuantExperiment
