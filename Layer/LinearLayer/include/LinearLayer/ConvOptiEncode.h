#pragma once
#include <LinearLayer/Conv.h>
#include <LinearOperator/OptiEncode.h>
#include <memory>
#include <unordered_map>

namespace LinearLayer {
struct OptiEncodeOptions {
    size_t threads = 1;
    size_t channels_per_block = 0;
    size_t tile_output_size = 0;
    size_t baby_steps = 0;
    bool add_bias = true;
    size_t output_mod_switches = 1;
    // Predivide the reconstructed input ciphertext in R_q before any Galois
    // operation (BumbleBee ZeroGap normalization). Weights remain unscaled.
    bool ciphertext_inverse_normalization = false;
};
struct OptiEncodeStats {
    size_t input_ciphertexts = 0, output_ciphertexts = 0;
    size_t weight_plaintexts = 0, nonzero_weight_plaintexts = 0;
    double offline_seconds = 0, compute_seconds = 0;
};

// BFV coefficient encoding on the host. Input/output are additive field shares.
// Bias is a server-owned field residue and is included once by default.
// Instances may be reused sequentially; simultaneous calls on one instance/IO
// channel are not supported. Worker threads never read/write the network.
class Conv2DOptiEncode : public Conv2D {
public:
    Conv2DOptiEncode(uint64_t feature_size, uint64_t stride, uint64_t padding,
                    const Tensor<uint64_t> &weight, const Tensor<uint64_t> &bias,
                    HE::HEEvaluator *he, OptiEncodeOptions options = {});
    Conv2DOptiEncode(uint64_t feature_size, uint64_t in_channels, uint64_t out_channels,
                    uint64_t kernel_size, uint64_t stride, HE::HEEvaluator *he,
                    OptiEncodeOptions options = {});
    Tensor<uint64_t> operator()(Tensor<uint64_t> &input) override;
    const Operator::OptiEncodeLayout &Layout() const { return layout_; }
    const OptiEncodeStats &Stats() const { return stats_; }
    // The small union covers all automatically selected layouts for this N.
    static std::vector<uint32_t> GaloisElements(size_t degree);
private:
    OptiEncodeOptions options_;
    Operator::OptiEncodeLayout layout_;
    OptiEncodeStats stats_;
    std::vector<uint8_t> nonzero_weights_;
    std::unordered_map<uint32_t, int> rotation_steps_;
    Tensor<UnifiedPlaintext> PackWeight() override;
    Tensor<uint64_t> PackActivation(Tensor<uint64_t> &input) override;
    Tensor<UnifiedCiphertext> HECompute(const Tensor<UnifiedPlaintext> &weights,
                                       Tensor<UnifiedCiphertext> &input) override;
#ifdef USE_HE_GPU
    Tensor<UnifiedCiphertext> HEComputeGPU(const Tensor<UnifiedPlaintext> &weights,
                                         Tensor<UnifiedCiphertext> &input);
    void ApplyAutomorphismGPU(UnifiedCiphertext &ciphertext, uint32_t element) const;
#endif
    Tensor<uint64_t> DepackResult(Tensor<uint64_t> &output) override;
    void ApplyAutomorphism(seal::Evaluator &evaluator, seal::Ciphertext &ciphertext,
                           uint32_t element) const;
};
} // namespace LinearLayer
