#pragma once
#include <LinearLayer/Conv.h>
#include <Model/Primitive.h>
#include <NonlinearLayer/Pool.h>
#include <memory>
#include <vector>

namespace Model {

class FaceResNet64 {
    public:
        static constexpr int image_size = 64;
        static constexpr int embedding_size = 128;
        static constexpr int scale = 12;
        static constexpr int bitwidth = 37;
        static constexpr int default_he_threads = 16;

        enum class ConvolutionMode { Hybrid, CheetahBaseline };
        enum class Backbone { Compact, ResNet18, ResNet32, ResNet34, ResNet20 };

        struct ConvShape {
            size_t size;
            size_t in_channels;
            size_t out_channels;
            size_t kernel;
            size_t stride;
        };

        struct CommunicationStats {
            // Local bytes sent, accumulated over all measured calls.
            uint64_t convolution_bytes = 0;
            uint64_t linear_layer_bytes = 0;
            uint64_t linear_preprocessing_bytes = 0;
            double forward_seconds = 0;
        };

        struct ConvWeights {
            Datatype::Tensor<uint64_t> weight;
            Datatype::Tensor<uint64_t> bias;
        };
        struct FloatConvWeights {
            Datatype::Tensor<float> weight;
            Datatype::Tensor<float> bias;
        };

        // Order: stem, each residual block (main path, then shortcut when
        // present), and the embedding projection.
        using Weights = std::vector<ConvWeights>;
        using FloatWeights = std::vector<FloatConvWeights>;
        static std::vector<ConvShape> ConvolutionShapes(Backbone backbone);

        // Encode a copy of the FP32 weights for the integer protocol.
        static Weights EncodeWeights(const FloatWeights &weights);

        // Only the server supplies weights. Weights use 12 fractional bits
        // and biases use 24, both stored as signed values in uint64_t.
        // Q24 convolution accumulators must fit in the signed 37-bit range
        // and have magnitude below half the BFV modulus (a quarter with M&M).
        // Both parties must use the same setting. The fast path fuses
        // field-to-ring conversion and truncation with up to one Q12 unit of error per convolution.
        FaceResNet64(CryptoPrimitive<uint64_t> &crypto, const Weights *weights = nullptr,
                     bool use_mm_conversion = false, ConvolutionMode mode = ConvolutionMode::Hybrid,
                     Backbone backbone = Backbone::Compact, size_t he_threads = default_he_threads);

        // Input is a share of an aligned RGB CHW crop in [-1, 1] at Q12.
        Datatype::Tensor<uint64_t> Forward(const Datatype::Tensor<uint64_t> &image,
                                            CommunicationStats *communication = nullptr);

        // Returns an XOR share of the match bit for Q12 embeddings in [-1, 1].
        static uint8_t MatchEmbeddings(const Datatype::Tensor<uint64_t> &x,
                                       const Datatype::Tensor<uint64_t> &y,
                                       double threshold,
                                       NonlinearOperator::FixPoint<uint64_t> &fixpoint);

        // Both parties call Verify. Each receives only the 1:1 match bit.
        bool Verify(const Datatype::Tensor<uint64_t> &first,
                    const Datatype::Tensor<uint64_t> &second, double threshold,
                    CommunicationStats *communication = nullptr);

    private:
        struct ResidualSpec {
            size_t index;
            bool shortcut;
        };
        struct Layout {
            std::vector<ConvShape> convolutions;
            std::vector<ResidualSpec> blocks;
            size_t pool_size;
            bool stem_max_pool;
        };

        CryptoPrimitive<uint64_t> &crypto;
        bool use_mm_conversion;
        Layout layout;
        NonlinearLayer::AvgPool2D<uint64_t> pool;
        std::vector<std::unique_ptr<LinearLayer::Conv2D>> convolutions;

        static Layout MakeLayout(Backbone backbone);
        Datatype::Tensor<uint64_t> Convolve(size_t index, Datatype::Tensor<uint64_t> input,
                                             CommunicationStats *communication);
        Datatype::Tensor<uint64_t> ResidualBlock(size_t index, const Datatype::Tensor<uint64_t> &input,
                                                  bool shortcut, CommunicationStats *communication);
        void Activate(Datatype::Tensor<uint64_t> &input);
        Datatype::Tensor<uint64_t> StemMaxPool(const Datatype::Tensor<uint64_t> &input);
        Datatype::Tensor<uint64_t> Clamp(const Datatype::Tensor<uint64_t> &input, int64_t low, int64_t high);
};

} // namespace Model
