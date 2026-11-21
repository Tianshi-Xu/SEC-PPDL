#pragma once
#include <LinearLayer/Conv.h>
#include <Model/Primitive.h>
#include <NonlinearLayer/Pool.h>
#include <array>
#include <memory>

namespace Model {

class FaceResNet64 {
    public:
        static constexpr int image_size = 64;
        static constexpr int embedding_size = 128;
        static constexpr int scale = 8;
        static constexpr int bitwidth = 60;

        enum class ConvolutionMode { Hybrid, CheetahBaseline };

        struct CommunicationStats {
            // Local bytes sent, accumulated over all measured calls.
            uint64_t convolution_bytes = 0;
            uint64_t linear_layer_bytes = 0;
        };

        struct ConvWeights {
            Datatype::Tensor<uint64_t> weight;
            Datatype::Tensor<uint64_t> bias;
        };

        // Order: stem, stage 1 (two convolutions), stages 2 and 3 (two
        // convolutions and a shortcut each), then the embedding projection.
        using Weights = std::array<ConvWeights, 10>;

        // Only the server supplies weights. Weights are signed Q8 and biases
        // signed Q16, both encoded as uint64_t and bounded in magnitude by 4.
        // Both parties must use the same setting. The fast path fuses
        // field-to-ring conversion and truncation with up to one Q8 unit of error per convolution.
        FaceResNet64(CryptoPrimitive<uint64_t> &crypto, const Weights *weights = nullptr,
                     bool use_mm_conversion = false, ConvolutionMode mode = ConvolutionMode::Hybrid);

        // Input is a share of an aligned RGB CHW crop in [-1, 1] at Q8.
        Datatype::Tensor<uint64_t> Forward(const Datatype::Tensor<uint64_t> &image,
                                            CommunicationStats *communication = nullptr);

        // Returns an XOR share of the match bit for Q8 embeddings in [-1, 1].
        static uint8_t MatchEmbeddings(const Datatype::Tensor<uint64_t> &x,
                                       const Datatype::Tensor<uint64_t> &y,
                                       double threshold,
                                       NonlinearOperator::FixPoint<uint64_t> &fixpoint);

        // Both parties call Verify. Each receives only the 1:1 match bit.
        bool Verify(const Datatype::Tensor<uint64_t> &first,
                    const Datatype::Tensor<uint64_t> &second, double threshold,
                    CommunicationStats *communication = nullptr);

    private:
        CryptoPrimitive<uint64_t> &crypto;
        bool use_mm_conversion;
        NonlinearLayer::AvgPool2D<uint64_t> pool{16};
        std::array<std::unique_ptr<LinearLayer::Conv2D>, 10> convolutions;

        Datatype::Tensor<uint64_t> Convolve(size_t index, Datatype::Tensor<uint64_t> input,
                                             CommunicationStats *communication);
        Datatype::Tensor<uint64_t> ResidualBlock(size_t index, const Datatype::Tensor<uint64_t> &input,
                                                  bool shortcut, CommunicationStats *communication);
        Datatype::Tensor<uint64_t> Clamp(const Datatype::Tensor<uint64_t> &input, int64_t low, int64_t high);
};

} // namespace Model
