#pragma once
#include <Datatype/Tensor.h>
#include <NonlinearOperator/FixPoint.h>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace NonlinearLayer {

template<typename T>
class AvgPool2D {
    public:
        uint64_t kernel_size;
        uint64_t stride;
        uint64_t padding;

        AvgPool2D(uint64_t kernel_size, uint64_t stride = -1, uint64_t padding = 0)
            : kernel_size(kernel_size), stride(stride == uint64_t(-1) ? kernel_size : stride), padding(padding) {
            if (kernel_size == 0 || this->stride == 0 ||
                kernel_size > std::numeric_limits<uint64_t>::max() / kernel_size) {
                throw std::invalid_argument("Invalid average pooling kernel or stride");
            }
        }

        // Plain CHW input. Zero padding counts toward the window size.
        Datatype::Tensor<T> operator()(const Datatype::Tensor<T> &x) const {
            using Accumulator = std::conditional_t<std::is_integral_v<T>, __int128_t, long double>;
            return Pool<Accumulator>(x, [this](Accumulator sum) {
                return static_cast<T>(sum / (kernel_size * kernel_size));
            });
        }

        // Signed fixed-point values encoded as additive uint64_t ring shares.
        // The reconstructed sum must fit in the signed accumulator bitwidth.
        Datatype::Tensor<T> ForwardShared(const Datatype::Tensor<T> &x,
                                         NonlinearOperator::FixPoint<T> &fixpoint, int bitwidth) const {
            static_assert(std::is_same_v<T, uint64_t>, "Average pooling expects uint64_t ring shares");
            const uint64_t area = kernel_size * kernel_size;
            if ((area & (area - 1)) != 0) {
                throw std::invalid_argument("Shared average pooling requires a power-of-two window area");
            }
            int shift = 0;
            for (uint64_t divisor = area; divisor > 1; divisor >>= 1) {
                ++shift;
            }
            if (bitwidth <= shift || bitwidth > 64) {
                throw std::invalid_argument("Invalid average pooling accumulator bitwidth");
            }
            auto y = Pool<uint64_t>(x, [](uint64_t sum) { return sum; });
            fixpoint.truncate(y, shift, bitwidth, false, true);
            y.bitwidth = bitwidth;
            return y;
        }

    private:
        template<typename Accumulator, typename Finish>
        Datatype::Tensor<T> Pool(const Datatype::Tensor<T> &x, Finish finish) const {
            const auto &shape = x.shape();
            if (shape.size() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0 ||
                padding > (std::numeric_limits<size_t>::max() - std::max(shape[1], shape[2])) / 2 ||
                shape[1] + 2 * padding < kernel_size || shape[2] + 2 * padding < kernel_size) {
                throw std::invalid_argument("Invalid CHW input for average pooling");
            }
            const size_t height = shape[1];
            const size_t width = shape[2];
            const size_t out_height = (height + 2 * padding - kernel_size) / stride + 1;
            const size_t out_width = (width + 2 * padding - kernel_size) / stride + 1;
            Datatype::Tensor<T> y({shape[0], out_height, out_width}, x.bitwidth, x.scale);
            for (size_t c = 0; c < shape[0]; ++c) {
                for (size_t h = 0; h < out_height; ++h) {
                    for (size_t w = 0; w < out_width; ++w) {
                        Accumulator sum = 0;
                        for (size_t ph = 0; ph < kernel_size; ++ph) {
                            const size_t row = h * stride + ph;
                            if (row < padding || row - padding >= height) {
                                continue;
                            }
                            for (size_t pw = 0; pw < kernel_size; ++pw) {
                                const size_t col = w * stride + pw;
                                if (col >= padding && col - padding < width) {
                                    sum += x({c, row - padding, col - padding});
                                }
                            }
                        }
                        y({c, h, w}) = finish(sum);
                    }
                }
            }
            return y;
        }
};

} // namespace NonlinearLayer
