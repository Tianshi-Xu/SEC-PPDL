#include <LinearLayer/Conv.h>
#include <cassert>

using namespace seal;
using namespace HE;
using namespace HE::unified;

namespace LinearLayer {
// Extract shared parameters. Let dim(w) = {Co, Ci, k, k}
Conv2D::Conv2D(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HEEvaluator* HE)
    : in_feature_size(in_feature_size), 
      stride(stride),
      padding(padding),
      weight(weight), 
      bias(bias), 
      HE(HE)
{
    std::vector<size_t> weight_shape = weight.shape();

    assert(weight_shape[0] == bias.shape()[0] && "Output channel does not match.");
    assert(weight_shape[2] == weight_shape[3] && "Input feature map is not a square.");
    assert(in_feature_size - weight_shape[2] + 2 * padding >= 0 && "Input feature map is too small.");

    out_channels = weight_shape[0];
    in_channels = weight_shape[1];
    kernel_size = weight_shape[2];
    out_feature_size = (in_feature_size + 2 * padding - kernel_size) / stride + 1;
};

} // namespace LinearLayer
