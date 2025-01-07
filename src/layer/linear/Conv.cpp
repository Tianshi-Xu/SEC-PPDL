#include "Conv.h"
#include "Conversion.h"
#include <cassert>

using namespace seal;

// 父类构造函数，从张量中提取共用的维度参数。dim(w) = {Ci, Co, H, W}
Conv2D::Conv2D(int in_feature_size, int stride, int padding, const Tensor<int>& weight, const Tensor<int>& bias)
    : in_feature_size(in_feature_size), 
      weight(weight), 
      bias(bias)
{
    std::vector<size_t> weight_shape = weight.shape();

    assert(weight_shape[1] == bias.shape()[0] && "Output channel does not match.");
    assert(weight_shape[2] == weight_shape[3] && "Input feature map is not a square.");
    assert(in_feature_size - weight_shape[2] + 2 * padding >= 0 && "Input feature map is too small.");

    in_channels = weight_shape[0];
    out_channels = weight_shape[1];
    kernel_size = weight_shape[2];
    out_feature_size = (in_feature_size + 2 * padding - kernel_size) / stride + 1;
};


Conv2DNest::Conv2DNest(int in_feature_size, int stride, int padding, const Tensor<int>& weight, const Tensor<int>& bias)
    : Conv2D(in_feature_size, stride, padding, weight, bias)
{
}


Tensor<int> Conv2DNest::operator()(Tensor<int> x) {
    auto PackActivation = [this, &x] {
        assert(false && "Function not implemented yet");
    };

    auto DepackResult = [this, &x] {
        assert(false && "Function not implemented yet");
    };

    PackActivation();
    SSToHE(x);
    assert(false && "Function not implemented yet");
    DepackResult();
    HEToSS(x);
};