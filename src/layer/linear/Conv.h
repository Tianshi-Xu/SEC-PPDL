#include <seal/seal.h>
#include "datatype/Tensor.h"
// #include "layer/Module.h"


class Conv2D {
    public:
        int in_channels;
        int out_channels;
        int tiled_in_channels;
        int tiled_out_channels;
        int tile_size;
        int in_feature_size;
        int padded_feature_size;
        int out_feature_size;
        int kernel_size;
        int stride;
        int padding;
        Tensor<int> weight;
        Tensor<int> bias;
        Conv2D(int in_feature_size, int stride, int padding, const Tensor<int>& weight, const Tensor<int>& bias);
        virtual Tensor<int> operator()(Tensor<int> x) = 0;
};


class Conv2DNest : public Conv2D {
    public:
        Conv2DNest(int in_feature_size, int stride, int padding, const Tensor<int>& weight, const Tensor<int>& bias);
        Tensor<int> operator()(Tensor<int> x);
};
