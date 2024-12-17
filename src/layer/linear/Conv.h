#include <seal/seal.h>
#include "datatype/Tensor.h"
#include "layer/Module.h"

class Conv2D : public Module{
    public:
        int in_channels;
        int out_channels;
        int kernel_size;
        int stride;
        int padding;
        Tensor<int> weight;
        Tensor<int> bias;
        Conv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding);
        Tensor<int> operator()(Tensor<int> x);
};