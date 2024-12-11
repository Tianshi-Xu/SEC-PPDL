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
        Tensor<int,4> weight;
        Tensor<int,1> bias;
        Conv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding);
        Tensor<int,4> operator()(Tensor<int,4> x);
};