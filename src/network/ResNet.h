#include "layer/__init__.h"
class BasicBlock{
    public:
        Conv2D conv1;
        Conv2D conv2;
        ReLU relu;
        int stride;
        BatchNorm2D bn1;
        BatchNorm2D bn2;


    BasicBlock(int inplanes,
        int planes,
        int stride,
        bool downsample,
        int groups,
        int base_width,
        int dilation,
        BatchNorm2D norm_layer);
    Tensor<int,4> operator()(Tensor<int,4> x);

}
class ResNet {
    
};