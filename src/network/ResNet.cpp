#include "ResNet.h"

Tensor<int,4> BasicBlock::operator()(Tensor<int,4> x){
    Tensor<int,4> identity = x;
    Tensor<int,4> out = conv1(x);
    out = bn1(out);
    out = relu(out);
    out = conv2(out);

    out = bn2(out);
    out = out + identity;
    out = relu(out);
    return out;
}

int main(){
    BasicBlock block(3, 3, 1, false, 1, 64, 1, BatchNorm2D(3));
    Tensor<int,4> x({1, 3, 32, 32});
    auto y = block(x);
    return 0;
}

// HE参数设置，he_2_ss接口公开，trunc/reduce/extend接口可能要暴露
// Tensor实现修改，底层都是一维，是定点数的抽象，有模数和scale
// 支持算子融合的简易开发
// load model的接口