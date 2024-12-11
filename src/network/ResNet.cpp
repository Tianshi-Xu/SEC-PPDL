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