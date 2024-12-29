#include <seal/seal.h>
#include "datatype/Tensor.h"
#include "layer/Module.h"
using namespace seal;
class Conv2D : public Module{
    public:
        int in_channels;
        int out_channels;
        int kernel_size;
        int stride;
        int padding;
        Tensor<int> weight;
        Tensor<int> bias;
        HE* he;
        Conv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding, HE* he);
        Tensor<Plaintext> PackWeight();
        Tensor<int> PackAct(Tensor<int> x);
        Tensor<int> operator()(Tensor<int> x);
};


void main(){
    HE* he = new HE({60,40,60}, 8192);
    Conv2D conv(1, 1, 3, 1, 1, he);
    Tensor<int> x({1, 1, 5, 5});
    x.data() = {1, 2, 3, 4, 5,
                6, 7, 8, 9, 10,
                11, 12, 13, 14, 15,
                16, 17, 18, 19, 20,
                21, 22, 23, 24, 25};
    Tensor<int> y = conv(x);
    //TODO plaintext to tensor
}