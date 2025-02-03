#include <seal/seal.h>
// #include <seal/secretkey.h>
// #include <seal/util/polyarithsmallmod.h>
// #include <seal/util/rlwe.h>
// #include <seal/secretkey.h>
// #include <seal/serializable.h>
#include "datatype/Tensor.h"
#include "layer/Module.h"
<<<<<<< HEAD
#include "HE/HE.h"
using namespace seal;
class Conv2D : public Module{
    public:
        uint64_t in_channels;
        uint64_t out_channels;
        uint64_t tiled_in_channels;
        uint64_t tiled_out_channels;
        uint64_t tile_size;
        uint64_t in_feature_size;
        uint64_t padded_feature_size;
        uint64_t out_feature_size;
        uint64_t input_rot;
        uint64_t kernel_size;
        uint64_t stride;
        uint64_t padding;
        Tensor<uint64_t> weight;
        Tensor<Plaintext> weight_pt;  // We denote all plaintext(ciphertext) variables with suffix '_pt'('_ct')
        Tensor<uint64_t> bias;
        HEEvaluator* HE;

        Conv2D(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HEEvaluator* HE);
        virtual Tensor<uint64_t> operator()(Tensor<uint64_t> x) = 0; // manual merging required

    private:
        virtual Tensor<Plaintext> PackWeight() = 0;
        virtual Tensor<uint64_t> PackActivation(Tensor<uint64_t> x) = 0;
        virtual Tensor<Ciphertext> HECompute(Tensor<Plaintext> weight_pt, Tensor<Ciphertext> ac_ct) = 0;
        virtual Tensor<uint64_t> DepackResult(Tensor<uint64_t> out) = 0;    
};


class Conv2DNest : public Conv2D {
    public:
        Conv2DNest(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HEEvaluator* HE);
        Tensor<uint64_t> operator()(Tensor<uint64_t> x);

    private:
        Tensor<Plaintext> PackWeight();
        Tensor<uint64_t> PackActivation(Tensor<uint64_t> x);
        Tensor<Ciphertext> HECompute(Tensor<Plaintext> weight_pt, Tensor<Ciphertext> ac_ct);
        Tensor<uint64_t> DepackResult(Tensor<uint64_t> out);
};
=======
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
>>>>>>> origin/main
