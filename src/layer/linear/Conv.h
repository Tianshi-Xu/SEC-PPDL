#include <seal/seal.h>
// #include <seal/secretkey.h>
// #include <seal/util/polyarithsmallmod.h>
// #include <seal/util/rlwe.h>
// #include <seal/secretkey.h>
// #include <seal/serializable.h>
#include "datatype/Tensor.h"
#include "layer/Module.h"
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
