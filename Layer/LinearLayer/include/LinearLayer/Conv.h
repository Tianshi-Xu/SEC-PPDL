#include <seal/seal.h>
#include <Datatype/Tensor.h>
#include <HE/HE.h>
#include <Operator/Conversion.h>
#include "../../../Layer/Module.h"

using namespace seal;
using namespace HE;
using namespace Datatype;
//using namespace Operator;

namespace LinearLayer {

class Conv2D : public Module {
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

class Conv2DCheetah : public Module {
public:
    unsigned long M, C, H, W, h, s;
    unsigned long N, HW, WW, CW, MW, dM, dC, dH, dW, OW, Hprime, Wprime, HWprime, WWprime;
    size_t polyModulusDegree = 8192;
    uint64_t plain;
    HEEvaluator* he;
    Tensor<Plaintext> packedKernel;
    Tensor<int> kernel;

    Conv2DCheetah(size_t H, size_t W, HEEvaluator* he, Tensor<int64_t> kernel);

    Tensor<seal::Ciphertext> EncryptTensor(Tensor<seal::Plaintext> plainTensor);
    Tensor<seal::Plaintext> PackTensor(Tensor<int64_t> x);
    Tensor<seal::Plaintext> PackKernel(Tensor<int64_t> x);
    Tensor<seal::Ciphertext> Conv(Tensor<seal::Ciphertext> T, Tensor<seal::Plaintext> K);
    Tensor<int64_t> ExtractResult(Tensor<Plaintext> ConvResultPlain);

private:
    int DivUpper(int a, int b);
    int CalculateCost(int H, int W, int h, int Hw, int Ww, int C, int N);
    void FindOptimalPartition(int H, int W, int h, int C, int N, int* optimal_Hw, int* optimal_Ww);
};

}