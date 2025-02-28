#include <seal/seal.h>
#include <hexl/hexl.hpp>
#include <Datatype/Tensor.h>
#include <HE/HE.h>
#include <LinearOperator/Conversion.h>
#include "../../../Layer/Module.h"

using namespace seal;
using namespace Datatype;
using namespace HE;
using namespace HE::unified;

namespace LinearLayer {

class Conv2D : public Module {
    public:
        uint64_t in_channels;
        uint64_t out_channels;
        uint64_t in_feature_size;
        uint64_t out_feature_size; 
        uint64_t kernel_size;
        uint64_t stride;
        uint64_t padding;
        Tensor<uint64_t> weight;
        Tensor<HE::unified::UnifiedPlaintext> weight_pt;  // We denote all plaintext(ciphertext) variables with suffix '_pt'('_ct')
        Tensor<uint64_t> bias;
        HE::HEEvaluator* he;
        bool fused_bn;
        // TODO: remove the parameter `in_feature_size`
        Conv2D(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HE::HEEvaluator* he);

        // TODO: implement this constructor
        Conv2D(uint64_t in_feature_size, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, HE::HEEvaluator* he);
    
        virtual ~Conv2D() = default;
    
        virtual Tensor<uint64_t> operator()(Tensor<uint64_t> &x) = 0;

        void fuse_bn(Tensor<uint64_t> *gamma, Tensor<uint64_t> *beta);
    private:
        virtual Tensor<HE::unified::UnifiedPlaintext> PackWeight() = 0;
        virtual Tensor<uint64_t> PackActivation(Tensor<uint64_t> &x) = 0;
        virtual Tensor<HE::unified::UnifiedCiphertext> HECompute(const Tensor<HE::unified::UnifiedPlaintext> &weight_pt, Tensor<HE::unified::UnifiedCiphertext> &ac_ct) = 0;
        virtual Tensor<uint64_t> DepackResult(Tensor<uint64_t> &out) = 0;
        // virtual void compute_he_params(Tensor<uint64_t> &x) = 0;
};


class Conv2DNest : public Conv2D {
    public:
        uint64_t tiled_in_channels;
        uint64_t tiled_out_channels;
        uint64_t tile_size;
        uint64_t padded_feature_size = 0;
        uint64_t input_rot;
        vector<uint64_t> tmp_w;

        Conv2DNest(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HE::HEEvaluator* he);
        Conv2DNest(uint64_t in_feature_size, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, HE::HEEvaluator* he);
        Tensor<uint64_t> operator()(Tensor<uint64_t> &x) override;

    private:
        Tensor<HE::unified::UnifiedPlaintext> PackWeight() override;
        Tensor<uint64_t> PackActivation(Tensor<uint64_t> &x) override;
        Tensor<HE::unified::UnifiedCiphertext> HECompute(const Tensor<HE::unified::UnifiedPlaintext> &weight_pt, Tensor<HE::unified::UnifiedCiphertext> &ac_ct) override;
        Tensor<uint64_t> DepackResult(Tensor<uint64_t> &out) override;
        void compute_he_params(uint64_t in_feature_size);
};


class Conv2DCheetah : public Conv2D {
public:
    unsigned long M, C, H, W, h, s;
    unsigned long N, HW, WW, CW, MW, dM, dC, dH, dW, OW, Hprime, Wprime, HWprime, WWprime;
    size_t polyModulusDegree = 8192;
    uint64_t plain;
    HEEvaluator* he;

    Conv2DCheetah(size_t H, size_t W, HEEvaluator* he, const Tensor<uint64_t>& kernel, size_t stride, const Tensor<uint64_t>& bias, uint64_t padding);

    Conv2DCheetah(size_t H, size_t W, HEEvaluator* he, const Tensor<uint64_t>& kernel, 
                  size_t stride, const Tensor<uint64_t>& bias, uint64_t padding, Tensor<uint64_t> *gamma, Tensor<uint64_t> *beta);

    Tensor<uint64_t> operator()(Tensor<uint64_t> x);

private:
    int DivUpper(int a, int b);
    int CalculateCost(int H, int W, int h, int Hw, int Ww, int C, int N);
    void FindOptimalPartition(int H, int W, int h, int C, int N, int* optimal_Hw, int* optimal_Ww);
    Tensor<UnifiedCiphertext> EncryptTensor(Tensor<UnifiedPlaintext> plainTensor);
    Tensor<uint64_t> PackActivation(Tensor<uint64_t> x);
    Tensor<UnifiedPlaintext> PackWeight();
    Tensor<UnifiedCiphertext> TensorTOHE(Tensor<uint64_t> PackActivationTensor);
    Tensor<UnifiedCiphertext> HECompute(Tensor<UnifiedPlaintext> weight_pt, Tensor<UnifiedCiphertext> ac_ct);
    Tensor<UnifiedCiphertext> sumCP(Tensor<UnifiedCiphertext> cipherTensor, Tensor<UnifiedPlaintext> plainTensor);
    Tensor<uint64_t> DepackResult(Tensor<uint64_t> out);
    Tensor<uint64_t> HETOTensor (Tensor<UnifiedCiphertext> inputCipher);
    void fuse_bn(Tensor<uint64_t> *gamma, Tensor<uint64_t> *beta);
};

}
