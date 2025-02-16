#include <seal/seal.h>
#include <hexl/hexl.hpp>
#include <Datatype/Tensor.h>
#include <HE/HE.h>
#include <Operator/Conversion.h>
#include "../../../Layer/Module.h"

using namespace seal;
using namespace Datatype;

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
    HE::HEEvaluator* HE;

    Conv2D(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HE::HEEvaluator* HE);
    
    virtual ~Conv2D() = default;
    
    virtual Tensor<uint64_t> operator()(Tensor<uint64_t> x) = 0;

    private:
        virtual Tensor<HE::unified::UnifiedPlaintext> PackWeight() = 0;
        virtual Tensor<uint64_t> PackActivation(Tensor<uint64_t> x) = 0;
        virtual Tensor<HE::unified::UnifiedCiphertext> HECompute(const Tensor<HE::unified::UnifiedPlaintext> &weight_pt, Tensor<HE::unified::UnifiedCiphertext> ac_ct) = 0;
        virtual Tensor<uint64_t> DepackResult(Tensor<uint64_t> out) = 0;    
};


class Conv2DNest : public Conv2D {
    public:
        uint64_t tiled_in_channels;
        uint64_t tiled_out_channels;
        uint64_t tile_size;
        uint64_t padded_feature_size;
        uint64_t input_rot;
        vector<uint64_t> tmp_w;

        Conv2DNest(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HE::HEEvaluator* HE);
        Tensor<uint64_t> operator()(Tensor<uint64_t> x);

    private:
        Tensor<HE::unified::UnifiedPlaintext> PackWeight();
        Tensor<uint64_t> PackActivation(Tensor<uint64_t> x);
        Tensor<HE::unified::UnifiedCiphertext> HECompute(const Tensor<HE::unified::UnifiedPlaintext> &weight_pt, Tensor<HE::unified::UnifiedCiphertext> ac_ct);
        Tensor<uint64_t> DepackResult(Tensor<uint64_t> out);
};

}