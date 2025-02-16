#include <seal/seal.h>
#include <hexl/hexl.hpp>
#include <Datatype/Tensor.h>
#include <HE/HE.h>
#include <Operator/Conversion.h>
#include "../../../Layer/Module.h"

using namespace seal;
using namespace HE::unified;
using namespace HE;
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
    Tensor<UnifiedPlaintext> weight_pt;  // We denote all plaintext(ciphertext) variables with suffix '_pt'('_ct')
    Tensor<uint64_t> bias;
    HE::HEEvaluator* HE;

    Conv2D(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HEEvaluator* HE);
    virtual Tensor<uint64_t> operator()(Tensor<uint64_t> x) = 0;

    private:
        virtual Tensor<UnifiedPlaintext> PackWeight() = 0;
        virtual Tensor<uint64_t> PackActivation(Tensor<uint64_t> x) = 0;
        virtual Tensor<UnifiedCiphertext> HECompute(Tensor<UnifiedPlaintext> weight_pt, Tensor<UnifiedCiphertext> ac_ct) = 0;
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

        Conv2DNest(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HEEvaluator* HE);
        Tensor<uint64_t> operator()(Tensor<uint64_t> x);

    private:
        Tensor<UnifiedPlaintext> PackWeight();
        Tensor<uint64_t> PackActivation(Tensor<uint64_t> x);
        Tensor<UnifiedCiphertext> HECompute(Tensor<UnifiedPlaintext> weight_pt, Tensor<UnifiedCiphertext> ac_ct);
        Tensor<uint64_t> DepackResult(Tensor<uint64_t> out);
};

}