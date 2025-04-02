
#include <seal/seal.h>
#include <Datatype/Tensor.h>
#include <HE/HE.h>
#include <LinearOperator/Conversion.h>
#include "../../../Layer/Module.h"

using namespace seal;
using namespace Datatype;
using namespace HE;
using namespace HE::unified;

namespace LinearLayer {

class Linear : public Module {
    public:
        uint64_t dim_0;  // The number of rows of the first matrix
        uint64_t dim_1;  // The number of columns of the first matrix, as well as the number of rows of the second matrix
        uint64_t dim_2;  // The number of columns of the second matrix
        Tensor<uint64_t> weight;
        Tensor<HE::unified::UnifiedPlaintext> weight_pt;  // We denote all plaintext(ciphertext) variables with suffix '_pt'('_ct')
        Tensor<uint64_t> bias;
        HE::HEEvaluator* HE;
        // TODO: remove the parameter `in_feature_size`
        Linear(uint64_t dim_0, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HE::HEEvaluator* HE);
        Linear(uint64_t dim_0, uint64_t dim_1, uint64_t dim_2, HE::HEEvaluator* HE);
    
        virtual ~Linear() = default;
    
        virtual Tensor<uint64_t> operator()(Tensor<uint64_t> &x) = 0;
    private:
        virtual Tensor<HE::unified::UnifiedPlaintext> PackWeight() = 0;
        virtual Tensor<uint64_t> PackActivation(Tensor<uint64_t> &x) = 0;
        virtual Tensor<HE::unified::UnifiedCiphertext> HECompute(const Tensor<HE::unified::UnifiedPlaintext> &weight_pt, Tensor<HE::unified::UnifiedCiphertext> &ac_ct) = 0;
        virtual Tensor<uint64_t> DepackResult(Tensor<uint64_t> &out) = 0;
};


class LinearBolt : public Linear {
    public:
        uint64_t padded_dim_0;
        uint64_t padded_dim_1;
        uint64_t padded_dim_2;
        uint64_t tiled_dim_1;
        uint64_t tiled_dim_2;
        uint64_t tile_size;
        uint64_t input_rot;
        Tensor<uint64_t> padded_weight;

        LinearBolt(uint64_t dim_0, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HE::HEEvaluator* HE);
        LinearBolt(uint64_t dim_0, uint64_t dim_1, uint64_t dim_2, HE::HEEvaluator* HE);
        Tensor<uint64_t> operator()(Tensor<uint64_t> &x) override;

    private:
        Tensor<HE::unified::UnifiedPlaintext> PackWeight() override;
        Tensor<uint64_t> PackActivation(Tensor<uint64_t> &x) override;
        Tensor<HE::unified::UnifiedCiphertext> HECompute(const Tensor<HE::unified::UnifiedPlaintext> &weight_pt, Tensor<HE::unified::UnifiedCiphertext> &ac_ct) override;
        Tensor<uint64_t> DepackResult(Tensor<uint64_t> &out) override;
        void compute_he_params(uint64_t in_feature_size);
};


class MatmulCtctBumble : public Module {
    public:
        HE::HEEvaluator* HE;

        MatmulCtctBumble(HE::HEEvaluator* HE);
        Tensor<uint64_t> operator()(Tensor<uint64_t> &x1, Tensor<uint64_t> &x2);

    private:
        Tensor<uint64_t> Transpose(Tensor<uint64_t> &x);
        Tensor<uint64_t> MatmulPtpt(Tensor<uint64_t> &x1, Tensor<uint64_t> &x2);
};

}
