#include <seal/seal.h>
#include <Datatype/Tensor.h>
#include <Layer/Module.h>
#include <HE/HE.h>

class Linear : public Module{
    public:
        int in_features;
        int out_features;
        Tensor<int> weight;
        Tensor<int> bias;
        HE::HEEvaluator* he;

    Linear(int in_features, int out_features, Tensor<int> weightMatrix, HE::HEEvaluator* HE, Tensor<int> biasVec)
        : in_features(in_features), 
          out_features(out_features), 
          weight(weightMatrix), 
          bias(biasVec), 
          he(HE) {}
    Tensor<int> operator()(Tensor<int> x);
};