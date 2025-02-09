#include <seal/seal.h>
#include <seal/secretkey.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/rlwe.h>
#include <seal/secretkey.h>
#include <seal/serializable.h>
#include "Tensor.h"
#include "Module.h"
#include "HE.h"

class Linear : public Module{
    public:
        int in_features;
        int out_features;
        Tensor<int> weight;
        Tensor<int> bias;
        HEEvaluator* he;

    Linear(int in_features, int out_features, Tensor<int> weightMatrix, HEEvaluator* HE, Tensor<int> biasVec)
        : in_features(in_features), 
          out_features(out_features), 
          weight(weightMatrix), 
          bias(biasVec), 
          he(HE) {}
    Tensor<int> operator()(Tensor<int> x);
};