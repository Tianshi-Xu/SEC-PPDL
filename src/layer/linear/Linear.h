#include <seal/seal.h>
#include "datatype/Tensor.h"
#include "layer/Module.h"
#include "HE/HE.h"

class Linear : public Module{
    public:
        int in_features;
        int out_features;
        Tensor<int> weight;
        Tensor<int> bias;
    Linear(int in_features, int out_features){
        this->in_features = in_features;
        this->out_features = out_features;
    }
    Tensor<int> operator()(Tensor<int> x);
};