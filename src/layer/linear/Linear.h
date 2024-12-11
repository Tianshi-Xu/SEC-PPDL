#include <seal/seal.h>
#include "datatype/Tensor.h"
#include "layer/Module.h"

class Linear : public Module{
    public:
        int in_features;
        int out_features;
        Tensor<int,2> weight;
        Tensor<int,1> bias;
    Linear(int in_features, int out_features){
        this->in_features = in_features;
        this->out_features = out_features;
    }
    Tensor<int,3> operator()(Tensor<int,3> x);
};