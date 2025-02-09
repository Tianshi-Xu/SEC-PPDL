#include "datatype/Tensor.h"
#include "layer/Module.h"

class BatchNorm2D : public Module{
    public:
        int num_features;
        Tensor<int,1> weight;
        Tensor<int,1> bias;
        Tensor<int,1> running_mean;
        Tensor<int,1> running_var;
        BatchNorm2D(int num_features){
            this->num_features = num_features;
        }
        Tensor<int,3> operator()(Tensor<int,3> x);
        Tensor<int,4> operator()(Tensor<int,4> x);
};