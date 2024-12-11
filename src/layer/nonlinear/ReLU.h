#include <datatype/Tensor.h>
#include "layer/Module.h"

class ReLU : public Module{
  public:
    Tensor<int,1> operator()(Tensor<int,1> x);
    Tensor<int,2> operator()(Tensor<int,2> x);
    Tensor<int,3> operator()(Tensor<int,3> x);
    Tensor<int,4> operator()(Tensor<int,4> x);
};