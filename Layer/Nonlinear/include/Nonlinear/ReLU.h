#include <Datatype/Tensor.h>
#include <Layer/Module.h>

using namespace Datatype;
using namespace Layer;

class ReLU : public Module{
    public:
    Tensor<uint64_t> operator()(Tensor<uint64_t> x);
};