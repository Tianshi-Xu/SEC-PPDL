#include <LinearLayer/Conv.h>
#include <NonlinearLayer/ReLU.h>
#include "Primitive.h"
using namespace LinearLayer;
using namespace NonlinearLayer;

namespace Model{
template <typename T, typename IO=Utils::NetIO>
class Bottleneck{
    public:
        int expansion = 4;
        int in_planes;
        int planes;
        int stride = 1;
        Conv2D *conv1;
        ReLU<T, IO> *relu;
        Conv2D *conv2;
        Conv2D *conv3;

        Bottleneck(int in_planes, int planes, int stride, CryptoPrimitive<T, IO> *cryptoPrimitive){
            this->in_planes = in_planes;
            this->planes = planes;
            this->stride = stride;
            switch (cryptoPrimitive->conv_type)
            {
            case Datatype::CONV_TYPE::Cheetah:
                this->conv1 = 
                break;
            case Datatype::CONV_TYPE::Nest:
                break;
            }
            
        }
};

class ResNet {
    
};
}