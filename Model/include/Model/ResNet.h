#include <LinearLayer/Conv.h>
#include <NonlinearLayer/ReLU.h>
#include "Primitive.h"
using namespace LinearLayer;
using namespace NonlinearLayer;
using namespace NonlinearOperator;

namespace Model{
template <typename T, typename IO=Utils::NetIO>
class Bottleneck{
    public:
        int expansion = 4;
        int in_planes;
        int planes;
        int stride = 1;
        bool has_shortcut = false;
        ReLU<T, IO> *relu;
        Truncation<T> *truncation;
        Conv2D* conv1;
        Conv2D* conv2;
        Conv2D* conv3;
        Conv2D* shortcut;

        Bottleneck(uint64_t in_feature_size, uint64_t in_planes, uint64_t planes, uint64_t stride, CryptoPrimitive<T, IO> *cryptoPrimitive){
            this->in_planes = in_planes;
            this->planes = planes;
            this->stride = stride;
            this->relu = cryptoPrimitive->relu;
            this->truncation = cryptoPrimitive->truncation;
            CreateConv(conv1, in_feature_size, in_planes, planes, 1, 1, 0, cryptoPrimitive);
            CreateConv(conv2, in_feature_size, planes, planes, 3, stride, 1, cryptoPrimitive);
            CreateConv(conv3, in_feature_size/stride, planes, planes * this->expansion, 1, 1, 0, cryptoPrimitive);
            if (stride != 1 || in_planes != planes * this->expansion){
                has_shortcut = true;
                CreateConv(shortcut, in_feature_size, in_planes, planes * this->expansion, 1, stride, 0, cryptoPrimitive);
            }
        }

        void CreateConv(Conv2D* conv, uint64_t in_feature_size, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, uint64_t padding, CryptoPrimitive<T, IO> *cryptoPrimitive){
            switch (cryptoPrimitive->conv_type)
            {
            case Datatype::CONV_TYPE::Nest:
                Tensor<uint64_t> weight({out_channels, in_channels, kernel_size, kernel_size});
                Tensor<uint64_t> bias({out_channels});
                conv = new Conv2DNest(in_feature_size, stride, padding, weight, bias, cryptoPrimitive->HE);
                break;
            }
        }
        // TODO: can be simplified
        Tensor<T> operator()(Tensor<T> x){
            Tensor<T> x1 = (*conv1)(x);
            (*relu)(&x1);
            uint8_t *msb_x = new uint8_t[x1.size()];
            memset(msb_x, 0, x1.size());
            (*truncation)(&x1,17,43,true,msb_x);
            x1 = (*conv2)(x1);
            (*relu)(&x1);
            uint8_t *msb_x1 = new uint8_t[x1.size()];
            memset(msb_x1, 0, x1.size());
            (*truncation)(&x1,17,43,true,msb_x1);
            x1 = (*conv3)(x1);
            (*truncation)(&x1,17,43,true,msb_x1);
            if (has_shortcut){
                x = (*shortcut)(x);
            }
            return x + x1;
        }
};

class ResNet {
    
};
}