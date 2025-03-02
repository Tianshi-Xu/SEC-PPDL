#include <LinearLayer/Conv.h>
#include <NonlinearLayer/ReLU.h>
#include "Primitive.h"
using namespace LinearLayer;
using namespace NonlinearLayer;
using namespace NonlinearOperator;
using namespace std;
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
        // TODO: can be simplified to use pointer
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
            conv1 = CreateConv(in_feature_size, in_planes, planes, 1, 1, cryptoPrimitive);
            if(cryptoPrimitive->party == 1){
                cout << "server conv1 weight shape: " << endl;
                conv1->weight.print_shape();
            }
            conv2 = CreateConv(in_feature_size, planes, planes, 3, stride, cryptoPrimitive);
            conv3 = CreateConv(in_feature_size/stride, planes, planes * this->expansion, 1, 1, cryptoPrimitive);
            if (stride != 1 || in_planes != planes * this->expansion){
                has_shortcut = true;
                shortcut = CreateConv(in_feature_size, in_planes, planes * this->expansion, 1, stride, cryptoPrimitive);
            }
        }

        Conv2D* CreateConv(uint64_t in_feature_size, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, CryptoPrimitive<T, IO> *cryptoPrimitive){
            Conv2D* conv;
            switch (cryptoPrimitive->conv_type)
            {
            case Datatype::CONV_TYPE::Nest:
                conv = new Conv2DNest(in_feature_size, in_channels, out_channels, kernel_size, stride, cryptoPrimitive->HE);
                cout << "Conv2DNest created" << endl;
                break;
            }
            return conv;
        }
        // TODO: can be simplified
        Tensor<T> operator()(Tensor<T> &x){
            Tensor<T> x_res = x;
            x.print_shape();
            cout << "Bottleneck operator called" << endl;
            // conv1->weight.print_shape();
            x = conv1->operator()(x);
            x.print_shape();
            cout << "conv1 done" << endl;
            (*relu)(&x);
            x.print_shape();
            cout << "relu done" << endl;
            uint8_t *msb_x = new uint8_t[x.size()];
            memset(msb_x, 0, x.size());
            (*truncation)(x,17,43,true,msb_x);
            x.print_shape();
            cout << "truncation done" << endl;
            x = conv2->operator()(x);
            cout << "conv2 done" << endl;
            (*relu)(&x);
            cout << "relu done" << endl;
            uint8_t *msb_x1 = new uint8_t[x.size()];
            memset(msb_x1, 0, x.size());
            (*truncation)(x,17,43,true,msb_x1);
            cout << "truncation done" << endl;
            x = conv3->operator()(x);
            cout << "conv3 done" << endl;
            (*truncation)(x,17,43,true,msb_x1);
            cout << "truncation done" << endl;
            if (has_shortcut){
                x_res = shortcut->operator()(x_res);
            }
            return x + x_res;
        }
};

class ResNet {
    
};
}