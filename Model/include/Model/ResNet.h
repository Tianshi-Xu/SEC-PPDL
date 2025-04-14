#include <LinearLayer/Conv.h>
#include <NonlinearLayer/ReLU.h>
#include <NonlinearLayer/Pool.h>
#include "Primitive.h"
#include <NonlinearOperator/FixPoint.h>
using namespace LinearLayer;
using namespace NonlinearLayer;
using namespace NonlinearOperator;
using namespace std;
namespace Model{

template <typename T, typename IO=Utils::NetIO>
Conv2D* CreateConv(uint64_t in_feature_size, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, CryptoPrimitive<T, IO> *cryptoPrimitive){
    Conv2D* conv;
    if (in_feature_size >=224){
        conv = new Conv2DCheetah(in_feature_size, in_channels, out_channels, kernel_size, stride, cryptoPrimitive->HE);
        return conv;
    }
    switch (cryptoPrimitive->conv_type)
    {
    case Datatype::CONV_TYPE::Nest:
        conv = new Conv2DNest(in_feature_size, in_channels, out_channels, kernel_size, stride, cryptoPrimitive->HE);
        break;
    case Datatype::CONV_TYPE::Cheetah:
        conv = new Conv2DCheetah(in_feature_size, in_channels, out_channels, kernel_size, stride, cryptoPrimitive->HE);
        break;
    }
    return conv;
}

template <typename T, typename IO=Utils::NetIO>
class BasicBlock{
    public:
        int expansion = 1;
        int in_planes;
        int planes;
        int stride = 1;
        ReLU<T, IO> *relu;
        FixPoint<T> *fixpoint;
        Conv2D *conv1;
        Conv2D *conv2;
        Conv2D *shortcut;
        bool has_shortcut = false;
        BasicBlock(uint64_t in_feature_size, uint64_t in_planes, uint64_t planes, uint64_t stride, CryptoPrimitive<T, IO> *cryptoPrimitive){
            this->in_planes = in_planes;
            this->planes = planes;
            this->stride = stride;
            this->relu = cryptoPrimitive->relu;
            this->fixpoint = cryptoPrimitive->fixpoint;
            conv1 = CreateConv<T, IO>(in_feature_size, in_planes, planes, 3, stride, cryptoPrimitive);
            conv2 = CreateConv<T, IO>(in_feature_size/stride, planes, planes, 3, 1, cryptoPrimitive);
            if (stride != 1 || in_planes != planes){
                has_shortcut = true;
                shortcut = CreateConv<T, IO>(in_feature_size, in_planes, planes, 1, stride, cryptoPrimitive);
            }
        }

        Tensor<T> operator()(Tensor<T> &x){
            Tensor<T> x_res = x;
            x = (*conv1)(x);
            (*relu)(x);
            fixpoint->truncate(x,17,43,true,true);
            x = (*conv2)(x);
            if (has_shortcut){
                x_res = (*shortcut)(x_res);
            }
            x = x + x_res;
            (*relu)(x);
            fixpoint->truncate(x,17,43,true,true);
            return x;
        }
};

template <typename T, typename IO=Utils::NetIO>
class Bottleneck{
    public:
        int expansion = 4;
        int in_planes;
        int planes;
        int stride = 1;
        bool has_shortcut = false;
        ReLU<T, IO> *relu;
        FixPoint<T> *fixpoint;
        // TODO: can be simplified to use pointer
        Conv2D *conv1;
        Conv2D *conv2;
        Conv2D *conv3;
        Conv2D *shortcut;

        Bottleneck(uint64_t in_feature_size, uint64_t in_planes, uint64_t planes, uint64_t stride, CryptoPrimitive<T, IO> *cryptoPrimitive){
            this->in_planes = in_planes;
            this->planes = planes;
            this->stride = stride;
            this->relu = cryptoPrimitive->relu;
            this->fixpoint = cryptoPrimitive->fixpoint;
            this->conv1 = CreateConv<T, IO>(in_feature_size, in_planes, planes, 1, 1, cryptoPrimitive);
            this->conv2 = CreateConv<T, IO>(in_feature_size, planes, planes, 3, stride, cryptoPrimitive);
            this->conv3 = CreateConv<T, IO>(in_feature_size/stride, planes, planes * this->expansion, 1, 1, cryptoPrimitive);
            if (stride != 1 || in_planes != planes * this->expansion){
                has_shortcut = true;
                this->shortcut = CreateConv<T, IO>(in_feature_size, in_planes, planes * this->expansion, 1, stride, cryptoPrimitive);
            }
        }

        // TODO: can be simplified
        Tensor<T> operator()(Tensor<T> &x){
            Tensor<T> x_res = x;
            x = (*conv1)(x);
            (*relu)(x);
            fixpoint->truncate(x,17,43,true,true);
            x = (*conv2)(x);
            (*relu)(x);
            fixpoint->truncate(x,17,43,true,true);
            x = (*conv3)(x);
            fixpoint->truncate(x,17,43,true,false);
            if (has_shortcut){
                x_res = (*shortcut)(x_res);
            }
            return x + x_res;
        }
};

template <typename T, typename IO=Utils::NetIO>
class ResNet_3stages {
    public:
        int in_feature_size;
        int num_classes;
        int in_planes = 16;
        int* num_layers;
        Conv2D *conv1;
        ReLU<T, IO> *relu;
        FixPoint<T> *fixpoint;
        vector<BasicBlock<T, IO>*> layer1;
        vector<BasicBlock<T, IO>*> layer2;
        vector<BasicBlock<T, IO>*> layer3;
        Conv2D *linear;
        AvgPool2D<T> *avg_pool;
        ResNet_3stages(uint64_t in_feature_size, int* num_layers,int num_classes, CryptoPrimitive<T, IO> *cryptoPrimitive){
            this->in_feature_size = in_feature_size;
            this->num_layers = num_layers;
            this->num_classes = num_classes;
            this->relu = cryptoPrimitive->relu;
            this->fixpoint = cryptoPrimitive->fixpoint;
            conv1 = CreateConv<T, IO>(in_feature_size, 3, 16, 3, 1, cryptoPrimitive);
            _make_layer(layer1, 16, num_layers[0], 1, cryptoPrimitive);
            _make_layer(layer2, 32, num_layers[1], 2, cryptoPrimitive);
            _make_layer(layer3, 64, num_layers[2], 2, cryptoPrimitive);
            linear = CreateConv<T, IO>(1, 64, num_classes, 1, 1, cryptoPrimitive);
            avg_pool = new AvgPool2D<T>(8);
        }

        void _make_layer(vector<BasicBlock<T, IO>*> &layer, int planes, int num_blocks, int stride, CryptoPrimitive<T, IO> *cryptoPrimitive){
            int strides[num_blocks];
            strides[0] = stride;
            for (int i = 1; i < num_blocks; i++){
                strides[i] = 1;
            }
            for (int i = 0; i < num_blocks; i++){
                layer.push_back(new BasicBlock<T, IO>(this->in_feature_size, this->in_planes, planes, strides[i], cryptoPrimitive));
                this->in_planes = planes * 1;
                this->in_feature_size = this->in_feature_size / strides[i];
            }
        }
        // TODO: implement nn.Sequential
        Tensor<T> operator()(Tensor<T> &x){
            x = (*conv1)(x);
            (*relu)(x);
            fixpoint->truncate(x,17,43,true,true);
            for (int i = 0; i < layer1.size(); i++){
                x = (*layer1[i])(x);
            }
            for (int i = 0; i < layer2.size(); i++){
                x = (*layer2[i])(x);
            }
            for (int i = 0; i < layer3.size(); i++){
                x = (*layer3[i])(x);
            }
            x = (*avg_pool)(x);
            x = (*linear)(x);
            return x;
        }
};

template <typename T, typename IO=Utils::NetIO>
class ResNet_4stages {
    public:
        int in_feature_size;
        int num_classes;
        int in_planes = 64;
        int* num_layers;
        Conv2D *conv1;
        ReLU<T, IO> *relu;
        FixPoint<T> *fixpoint;
        vector<BasicBlock<T, IO>*> layer1;
        vector<BasicBlock<T, IO>*> layer2;
        vector<BasicBlock<T, IO>*> layer3;
        vector<BasicBlock<T, IO>*> layer4;
        Conv2D *linear;
        AvgPool2D<T> *avg_pool;
        ResNet_4stages(uint64_t in_feature_size, int* num_layers,int num_classes, CryptoPrimitive<T, IO> *cryptoPrimitive){
            this->in_feature_size = in_feature_size;
            this->num_layers = num_layers;
            this->num_classes = num_classes;
            this->relu = cryptoPrimitive->relu;
            this->fixpoint = cryptoPrimitive->fixpoint;
            conv1 = CreateConv<T, IO>(in_feature_size, 3, this->in_planes, 7, 4, cryptoPrimitive); // we do not support maxpool for now, so we use stride = 4
            this->in_feature_size = this->in_feature_size / 4;
            _make_layer(layer1, 64, num_layers[0], 1, cryptoPrimitive);
            _make_layer(layer2, 128, num_layers[1], 2, cryptoPrimitive);
            _make_layer(layer3, 256, num_layers[2], 2, cryptoPrimitive);
            _make_layer(layer4, 512, num_layers[3], 2, cryptoPrimitive);
            avg_pool = new AvgPool2D<T>(7);
            linear = CreateConv<T, IO>(1, 512, num_classes, 1, 1, cryptoPrimitive);
        }

        void _make_layer(vector<BasicBlock<T, IO>*> &layer, int planes, int num_blocks, int stride, CryptoPrimitive<T, IO> *cryptoPrimitive){
            int strides[num_blocks];
            strides[0] = stride;
            for (int i = 1; i < num_blocks; i++){
                strides[i] = 1;
            }
            for (int i = 0; i < num_blocks; i++){
                layer.push_back(new BasicBlock<T, IO>(this->in_feature_size, this->in_planes, planes, strides[i], cryptoPrimitive));
                this->in_planes = planes * 1;
                this->in_feature_size = this->in_feature_size / strides[i];
            }
        }
        // TODO: implement nn.Sequential
        Tensor<T> operator()(Tensor<T> &x){
            x = (*conv1)(x);
            (*relu)(x);
            fixpoint->truncate(x,17,43,true,true);
            for (int i = 0; i < layer1.size(); i++){
                x = (*layer1[i])(x);
            }
            for (int i = 0; i < layer2.size(); i++){
                x = (*layer2[i])(x);
            }
            for (int i = 0; i < layer3.size(); i++){
                x = (*layer3[i])(x);
            }
            for (int i = 0; i < layer4.size(); i++){
                x = (*layer4[i])(x);
            }
            x = (*avg_pool)(x);
            x = (*linear)(x);
            return x;
        }
};

template <typename T, typename IO=Utils::NetIO>
ResNet_3stages<uint64_t> resnet_32_c10(CryptoPrimitive<T, IO> *cryptoPrimitive){
    return ResNet_3stages<uint64_t>(32, new int[3]{5,5,5}, 10, cryptoPrimitive);
}

template <typename T, typename IO=Utils::NetIO>
ResNet_4stages<uint64_t> resnet_18(CryptoPrimitive<T, IO> *cryptoPrimitive){
    return ResNet_4stages<uint64_t>(224, new int[4]{2,2,2,2}, 1000, cryptoPrimitive);
}

template <typename T, typename IO=Utils::NetIO>
ResNet_4stages<uint64_t> resnet_50(CryptoPrimitive<T, IO> *cryptoPrimitive){
    return ResNet_4stages<uint64_t>(224, new int[4]{3,4,6,3}, 1000, cryptoPrimitive);
}



}