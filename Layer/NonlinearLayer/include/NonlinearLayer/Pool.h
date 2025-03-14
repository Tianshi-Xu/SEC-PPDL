#include <Datatype/Tensor.h>

using namespace Datatype;
using namespace std;
namespace NonlinearLayer {

template<typename T>
class AvgPool2D{
    public:
        uint64_t kernel_size;
        uint64_t stride;
        uint64_t padding;
        AvgPool2D(uint64_t kernel_size, uint64_t stride = -1, uint64_t padding = 0){
            this->kernel_size = kernel_size;
            if (stride == -1){
                this->stride = kernel_size;
            } else {
                this->stride = stride;
            }
            this->padding = padding;
        }
        // we do not support batch size > 1 for now
        Tensor<T> operator()(Tensor<T> &x){
            uint32_t C,H,W,Hout,Wout;
            std::vector<size_t> shape = x.shape();
            C = shape[0];
            H = shape[1];
            W = shape[2];
            // cout << "C,H,W: " << C << "," << H << "," << W << endl;
            Hout = (H - kernel_size + 2 * padding) / stride + 1;
            Wout = (W - kernel_size + 2 * padding) / stride + 1;
            Tensor<T> y({C,Hout,Wout});
            // implement a avg pool below!
            for (uint32_t c = 0; c < C; c++){
                for (uint32_t h = 0; h < Hout; h++){
                    for (uint32_t w = 0; w < Wout; w++){
                            // Initialize sum for averaging
                            float sum = 0;
                            // Sum all values in the pooling window
                        for (uint32_t ph = 0; ph < kernel_size; ph++) {
                            for (uint32_t pw = 0; pw < kernel_size; pw++) {
                                sum += x({c,h*stride + ph,w*stride + pw});
                            }
                        }
                        // Compute average by dividing by window size
                        y({c,h,w}) = sum / (kernel_size * kernel_size);
                    }
                }
            }
            return y;
        }
};





}

