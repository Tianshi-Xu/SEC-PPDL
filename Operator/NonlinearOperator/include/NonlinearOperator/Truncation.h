#include <OTProtocol/truncation.h>

namespace NonlinearOperator {

// now only support uint64_t
template <typename T>
class Truncation {
    public:
        int num_threads;
        
        Truncation(TruncationProtocol **truncationProtocol, int num_threads=4){
            this->num_threads = num_threads;
            this->truncationProtocol = truncationProtocol;
        }
        // for now, only support uint64_t
        void operator()(Tensor<T> &x, int32_t shift, int32_t bw, bool signed_arithmetic=true, uint8_t *msb_x=nullptr){
            auto shape = x.shape();
            int dim = x.size();
            x.flatten();
            T* x_flatten = x.data().data();
            std::thread truncation_threads[num_threads];
            int chunk_size = dim / num_threads;
            for (int i = 0; i < num_threads; i++) {
                int offset = i * chunk_size;
                truncation_threads[i] = std::thread(truncation_thread, truncationProtocol[i], x_flatten+offset, x_flatten+offset, chunk_size, shift, bw, signed_arithmetic, msb_x);
            }
            for (int i = 0; i < num_threads; i++) {
                truncation_threads[i].join();
            }
            x.reshape(shape);
        }

    private:
        TruncationProtocol **truncationProtocol = nullptr;
        void static truncation_thread(TruncationProtocol *truncationProtocol, T* result, T* input, int lnum_ops, int32_t shift, int32_t bw, bool signed_arithmetic=true, uint8_t *msb_x=nullptr){
            truncationProtocol->truncate(lnum_ops, input, result, shift, bw, signed_arithmetic, msb_x);
        }
};

}