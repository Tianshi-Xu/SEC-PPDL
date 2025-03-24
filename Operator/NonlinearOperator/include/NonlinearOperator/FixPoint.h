#include <OTProtocol/millionaire.h>
#include <OTProtocol/truncation.h>
#pragma once
namespace NonlinearOperator {

// This class contains all the protocols for fixpoint arithmetic
template <typename T>
class FixPoint {
    public:
        int num_threads;
        int party;
        FixPoint(TruncationProtocol **truncationProtocol, OTProtocol::AuxProtocols **aux, int num_threads=4){
            this->num_threads = num_threads;
            this->truncationProtocol = truncationProtocol;
            this->aux = aux;
        }

        FixPoint(int party, OTPack<Utils::NetIO> **otpack, int num_threads=4){
            this->party = party;
            this->num_threads = num_threads;
            this->truncationProtocol = new TruncationProtocol*[num_threads];
            this->aux = new OTProtocol::AuxProtocols*[num_threads];
            for (int i = 0; i < num_threads; i++){
                this->truncationProtocol[i] = new TruncationProtocol(party, otpack[i]);
                this->aux[i] = new OTProtocol::AuxProtocols(party, otpack[i]->io, otpack[i]);
            }
        }

        // we do not implement larger than to avoid the complexity of millionaire protocol
        void less_than_zero(Tensor<T> &x, Tensor<uint8_t> &result, int32_t bw){
            auto shape = x.shape();
            int dim = x.size();
            T* x_flatten = x.data().data();
            uint8_t* result_flatten = result.data().data();
            std::thread less_than_threads[num_threads];
            int chunk_size = dim / num_threads;
            for (int i = 0; i < num_threads; i++) {
                int offset = i * chunk_size;
                less_than_threads[i] = std::thread(less_than_thread, aux[i], x_flatten+offset, result_flatten+offset, chunk_size, bw);
            }
            for (int i = 0; i < num_threads; i++) {
                less_than_threads[i].join();
            }
        }
        
        // return 1{x < constant}
        void less_than_constant(Tensor<T> &x, T constant, Tensor<uint8_t> &result, int32_t bw){
            auto shape = x.shape();
            int dim = x.size();
            Tensor<T> y = Tensor<T>(shape, constant);
            Tensor<T> z;
            if (party == ALICE){
                z = x - y;
            }
            else{
                z = x;
            }
            less_than_zero(z, result, bw);
        }

        // return 1{x < y}
        void less_than(Tensor<T> &x, Tensor<T> &y, Tensor<uint8_t> &result, int32_t bw){
            auto shape = x.shape();
            int dim = x.size();
            auto z = x - y;
            less_than_zero(z, result, bw);
        }

        // for now, only support uint64_t. TODO: support other types
        void truncate(Tensor<T> &x, int32_t shift, int32_t bw, bool signed_arithmetic=true, bool msb_zero=false){
            uint8_t *msb_x = nullptr;
            if (msb_zero){
                msb_x = new uint8_t[x.size()];
                memset(msb_x, 0, x.size());
            }
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

        // for now, only support uint64_t
        void truncate_reduce(Tensor<T> &x, int32_t shift, int32_t bw){
            auto shape = x.shape();
            int dim = x.size();
            x.flatten();
            T* x_flatten = x.data().data();
            std::thread truncation_threads[num_threads];
            int chunk_size = dim / num_threads;
            for (int i = 0; i < num_threads; i++) {
                int offset = i * chunk_size;
                truncation_threads[i] = std::thread(truncate_reduce_thread, truncationProtocol[i], x_flatten+offset, x_flatten+offset, chunk_size, shift, bw);
            }
            for (int i = 0; i < num_threads; i++) {
                truncation_threads[i].join();
            }
            x.reshape(shape);
        }

        // for now, T only support uint64_t
        void extend(Tensor<T> &x, int32_t bwA, int32_t bwB, bool signed_arithmetic=true, bool msb_zero=false){
            uint8_t *msb_x = nullptr;
            if (msb_zero){
                msb_x = new uint8_t[x.size()];
                memset(msb_x, 0, x.size());
            }
            int dim = x.size();
            T* x_flatten = x.data().data();
            std::thread extend_threads[num_threads];
            int chunk_size = dim / num_threads;
            for (int i = 0; i < num_threads; i++) {
                int offset = i * chunk_size;
                extend_threads[i] = std::thread(extend_thread, aux[i], x_flatten+offset, x_flatten+offset, chunk_size, bwA, bwB, signed_arithmetic, msb_x);
            }
            for (int i = 0; i < num_threads; i++) {
                extend_threads[i].join();
            }
        }

        // Conversion from ring to field
        void Ring2Field(Tensor<T> &x, int Q, int bitwidth = 0, bool signed_arithmetic=false){
            if (bitwidth == 0){
                bitwidth = x.bitwidth;
            }
            // cout << "bitwidth: " << bitwidth << endl;
            int ext_bit = 20;
            extend(x, bitwidth, bitwidth+ext_bit, signed_arithmetic);
            if (party == ALICE){
                for (int i = 0; i < x.size(); i++){
                    x(i) = x(i) % Q;
                }
            }
            else{
                for (int i = 0; i < x.size(); i++){
                    x(i) = (x(i) + (1ULL<<(bitwidth+ext_bit))*(Q-1))% Q; // can not use -1ULL<<bitwidth, because it is negative, no modulo operation. It may go wrong when it exceeds uint64_t
                }
            }
        }

        // Conversion from field to ring, ring must larger than field
        void Field2Ring(Tensor<T> &x, int Q, int bitwidth = 0){
            if (bitwidth == 0){
                bitwidth = x.bitwidth;
            }
            std::thread ring2field_threads[num_threads];
            int chunk_size = x.size() / num_threads;
            for (int i = 0; i < num_threads; i++) {
                int offset = i * chunk_size;
                ring2field_threads[i] = std::thread(ring2field_thread, aux[i], x.data().data()+offset, x.data().data()+offset, chunk_size, bitwidth, Q);
            }
            for (int i = 0; i < num_threads; i++) {
                ring2field_threads[i].join();
            }
        }
        
    private:
        TruncationProtocol **truncationProtocol = nullptr;
        OTProtocol::AuxProtocols **aux = nullptr;

        void static less_than_thread(AuxProtocols *aux, T* input, uint8_t* result, int lnum_ops, int32_t bw){
            aux->MSB<T>(input, result,lnum_ops,bw);
        }

        void static truncation_thread(TruncationProtocol *truncationProtocol, T* input, T* result, int lnum_ops, int32_t shift, int32_t bw, bool signed_arithmetic=true, uint8_t *msb_x=nullptr){
            truncationProtocol->truncate(lnum_ops, input, result, shift, bw, signed_arithmetic, msb_x);
        }

        void static truncate_reduce_thread(TruncationProtocol *truncationProtocol, T* input, T* result, int lnum_ops, int32_t shift, int32_t bw){
            truncationProtocol->truncate_and_reduce(lnum_ops, input, result, shift, bw);
        }

        void static extend_thread(AuxProtocols *aux, T* input, T* result, int lnum_ops, int32_t bwA, int32_t bwB, bool signed_arithmetic=true, uint8_t *msb_x=nullptr){
            if (signed_arithmetic){
                aux->s_extend(lnum_ops, input, result, bwA, bwB, msb_x);
            } else {
                aux->z_extend(lnum_ops, input, result, bwA, bwB, msb_x);
            }
        }

        void static ring2field_thread(AuxProtocols *aux, T* input, T* result, int lnum_ops, int32_t bw, int Q){
            assert(bw>=(int32_t)ceil(std::log2(Q)));
            uint64_t mask_bw = (bw == 64 ? -1 : ((1ULL << bw) - 1));
            uint8_t *wrap_x = new uint8_t[lnum_ops];
            aux->wrap_computation_prime(input, wrap_x, lnum_ops, bw, Q);
            uint64_t *arith_wrap = new uint64_t[lnum_ops];
            aux->B2A(wrap_x, arith_wrap, lnum_ops, bw);
            for (int i = 0; i < lnum_ops; i++){
                result[i] = ((input[i]%Q) - Q * arith_wrap[i]) & mask_bw;
            }
            delete[] wrap_x;
            delete[] arith_wrap;
        }
};

}