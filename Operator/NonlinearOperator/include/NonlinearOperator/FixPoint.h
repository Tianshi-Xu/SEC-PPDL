#include <OTProtocol/millionaire.h>
#include <OTProtocol/truncation.h>
#include <seal/util/common.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <type_traits>
#include <array>
#include <Utils/prg.h>
#pragma once
namespace NonlinearOperator {

// This class contains all the protocols for fixpoint arithmetic
template <typename T>
class FixPoint {
    public:
        int num_threads;
        int party;
        FixPoint(TruncationProtocol **truncationProtocol, OTProtocol::AuxProtocols **aux, int num_threads=4){
            this->party = aux[0]->party;
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

        // we do not implement larger than to reduce the complexity of millionaire protocol
        void less_than_zero(Tensor<T> &x, Tensor<uint8_t> &result, int32_t bw){
            auto shape = x.shape();
            int dim = x.size();
            T* x_flatten = x.data().data();
            uint8_t* result_flatten = result.data().data();
            const int workers = std::min(dim, num_threads);
            std::vector<std::thread> less_than_threads;
            for (int i = 0; i < workers; i++) {
                const int offset = int64_t(dim) * i / workers;
                const int end = int64_t(dim) * (i + 1) / workers;
                less_than_threads.emplace_back(less_than_thread, aux[i], x_flatten+offset, result_flatten+offset, end-offset, bw);
            }
            for (auto &worker : less_than_threads) {
                worker.join();
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

        // return 1{constant < x}
        void less_than_constant(T constant, Tensor<T> &x, Tensor<uint8_t> &result, int32_t bw){
            auto shape = x.shape();
            int dim = x.size();
            Tensor<T> y = Tensor<T>(shape, constant);
            Tensor<T> z;
            if (party == ALICE){
                z = y - x;
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
        void truncate(Tensor<T> &x, int32_t shift, int32_t bw, bool msb_zero=false){
            truncate(x, shift, bw, msb_zero, std::is_signed_v<T>);
        }

        void truncate(Tensor<T> &x, int32_t shift, int32_t bw, bool msb_zero, bool signed_arithmetic){
            if (shift == 0) {
                return;
            }
            int dim = x.size();
            T* x_flatten = x.data().data();
            std::vector<uint8_t> msb(msb_zero ? dim : 0, 0);
            const int workers = std::min(dim, num_threads);
            std::vector<std::thread> truncation_threads;
            for (int i = 0; i < workers; i++) {
                const int offset = int64_t(dim) * i / workers;
                const int end = int64_t(dim) * (i + 1) / workers;
                truncation_threads.emplace_back(truncation_thread, truncationProtocol[i], x_flatten+offset, x_flatten+offset, end-offset, shift, bw, signed_arithmetic, msb_zero ? msb.data()+offset : nullptr);
            }
            for (auto &worker : truncation_threads) {
                worker.join();
            }
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
        void extend(Tensor<T> &x, int32_t bwA, int32_t bwB, bool msb_zero=false){
            int dim = x.size();
            T* x_flatten = x.data().data();
            const int workers = std::min(dim, num_threads);
            std::vector<std::thread> extend_threads;
            const bool signed_arithmetic = std::is_signed_v<T>;
            for (int i = 0; i < workers; i++) {
                const int offset = int64_t(dim) * i / workers;
                const int end = int64_t(dim) * (i + 1) / workers;
                extend_threads.emplace_back(extend_thread, aux[i], x_flatten+offset, x_flatten+offset, end-offset, bwA, bwB, signed_arithmetic, msb_zero);
            }
            for (auto &worker : extend_threads) {
                worker.join();
            }
        }

        // Helper traits to keep modulo arithmetic in a wide accumulator
        template <typename FieldT = T>
        struct FieldTraits {
            using Wide = __uint128_t;
            static inline Wide widen(FieldT value) { return static_cast<Wide>(value); }
            static inline Wide mod_reduce(Wide value, Wide Q) { return value % Q; }
            static inline FieldT narrow(Wide value) { return static_cast<FieldT>(value); }
        };

        using ModulusType = std::conditional_t<std::is_same_v<T, int128_t>, int128_t, uint64_t>;

        void check_share(Tensor<T> &x, ModulusType mod, const std::string &name) {
            if (mod == 0) {
                if (party == BOB) {
                    std::cout << "check " << name << ": skipped (mod=0)" << std::endl;
                }
                return;
            }

            if (party == ALICE) {
                aux[0]->io->send_tensor(x);
                return;
            }

            Tensor<T> reconstructed(x.shape());
            aux[0]->io->recv_tensor(reconstructed);

            using Wide = typename FieldTraits<T>::Wide;
            Wide modulus = static_cast<Wide>(mod);
            for (size_t i = 0; i < x.size(); i++) {
                Wide sum = FieldTraits<T>::widen(reconstructed(i));
                sum += FieldTraits<T>::widen(x(i));
                reconstructed(i) = FieldTraits<T>::narrow(FieldTraits<T>::mod_reduce(sum, modulus));
            }
            cout << "check " << name << ": ";
            for (int i=2040; i<2060; i++){
                printf("reconstructed[%d]: %lu\n", i, reconstructed(i));
            }
            for (int i=4090; i<4110; i++){
                printf("reconstructed[%d]: %lu\n", i, reconstructed(i));
            }
        }
        // Conversion from ring to field
        void Ring2Field(Tensor<T> &x, ModulusType Q, int bitwidth = 0){
            if (bitwidth == 0){
                bitwidth = x.bitwidth;
            }
            // cout << "bitwidth: " << bitwidth << endl;
            int ext_bit = 60 - bitwidth;
            extend(x, bitwidth, bitwidth + ext_bit);
            int total_bw = bitwidth + ext_bit;
            if (total_bw < 0) total_bw = 0;
            if (total_bw >= static_cast<int>(sizeof(ModulusType) * 8)) {
                total_bw = static_cast<int>(sizeof(ModulusType) * 8) - 1;
            }
            ModulusType debug_mod = ModulusType(1) << total_bw;
            // check_share(x, debug_mod, "x after extend");
            // for (int i=2040; i<2070; i++){
            //     printf("x after extend[%d]: %lu\n", i, x(i));
            // }
            using Wide = typename FieldTraits<T>::Wide;
            Wide modulus = static_cast<Wide>(Q);
            if (party == ALICE){
                for (int i = 0; i < x.size(); i++){
                    Wide reduced = FieldTraits<T>::mod_reduce(FieldTraits<T>::widen(x(i)), modulus);
                    x(i) = FieldTraits<T>::narrow(reduced);
                }
            }
            else{
                Wide neg_2k = static_cast<Wide>(1);
                neg_2k <<= (bitwidth + ext_bit);
                Wide modulus_minus_one = modulus - static_cast<Wide>(1);
                neg_2k = FieldTraits<T>::mod_reduce(neg_2k * modulus_minus_one, modulus);
                for (int i = 0; i < x.size(); i++){
                    Wide value = FieldTraits<T>::widen(x(i)) + neg_2k;
                    x(i) = FieldTraits<T>::narrow(FieldTraits<T>::mod_reduce(value, modulus)); // can not use -1ULL<<bitwidth, because it is negative, no modulo operation. It may go wrong when it exceeds uint64_t
                }
            }
        }

        // Conversion from Q to bitwidth, if ceil(log2(Q)) > bitwidth, first extend to ceil(log2(Q)), then truncate to bitwidth
        void Field2Ring(Tensor<T> &x, ModulusType Q, int bitwidth = 0){
            if (bitwidth == 0){
                bitwidth = x.bitwidth; 
            }
            const bool signed_arithmetic = std::is_signed_v<T>;
            // cout << "bw, log2Q:" << bitwidth << " " << ceil(std::log2(Q)) << endl;
            if constexpr (sizeof(T) == sizeof(uint64_t)) {
                const int dim = x.size();
                const int workers = std::min(dim, num_threads);
                std::vector<std::thread> field2ring_threads;
                for (int i = 0; i < workers; i++) {
                    const int offset = int64_t(dim) * i / workers;
                    const int end = int64_t(dim) * (i + 1) / workers;
                    field2ring_threads.emplace_back(field2ring_thread, aux[i], x.data().data()+offset, x.data().data()+offset, end-offset, bitwidth, Q, signed_arithmetic);
                }
                for (auto &worker : field2ring_threads) {
                    worker.join();
                }
            } else if constexpr (std::is_same_v<T, int128_t>) {
                auto chunk_size = x.size() / num_threads;
                std::thread field2ring_threads[num_threads];
                for (int i = 0; i < num_threads; i++) {
                    int offset = i * chunk_size;
                    field2ring_threads[i] = std::thread(field2ring_thread_128, aux[i], x.data().data()+offset, x.data().data()+offset, chunk_size, bitwidth, Q, signed_arithmetic);
                }
                for (int i = 0; i < num_threads; i++) {
                    field2ring_threads[i].join();
                }
            }
        }

        // The signed input must fit in both the input ring and the centered field.
        void SignedRing2Field(Tensor<uint64_t> &x, uint64_t modulus, int bitwidth) {
            if (modulus < 2 || modulus > (uint64_t(1) << 63) || bitwidth < 1 || bitwidth > 64) {
                throw std::invalid_argument("Invalid signed ring-to-field parameters");
            }
            if (x.size() == 0) {
                return;
            }
            const int size = x.size();
            const uint64_t half_ring = uint64_t(1) << (bitwidth - 1);
            const uint64_t mask = half_ring + (half_ring - 1);
            const uint64_t correction = (static_cast<__uint128_t>(half_ring) * 2) % modulus;
            const uint64_t offset = half_ring % modulus;
            if (party == ALICE) {
                for (auto &value : x.data()) {
                    value += half_ring;
                }
            }
            std::vector<uint8_t> wrap(size);
            aux[0]->wrap_computation(x.data().data(), wrap.data(), size, bitwidth);
            std::vector<uint64_t> correction_share(size);
            if (party == ALICE) {
                Utils::RandomFieldElements(correction_share.data(), size, modulus);
                std::vector<std::array<uint64_t, 4>> messages(size);
                std::vector<uint64_t *> rows(size);
                for (int i = 0; i < size; ++i) {
                    for (int choice = 0; choice < 4; ++choice) {
                        const uint64_t value = (wrap[i] ^ (choice & 1)) * correction;
                        messages[i][choice] = (value + modulus - correction_share[i]) % modulus;
                    }
                    rows[i] = messages[i].data();
                }
                aux[0]->lookup_table<uint64_t>(rows.data(), nullptr, nullptr, size, 2, 64);
            } else {
                std::vector<uint64_t> choices(wrap.begin(), wrap.end());
                aux[0]->lookup_table<uint64_t>(nullptr, choices.data(), correction_share.data(), size, 2, 64);
            }
            for (int i = 0; i < size; ++i) {
                uint64_t value = ((x(i) & mask) % modulus + modulus - correction_share[i]) % modulus;
                x(i) = party == ALICE ? (value + modulus - offset) % modulus : value;
            }
        }

        // The reconstructed signed input plus 2^(bitwidth-2) must be in
        // the lower half of the source ring. Its wrap bit is then the OR
        // of the two local MSBs.
        void BoundedRing2Field(Tensor<uint64_t> &x, uint64_t modulus, int bitwidth) {
            if (modulus < 3 || modulus > (uint64_t(1) << 63) || bitwidth < 2 || bitwidth > 64) {
                throw std::invalid_argument("Invalid bounded ring-to-field parameters");
            }
            if (x.size() == 0) {
                return;
            }
            const int size = x.size();
            const uint64_t half_ring = uint64_t(1) << (bitwidth - 1);
            const uint64_t mask = half_ring + (half_ring - 1);
            const uint64_t offset = half_ring / 2;
            const uint64_t correction = (static_cast<__uint128_t>(half_ring) * 2) % modulus;
            std::vector<uint8_t> high(size);
            for (int i = 0; i < size; ++i) {
                x(i) = (x(i) + (party == ALICE ? offset : 0)) & mask;
                high[i] = (x(i) & half_ring) != 0;
            }

            // A masked 1-out-of-2 OT implements the prime-field correction
            // using the existing OT interface.
            auto *ot = aux[0]->otpack->iknp_straight;
            const int payload_bits = seal::util::get_significant_bit_count(modulus - 1);
            std::vector<uint64_t> correction_share(size);
            if (party == ALICE) {
                Utils::RandomFieldElements(correction_share.data(), size, modulus);
                std::vector<std::array<uint64_t, 2>> messages(size);
                std::vector<uint64_t *> rows(size);
                for (int i = 0; i < size; ++i) {
                    const uint64_t negative_mask = modulus - correction_share[i];
                    messages[i] = {(high[i] * correction + negative_mask) % modulus,
                                   (correction + negative_mask) % modulus};
                    rows[i] = messages[i].data();
                }
                ot->send(rows.data(), size, payload_bits);
            } else {
                ot->recv(correction_share.data(), high.data(), size, payload_bits);
            }

            const uint64_t local_offset = party == ALICE ? offset % modulus : 0;
            for (int i = 0; i < size; ++i) {
                const uint64_t unwrapped = (x(i) % modulus + modulus - correction_share[i]) % modulus;
                x(i) = (unwrapped + modulus - local_offset) % modulus;
            }
        }

        // The reconstructed signed input plus floor(modulus/4) must be
        // in [0, ceil(modulus/2)). A nonzero shift fuses truncation and
        // can differ from floor(x / 2^shift) by one output unit.
        void BoundedField2Ring(Tensor<uint64_t> &x, uint64_t modulus, int bitwidth, int shift = 0) {
            if (modulus < 3 || modulus > (uint64_t(1) << 63) || bitwidth < 1 || bitwidth > 64 ||
                shift < 0 || shift >= 63) {
                throw std::invalid_argument("Invalid bounded field-to-ring parameters");
            }
            if (x.size() == 0) {
                return;
            }
            const int size = x.size();
            const uint64_t mask = bitwidth == 64 ? UINT64_MAX : (uint64_t(1) << bitwidth) - 1;
            const uint64_t offset = party == ALICE ? modulus / 4 : 0;
            const uint64_t half_field = (modulus + 1) / 2;
            const uint64_t correction = (modulus >> shift) & mask;
            const int64_t divisor = int64_t(1) << shift;
            std::vector<uint8_t> high(size);
            for (int i = 0; i < size; ++i) {
                const uint64_t shifted_share = (x(i) % modulus + offset) % modulus;
                high[i] = shifted_share >= half_field;
                const int64_t local = static_cast<int64_t>(shifted_share) - static_cast<int64_t>(offset);
                x(i) = static_cast<uint64_t>(local / divisor - (local % divisor < 0));
            }

            auto *ot = aux[0]->otpack->iknp_straight;
            std::vector<uint64_t> correction_share(size);
            if (party == ALICE) {
                std::vector<uint64_t> correlation(size);
                for (int i = 0; i < size; ++i) {
                    correlation[i] = high[i] ? 0 : correction;
                }
                ot->send_cot(correction_share.data(), correlation.data(), size, bitwidth);
                for (int i = 0; i < size; ++i) {
                    correction_share[i] = (high[i] ? correction : 0) - correction_share[i];
                }
            } else {
                ot->recv_cot(correction_share.data(), reinterpret_cast<bool *>(high.data()), size, bitwidth);
            }
            for (int i = 0; i < size; ++i) {
                x(i) = (x(i) - correction_share[i]) & mask;
            }
        }

        // Multiply additive shares in Z/(2^bitwidth). Each OT lane selects one
        // party's local multiplier bit, so no carry bits need to be opened.
        Tensor<uint64_t> Multiply(const Tensor<uint64_t> &x, const Tensor<uint64_t> &y, int bitwidth) {
            if (x.shape() != y.shape() || x.size() == 0 || bitwidth < 1 || bitwidth > 64) {
                throw std::invalid_argument("Invalid ring multiplication inputs");
            }
            const size_t size = x.size();
            const size_t count = size * bitwidth;
            Tensor<uint64_t> input({2 * count});
            Tensor<uint64_t> product({2 * count});
            Tensor<uint8_t> select({2 * count});
            const size_t lane = party == ALICE ? 0 : count;
            for (size_t i = 0; i < size; ++i) {
                for (int bit = 0; bit < bitwidth; ++bit) {
                    const size_t index = i * bitwidth + bit;
                    input(index) = input(index + count) = x(i);
                    select(index + lane) = (y(i) >> bit) & 1;
                }
            }
            mux(select, input, product, bitwidth, bitwidth);
            const uint64_t mask = bitwidth == 64 ? UINT64_MAX : (uint64_t(1) << bitwidth) - 1;
            Tensor<uint64_t> result(x.shape());
            for (size_t i = 0; i < size; ++i) {
                for (int bit = 0; bit < bitwidth; ++bit) {
                    const size_t index = i * bitwidth + bit;
                    result(i) += (product(index) + product(index + count)) << bit;
                }
                result(i) &= mask;
            }
            return result;
        }

        uint8_t BooleanAnd(uint8_t x, uint8_t y) {
            uint8_t result = 0;
            aux[0]->AND(&x, &y, &result, 1);
            return result;
        }
        
        // return b*input
        void mux(Tensor<uint8_t> &b, Tensor<T> &input, Tensor<T> &result, int32_t bwA, int32_t bwB){
            int dim = input.size();
            T* input_flatten = input.data().data();
            T* result_flatten = result.data().data();
            const int workers = std::min(dim, num_threads);
            std::vector<std::thread> mux_threads;
            for (int i = 0; i < workers; i++) {
                const int offset = int64_t(dim) * i / workers;
                const int end = int64_t(dim) * (i + 1) / workers;
                mux_threads.emplace_back(mux_thread, aux[i], b.data().data()+offset, input_flatten+offset, result_flatten+offset, end-offset, bwA, bwB);
            }
            for (auto &worker : mux_threads) {
                worker.join();
            }
        }
    
        void max_2d(Tensor<T> &x, Tensor<T> &result, int32_t dim=1, int32_t bw=0, int32_t scale=0){
            if(x.shape().size() != 2){
                throw std::invalid_argument("max_2d only support 2D tensor");
            }
            if((bw==0 && x.bitwidth == 0) || (scale==0 && x.scale==0)){
                throw std::invalid_argument("bw or scale is 0, please set bw and scale for max_2d");
            }
            if(bw==0){
                bw = x.bitwidth;
            }
            if(scale==0){
                scale = x.scale;
            }
            uint64_t d0 = x.shape()[0];
            uint64_t d1 = x.shape()[1];
            result.reshape({d0, d1});
        }

        // Secure rounding protocol
        void secure_round(Tensor<T> &x, int32_t s_fix, int32_t bw_fix, int32_t bw_acc) {
            auto shape = x.shape();
            int dim = x.size();
            x.flatten();
            T* x_flatten = x.data().data();
            
            std::thread round_threads[num_threads];
            int chunk_size = dim / num_threads;
            for (int i = 0; i < num_threads; i++) {
                int offset = i * chunk_size;
                round_threads[i] = std::thread(secure_round_thread, aux[i], 
                                                x_flatten + offset, x_flatten + offset, 
                                                chunk_size, s_fix, bw_fix, bw_acc, party);
            }
            for (int i = 0; i < num_threads; i++) {
                round_threads[i].join();
            }
            x.reshape(shape);
        }

        // Secure requantization protocol
        // Algorithm 1: from b_acc to b_acc with scale change (s to s')
        // Algorithm 2: from b_acc to b_fix with scale s to scale 2^{s_fix}
        // Algorithm 3: from b_fix to b_acc with scale change
        void secure_requant(Tensor<T> &x, double scale_in, double scale_out, 
                           int32_t bw_in, int32_t bw_out, int32_t s_fix) {
            auto shape = x.shape();
            int dim = x.size();
            x.flatten();
            T* x_flatten = x.data().data();
            
            std::thread requant_threads[num_threads];
            int chunk_size = dim / num_threads;
            for (int i = 0; i < num_threads; i++) {
                int offset = i * chunk_size;
                requant_threads[i] = std::thread(secure_requant_thread, aux[i],
                                                  x_flatten + offset, x_flatten + offset,
                                                  chunk_size, scale_in, scale_out,
                                                  bw_in, bw_out, s_fix, party);
            }
            for (int i = 0; i < num_threads; i++) {
                requant_threads[i].join();
            }
            x.reshape(shape);
        }
    private:
        TruncationProtocol **truncationProtocol = nullptr;
        OTProtocol::AuxProtocols **aux = nullptr;

        void static less_than_thread(AuxProtocols *aux, T* input, uint8_t* result, int lnum_ops, int32_t bw){
            aux->MSB<T>(input, result,lnum_ops,bw);
        }

        void static truncation_thread(TruncationProtocol *truncationProtocol, T* input, T* result, int lnum_ops, int32_t shift, int32_t bw, bool signed_arithmetic=true, uint8_t *msb_x=nullptr){
            if constexpr (sizeof(T) == sizeof(uint64_t)) {
                auto input_u64 = reinterpret_cast<uint64_t*>(input);
                auto result_u64 = reinterpret_cast<uint64_t*>(result);
                truncationProtocol->truncate(lnum_ops, input_u64, result_u64, shift, bw, signed_arithmetic, msb_x);
            } else {
                static_assert(sizeof(T) == sizeof(uint64_t),
                              "truncate only supports 64-bit tensor types at the moment");
            }
        }

        void static truncate_reduce_thread(TruncationProtocol *truncationProtocol, T* input, T* result, int lnum_ops, int32_t shift, int32_t bw){
            if constexpr (sizeof(T) == sizeof(uint64_t)) {
                auto input_u64 = reinterpret_cast<uint64_t*>(input);
                auto result_u64 = reinterpret_cast<uint64_t*>(result);
                truncationProtocol->truncate_and_reduce(lnum_ops, input_u64, result_u64, shift, bw);
            } else {
                static_assert(sizeof(T) == sizeof(uint64_t),
                              "truncate_reduce only supports 64-bit tensor types at the moment");
            }
        }

        void static extend_thread(AuxProtocols *aux, T* input, T* result, int lnum_ops, int32_t bwA, int32_t bwB, bool signed_arithmetic=true, bool msb_zero=false){
            uint8_t *msb_x = nullptr;
            if (msb_zero){
                msb_x = new uint8_t[lnum_ops];
                memset(msb_x, 0, lnum_ops);
            }
            if constexpr (std::is_same_v<T, int128_t> || std::is_same_v<T, __int128>) {
                // 128-bit version
                if (signed_arithmetic){
                    aux->s_extend(lnum_ops, reinterpret_cast<int128_t*>(input), reinterpret_cast<int128_t*>(result), bwA, bwB, msb_x);
                } else {
                    aux->z_extend(lnum_ops, reinterpret_cast<int128_t*>(input), reinterpret_cast<int128_t*>(result), bwA, bwB, msb_x);
                }
            } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
                // Treat any 64-bit integral T as the uint64_t view expected by AuxProtocols
                auto input_u64 = reinterpret_cast<uint64_t*>(input);
                auto result_u64 = reinterpret_cast<uint64_t*>(result);
                if (signed_arithmetic){
                    aux->s_extend(lnum_ops, input_u64, result_u64, bwA, bwB, msb_x);
                } else {
                    aux->z_extend(lnum_ops, input_u64, result_u64, bwA, bwB, msb_x);
                }
            } else {
                static_assert(std::is_same_v<T, int128_t> || std::is_same_v<T, __int128> || sizeof(T) == sizeof(uint64_t),
                              "FixPoint::extend only supports 64-bit or 128-bit integer tensor types");
            }
            if (msb_zero) {
                delete[] msb_x;
            }
        }

        static void field2ring_thread(AuxProtocols *aux, T* input, T* result, int lnum_ops, int32_t bw, uint64_t Q, bool signed_arithmetic){
            auto input_u64 = reinterpret_cast<uint64_t*>(input);
            uint64_t mask_bw = (bw == 64 ? static_cast<uint64_t>(-1) : ((1ULL << bw) - 1));
            uint8_t *wrap_x = new uint8_t[lnum_ops];
            int32_t logQ = static_cast<int32_t>(ceil(std::log2(Q)));
            if (bw < logQ){
                aux->wrap_computation_prime(input_u64, wrap_x, lnum_ops, logQ, Q);
            }
            else{
                aux->wrap_computation_prime(input_u64, wrap_x, lnum_ops, bw, Q);
            }
            uint64_t *arith_wrap = new uint64_t[lnum_ops];
            aux->B2A(wrap_x, arith_wrap, lnum_ops, bw);
            const uint64_t signed_threshold = (bw == 0 ? 0 : (1ULL << (bw - 1)));
            for (int i = 0; i < lnum_ops; i++){
                __int128 raw = static_cast<uint64_t>(input_u64[i]);
                raw -= static_cast<__int128>(Q) * static_cast<__int128>(arith_wrap[i]);
                uint64_t masked = static_cast<uint64_t>(raw) & mask_bw;
                result[i] = static_cast<T>(masked);
            }
            delete[] wrap_x;
            delete[] arith_wrap;
        }

        static void field2ring_thread_128(AuxProtocols *aux, int128_t *input, int128_t *result, int lnum_ops, int32_t bw, int128_t Q, bool signed_arithmetic) {
            int128_t mask_bw = (bw == 128 ? int128_t(-1) : ((int128_t(1) << bw) - 1));
            uint8_t *wrap_x = new uint8_t[lnum_ops];
            long double q_ld = static_cast<long double>(Q);
            int32_t logQ = static_cast<int32_t>(ceil(std::log2(q_ld)));
            int32_t wrap_bw = (bw < logQ) ? logQ : bw;
            aux->wrap_computation_prime(input, wrap_x, lnum_ops, wrap_bw, static_cast<int128_t>(Q));
            uint64_t *arith_wrap = new uint64_t[lnum_ops];
            aux->B2A(wrap_x, arith_wrap, lnum_ops, bw);
            int128_t signed_threshold = (bw == 0 ? 0 : (int128_t(1) << (bw - 1)));
            int128_t signed_modulus = (bw == 0 ? 0 : (int128_t(1) << bw));
            for (int i = 0; i < lnum_ops; i++) {
                __int128_t value = static_cast<__int128_t>(input[i]) % static_cast<__int128_t>(Q);
                __int128_t wrap = static_cast<__int128_t>(arith_wrap[i]);
                __int128_t corrected = value - static_cast<__int128_t>(Q) * wrap;
                corrected &= mask_bw;
                if (signed_arithmetic && bw > 0 && corrected >= signed_threshold) {
                    corrected -= signed_modulus;
                }
                result[i] = corrected;
            }
            delete[] wrap_x;
            delete[] arith_wrap;
        }

        void static mux_thread(AuxProtocols *aux, uint8_t* b, T* input, T* result, int32_t lnum_ops, int32_t bwA, int32_t bwB){
            aux->multiplexer<T>(b, input, result, lnum_ops, bwA, bwB);
        }

        // input b_fix-bit fixed-point value X_fix with scale 2^{s_fix}, output b_acc-bit integer X_q
        void static secure_round_thread(AuxProtocols *aux, T* input, T* result, 
                                        int lnum_ops, int32_t s_fix, int32_t bw_fix, 
                                        int32_t bw_acc, int party) {
            // Step 1: Compute b_1 = 1{X_fix >= 2^{s_fix-1}}
            // 这等价于比较 X_fix 与 2^{s_fix-1}
            // 通过计算 MSB(X_fix - 2^{s_fix-1})，如果MSB=1表示负数，即X_fix < 2^{s_fix-1}
            // 所以 b_1 = NOT(MSB(X_fix - 2^{s_fix-1}))
            uint64_t threshold = (1ULL << (s_fix - 1));
            uint8_t *b_1 = new uint8_t[lnum_ops];
            
            uint64_t *tmp_x = new uint64_t[lnum_ops];
            uint64_t mask_fix = (bw_fix == 64 ? -1ULL : ((1ULL << bw_fix) - 1));
            
            if (party == ALICE) {
                for (int i = 0; i < lnum_ops; i++) {
                    tmp_x[i] = (input[i] - threshold) & mask_fix;
                }
            } else { // BOB
                for (int i = 0; i < lnum_ops; i++) {
                    tmp_x[i] = input[i];
                }
            }
            
            // MSB(tmp_x) gives us 1{X_fix < 2^{s_fix-1}}
            // So b_1 = NOT(MSB(tmp_x))
            aux->MSB<uint64_t>(tmp_x, b_1, lnum_ops, bw_fix);
            
            // 取反得到 b_1
            for (int i = 0; i < lnum_ops; i++) {
                if (party == ALICE) {
                    b_1[i] = b_1[i] ^ 1;
                }
                // BOB的share不变
            }
            
            // Step 2: Compute b_2 via 1-out-of-4 OT
            uint8_t *b_2 = new uint8_t[lnum_ops];
            
            // 提取 m_0 和 m_1 (第 s_fix 位)
            uint8_t *m_local = new uint8_t[lnum_ops];
            for (int i = 0; i < lnum_ops; i++) {
                m_local[i] = (input[i] >> s_fix) & 1;
            }
            
            if (party == ALICE) {
                // ALICE 准备查找表（4个消息）
                PRG128 prg;
                prg.random_bool((bool *)b_2, lnum_ops);  // 生成随机 r 作为 b_2_1
                
                uint8_t **spec = new uint8_t*[lnum_ops];
                for (int i = 0; i < lnum_ops; i++) {
                    spec[i] = new uint8_t[4];
                    uint8_t b_1_1 = b_1[i];      // ALICE的b_1 share
                    uint8_t m_1 = m_local[i];    // ALICE的m bit
                    uint8_t r = b_2[i];          // 随机掩码，也是b_2_1
                    
                    // 对于BOB的4种可能选择 (b_1_0, m_0) = (j0, j1)
                    // j的低位是b_1_0，高位是m_0
                    for (int j = 0; j < 4; j++) {
                        uint8_t j0 = j & 1;        // b_1_0
                        uint8_t j1 = (j >> 1) & 1; // m_0
                        
                        // 根据公式 b_2 = ((1 ⊕ b_1) ∧ (m_0 ⊕ m_1)) ⊕ (m_0 ∧ m_1)
                        // 将 b_1 = b_1_0 ⊕ b_1_1, 代入并整理得到ALICE的消息
                        uint8_t term1 = ((1 ^ j0 ^ b_1_1) & (j1 ^ m_1));
                        uint8_t term2 = (j1 & m_1);
                        spec[i][j] = (term1 ^ term2 ^ r) & 1;
                    }
                }
                
                // 调用 lookup_table (1-out-of-4 OT)
                aux->lookup_table<uint8_t>(spec, nullptr, nullptr, lnum_ops, 2, 1);
                
                for (int i = 0; i < lnum_ops; i++) delete[] spec[i];
                delete[] spec;
                
            } else { // BOB
                // BOB 提供选择输入 (b_1_0, m_0)
                uint8_t *lut_in = new uint8_t[lnum_ops];
                for (int i = 0; i < lnum_ops; i++) {
                    uint8_t b_1_0 = b_1[i];      // BOB的b_1 share
                    uint8_t m_0 = m_local[i];    // BOB的m bit
                    lut_in[i] = (m_0 << 1) | b_1_0;  // 组合成2位选择
                }
                
                // 调用 lookup_table 接收结果
                aux->lookup_table<uint8_t>(nullptr, lut_in, b_2, lnum_ops, 2, 1);
                
                delete[] lut_in;
            }
            
            // Step 3: B2A conversion
            uint64_t *b_1_arith = new uint64_t[lnum_ops];
            uint64_t *b_2_arith = new uint64_t[lnum_ops];
            aux->B2A(b_1, b_1_arith, lnum_ops, bw_acc);
            aux->B2A(b_2, b_2_arith, lnum_ops, bw_acc);
            
            // Step 4: 本地计算 X_q = X_fix / 2^{s_fix} + b_1 + b_2
            uint64_t mask_acc = (bw_acc == 64 ? -1ULL : ((1ULL << bw_acc) - 1));
            for (int i = 0; i < lnum_ops; i++) {
                uint64_t shifted = input[i] >> s_fix;
                result[i] = (shifted + b_1_arith[i] + b_2_arith[i]) & mask_acc;
            }
            
            // 清理
            delete[] b_1;
            delete[] b_2;
            delete[] m_local;
            delete[] tmp_x;
            delete[] b_1_arith;
            delete[] b_2_arith;
        }

        // Secure requantization thread
        // Implements 3 algorithms from protocol.tex:
        // 1. From b_acc to b_acc with scale change (s to s')
        // 2. From b_acc to b_fix with scale s to scale 2^{s_fix}
        // 3. From b_fix to b_acc with scale change
        void static secure_requant_thread(AuxProtocols *aux, T* input, T* result,
                                          int lnum_ops, double scale_in, double scale_out,
                                          int32_t bw_in, int32_t bw_out, int32_t s_fix, int party) {

            const bool signed_arithmetic = std::is_signed_v<T>;
            auto z_or_s_extend = [&](int32_t from_bw, int32_t to_bw) {
                if constexpr (std::is_same_v<T, int128_t> || std::is_same_v<T, __int128>) {
                    if (signed_arithmetic) {
                        aux->s_extend(lnum_ops, reinterpret_cast<int128_t*>(input), reinterpret_cast<int128_t*>(result), from_bw, to_bw, nullptr);
                    } else {
                        aux->z_extend(lnum_ops, reinterpret_cast<int128_t*>(input), reinterpret_cast<int128_t*>(result), from_bw, to_bw, nullptr);
                    }
                } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
                    auto *in_u64 = reinterpret_cast<uint64_t*>(input);
                    auto *out_u64 = reinterpret_cast<uint64_t*>(result);
                    if (signed_arithmetic) {
                        aux->s_extend(lnum_ops, in_u64, out_u64, from_bw, to_bw, nullptr);
                    } else {
                        aux->z_extend(lnum_ops, in_u64, out_u64, from_bw, to_bw, nullptr);
                    }
                } else {
                    static_assert(std::is_same_v<T, int128_t> || std::is_same_v<T, __int128> || sizeof(T) == sizeof(uint64_t),
                                  "secure_requant only supports 64-bit or 128-bit integer tensor types");
                }
            };

            // Determine the type of requantization based on bitwidths
            bool is_acc_to_fix = (bw_in < bw_out);  // Algorithm 2: b_acc -> b_fix
            bool is_fix_to_acc = (bw_in > bw_out);  // Algorithm 3: b_fix -> b_acc
            bool is_acc_to_acc = (bw_in == bw_out); // Algorithm 1: b_acc -> b_acc
            
            if (is_acc_to_acc) {
                // Algorithm 1: From b_acc to b_acc with scale change (s to s')
                // Step 1: Extend from b_acc to b_fix
                z_or_s_extend(bw_in, s_fix * 2);
                
                // Step 2: Locally compute X_fix = X_q * (s/s') * 2^{s_fix}
                double scale_ratio = scale_in / scale_out;
                int128_t scale_factor = static_cast<int128_t>(scale_ratio * (1ULL << s_fix));
                uint64_t mask_fix = (s_fix * 2 == 64 ? static_cast<uint64_t>(-1) : ((1ULL << (s_fix * 2)) - 1));
                for (int i = 0; i < lnum_ops; i++) {
                    int128_t mul = static_cast<int128_t>(result[i]) * scale_factor;
                    result[i] = static_cast<T>(static_cast<uint64_t>(mul) & mask_fix);
                }
                
                // Step 3: Secure rounding to get X'_q
                secure_round_thread(aux, result, result, lnum_ops, s_fix, s_fix * 2, bw_out, party);
                
            } else if (is_acc_to_fix) {
                // Algorithm 2: From b_acc to b_fix with scale s to scale 2^{s_fix}
                // Step 1: Extend from b_acc to b_fix
                z_or_s_extend(bw_in, bw_out);
                
                // Step 2: Locally compute X_f = X_q * s * 2^{s_fix}
                int128_t scale_factor = static_cast<int128_t>(scale_in * (1ULL << s_fix));
                uint64_t mask_out = (bw_out == 64 ? static_cast<uint64_t>(-1) : ((1ULL << bw_out) - 1));
                for (int i = 0; i < lnum_ops; i++) {
                    int128_t mul = static_cast<int128_t>(result[i]) * scale_factor;
                    result[i] = static_cast<T>(static_cast<uint64_t>(mul) & mask_out);
                }
                
            } else if (is_fix_to_acc) {
                // Algorithm 3: From b_fix to b_acc with scale change
                // Step 1: Extend from b_fix to 2*b_fix
                z_or_s_extend(bw_in, 2 * bw_in);
                
                // Step 2: Locally compute X_f * (1/s') * 2^{s_fix}
                int128_t scale_factor = static_cast<int128_t>((1ULL << s_fix) / scale_out);
                uint64_t mask_inter = (2 * bw_in == 64 ? static_cast<uint64_t>(-1) : ((1ULL << (2 * bw_in)) - 1));
                for (int i = 0; i < lnum_ops; i++) {
                    int128_t mul = static_cast<int128_t>(result[i]) * scale_factor;
                    result[i] = static_cast<T>(static_cast<uint64_t>(mul) & mask_inter);
                }
                
                // Step 3: Secure rounding to get X'_q
                secure_round_thread(aux, result, result, lnum_ops, s_fix, 2 * bw_in, bw_out, party);
            }
        }
};

}
