#pragma once

#include <cmath>
#include <vector>

#include "ciphertext.h"
#include "context.cuh"
#include "ntt.cuh"
#include "plaintext.h"
#include "secretkey.h"
#include "cuda_wrapper.cuh"
#include "batch_view.h"

namespace phantom {

    size_t
    FindLevelsToDrop(const PhantomContext &context, size_t multiplicativeDepth, double dcrtBits, bool isKeySwitch,
                     bool isAsymmetric);

    __global__ void key_switch_inner_prod_c2_and_evk(uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
                                                     const DModulus *modulus, size_t n, size_t size_QP,
                                                     size_t size_QP_n,
                                                     size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
                                                     size_t beta, size_t reduction_threshold);

    // used by keyswitch_inplace
    void key_switch_inner_prod(uint64_t *p_cx, const uint64_t *p_t_mod_up, const uint64_t *const *rlk,
                               const phantom::DRNSTool &rns_tool, const DModulus *modulus_QP,
                               size_t reduction_threshold, const cudaStream_t &stream);

    template <bool batch = false>
    void keyswitch_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, uint64_t *c2,
                           const PhantomRelinKey &relin_keys,
                           bool is_relin, // false
                           const cudaStream_t &stream);

    void keyswitch_inplace(const PhantomContext &context, PhantomBatchCiphertext &encrypted,
                           const PhantomRelinKey &relin_keys, bool is_relin,
                           const cudaStream_t &stream);

/***************************************************** Core APIs ******************************************************/

    // encrypted = -encrypted
    void negate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted);

    inline auto negate(const PhantomContext &context, const PhantomCiphertext &encrypted) {
        PhantomCiphertext destination = encrypted;
        negate_inplace(context, destination);
        return destination;
    }

    // encrypted1 += encrypted2
    void add_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);

    inline auto
    add(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) {
        PhantomCiphertext destination = encrypted1;
        add_inplace(context, destination, encrypted2);
        return destination;
    }

    // destination = encrypteds[0] + encrypteds[1] + ...
    void add_many(const PhantomContext &context, const std::vector<PhantomCiphertext> &encrypteds,
                  PhantomCiphertext &destination);

    // if negate = false (default): encrypted1 -= encrypted2
    // if negate = true: encrypted1 = encrypted2 - encrypted1
    void sub_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                     const bool &negate = false);

    inline auto
    sub(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
        const bool &negate = false) {
        PhantomCiphertext destination = encrypted1;
        sub_inplace(context, destination, encrypted2, negate);
        return destination;
    }

    // encrypted += plain
    void add_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

    inline auto
    add_plain(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomPlaintext &plain) {
        PhantomCiphertext destination = encrypted;
        add_plain_inplace(context, destination, plain);
        return destination;
    }

    // encrypted -= plain
    void sub_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

    inline auto
    sub_plain(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomPlaintext &plain) {
        PhantomCiphertext destination = encrypted;
        sub_plain_inplace(context, destination, plain);
        return destination;
    }

    // encrypted *= plain
    void
    multiply_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

    inline auto multiply_plain(const PhantomContext &context, const PhantomCiphertext &encrypted,
                               const PhantomPlaintext &plain) {
        PhantomCiphertext destination = encrypted;
        multiply_plain_inplace(context, destination, plain);
        return destination;
    }

    void
    multiply_plain_ntt_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

    inline auto multiply_plain_ntt(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                   const PhantomPlaintext &plain) {
        PhantomCiphertext destination = encrypted;
        multiply_plain_ntt_inplace(context, destination, plain);
        return destination;
    }

    // Batch version: each ciphertext in `encrypteds` is multiplied with the same plaintext.
    void multiply_plain_ntt_inplace(
        const PhantomContext &context, BatchCipherView encrypteds, const PhantomPlaintext &plain);

    inline void multiply_plain_ntt_inplace(
        const PhantomContext &context, PhantomBatchCiphertext &encrypteds, const PhantomPlaintext &plain) {
        auto view = encrypteds.view();
        multiply_plain_ntt_inplace(context, view, plain);
        encrypteds.set_scale(encrypteds.scale() * plain.scale());
        encrypteds.set_ntt_form(true);
    }

    // destination += encrypted * plain, where encrypted/plain/destination are expected to be in NTT form.
    void multiply_plain_ntt_and_add_inplace(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                            const PhantomPlaintext &plain, PhantomCiphertext &destination);

    // Batch version: destination[i] += encrypteds[i] * plain.
    void multiply_plain_ntt_and_add_inplace(
        const PhantomContext &context, ConstBatchCipherView encrypteds, const PhantomPlaintext &plain,
        BatchCipherView destination);

    inline void multiply_plain_ntt_and_add_inplace(
        const PhantomContext &context, const PhantomBatchCiphertext &encrypteds, const PhantomPlaintext &plain,
        PhantomBatchCiphertext &destination) {
        multiply_plain_ntt_and_add_inplace(context, encrypteds.view(), plain, destination.view());
    }

    // Generic helper that falls back to multiply_plain + add when NTT fused path is unavailable.
    void multiply_plain_and_add_inplace(const PhantomContext &context, const PhantomCiphertext &encrypted,
                                        const PhantomPlaintext &plain, PhantomCiphertext &destination);

    // destination = sum_i (encrypteds[i] * plains[i]), all in NTT form.
    // This fuses the reduction over i into one kernel per ciphertext component.
    void multiply_plain_ntt_many(
        const PhantomContext &context, const std::vector<PhantomCiphertext> &encrypteds,
        const std::vector<PhantomPlaintext> &plains, PhantomCiphertext &destination);

    // Pointer-view variant to avoid temporary plaintext object copies.
    void multiply_plain_ntt_many_ptrs(
        const PhantomContext &context, const std::vector<PhantomCiphertext> &encrypteds,
        const std::vector<const PhantomPlaintext *> &plains, PhantomCiphertext &destination);

    // Fused 2D broadcast MAC for single-query PIR in NTT domain:
    // ans[i] += sum_{j=0}^{col_count-1} (pt[i][j] * ct[j]), where ct has 2 polys.
    // All buffers are contiguous and described by base pointers + strides.
    // Strides are in uint64_t elements; pass 0 to use dense defaults.
    __host__ void launch_multiply_add_2d_fusion(
        const PhantomContext &context, const uint64_t *pt_matrix, const uint64_t *ct_terms, uint64_t *ans,
        std::size_t row_count, std::size_t col_count, std::size_t chain_index,
        std::size_t pt_row_stride = 0, std::size_t pt_col_stride = 0,
        std::size_t ct_stride = 0, std::size_t ans_stride = 0,
        const cudaStream_t &stream = cudaStreamPerThread);

    // BSGS-style 2D fused MAC over ct-pt terms:
    // giant_outputs[g] = sum_j baby_ctxts[j] * plains[g * baby_step_size + j]
    // for valid diagonal indices only.
    inline void multiply_plain_ntt_bsgs_mac_ptrs(
        const PhantomContext &context, const std::vector<PhantomCiphertext> &baby_ctxts,
        const std::vector<const PhantomPlaintext *> &plains, std::size_t baby_step_size,
        std::size_t total_diagonals, std::vector<PhantomCiphertext> &giant_outputs)
    {
        if (baby_ctxts.empty()) {
            throw std::invalid_argument("multiply_plain_ntt_bsgs_mac_ptrs: baby_ctxts is empty");
        }
        if (baby_step_size == 0) {
            throw std::invalid_argument("multiply_plain_ntt_bsgs_mac_ptrs: baby_step_size must be positive");
        }
        if (baby_ctxts.size() != baby_step_size) {
            throw std::invalid_argument(
                "multiply_plain_ntt_bsgs_mac_ptrs: baby_ctxts.size() must equal baby_step_size");
        }
        if (plains.size() != total_diagonals) {
            throw std::invalid_argument(
                "multiply_plain_ntt_bsgs_mac_ptrs: plains.size() must equal total_diagonals");
        }
        if (total_diagonals == 0) {
            giant_outputs.clear();
            return;
        }

        const std::size_t giant_step_count =
            (total_diagonals + baby_step_size - 1) / baby_step_size;
        giant_outputs.resize(giant_step_count);

        std::vector<const PhantomPlaintext *> full_plain_terms(baby_step_size, nullptr);
        for (std::size_t giant_idx = 0; giant_idx < giant_step_count; ++giant_idx) {
            const std::size_t diagonal_base = giant_idx * baby_step_size;
            const std::size_t term_count =
                std::min(baby_step_size, total_diagonals - diagonal_base);

            if (term_count == baby_step_size) {
                for (std::size_t t = 0; t < term_count; ++t) {
                    if (plains[diagonal_base + t] == nullptr) {
                        throw std::invalid_argument(
                            "multiply_plain_ntt_bsgs_mac_ptrs: null plaintext pointer");
                    }
                    full_plain_terms[t] = plains[diagonal_base + t];
                }
                multiply_plain_ntt_many_ptrs(
                    context, baby_ctxts, full_plain_terms, giant_outputs[giant_idx]);
                continue;
            }

            if (plains[diagonal_base] == nullptr) {
                throw std::invalid_argument(
                    "multiply_plain_ntt_bsgs_mac_ptrs: null plaintext pointer");
            }
            PhantomCiphertext accum =
                multiply_plain_ntt(context, baby_ctxts[0], *plains[diagonal_base]);
            for (std::size_t t = 1; t < term_count; ++t) {
                if (plains[diagonal_base + t] == nullptr) {
                    throw std::invalid_argument(
                        "multiply_plain_ntt_bsgs_mac_ptrs: null plaintext pointer");
                }
                multiply_plain_ntt_and_add_inplace(
                    context, baby_ctxts[t], *plains[diagonal_base + t], accum);
            }
            giant_outputs[giant_idx] = std::move(accum);
        }
    }

    inline void multiply_plain_ntt_bsgs_mac(
        const PhantomContext &context, const std::vector<PhantomCiphertext> &baby_ctxts,
        const std::vector<PhantomPlaintext> &plains, std::size_t baby_step_size,
        std::size_t total_diagonals, std::vector<PhantomCiphertext> &giant_outputs)
    {
        std::vector<const PhantomPlaintext *> plain_ptrs(plains.size(), nullptr);
        for (std::size_t i = 0; i < plains.size(); ++i) {
            plain_ptrs[i] = &plains[i];
        }
        multiply_plain_ntt_bsgs_mac_ptrs(
            context, baby_ctxts, plain_ptrs, baby_step_size, total_diagonals,
            giant_outputs);
    }

    // encrypted1 *= encrypted2
    void
    multiply_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);

    // Batch ciphertext multiply: encrypted1[i] *= encrypted2[i].
    void multiply_inplace(
        const PhantomContext &context, PhantomBatchCiphertext &encrypted1,
        const PhantomBatchCiphertext &encrypted2);

    inline auto
    multiply(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) {
        PhantomCiphertext destination = encrypted1;
        multiply_inplace(context, destination, encrypted2);
        return destination;
    }

    inline auto
    multiply(
        const PhantomContext &context, const PhantomBatchCiphertext &encrypted1,
        const PhantomBatchCiphertext &encrypted2) {
        PhantomBatchCiphertext destination = encrypted1;
        multiply_inplace(context, destination, encrypted2);
        return destination;
    }

    // encrypted1 *= encrypted2
    void multiply_and_relin_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1,
                                    const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys);

    // Batch ciphertext fused multiply+relinearize: encrypted1[i] = relin(encrypted1[i] * encrypted2[i]).
    void multiply_and_relin_inplace(
        const PhantomContext &context, PhantomBatchCiphertext &encrypted1,
        const PhantomBatchCiphertext &encrypted2, const PhantomRelinKey &relin_keys);

    inline auto multiply_and_relin(const PhantomContext &context, const PhantomCiphertext &encrypted1,
                                   const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys) {
        PhantomCiphertext destination = encrypted1;
        multiply_and_relin_inplace(context, destination, encrypted2, relin_keys);
        return destination;
    }

    void relinearize_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                             const PhantomRelinKey &relin_keys);

    // Batch ciphertext relinearize: encrypted[i] = relin(encrypted[i]).
    void relinearize_inplace(
        const PhantomContext &context, PhantomBatchCiphertext &encrypted,
        const PhantomRelinKey &relin_keys);

    inline auto relinearize(const PhantomContext &context, const PhantomCiphertext &encrypted,
                            const PhantomRelinKey &relin_keys) {
        PhantomCiphertext destination = encrypted;
        relinearize_inplace(context, destination, relin_keys);
        return destination;
    }

    // ciphertext
    [[nodiscard]]
    PhantomCiphertext rescale_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted);

    // ciphertext
    inline void rescale_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted) {
        encrypted = rescale_to_next(context, encrypted);
    }

    // ciphertext
    [[nodiscard]]
    PhantomCiphertext mod_switch_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted);

    // ciphertext
    inline void mod_switch_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted) {
        encrypted = mod_switch_to_next(context, encrypted);
    }

    // ciphertext
    inline auto mod_switch_to(const PhantomContext &context, const PhantomCiphertext &encrypted, size_t chain_index) {
        if (encrypted.chain_index() > chain_index) {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        PhantomCiphertext destination = encrypted;

        while (destination.chain_index() != chain_index) {
            mod_switch_to_next_inplace(context, destination);
        }

        return destination;
    }

    // ciphertext
    inline void mod_switch_to_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t chain_index) {
        if (encrypted.chain_index() > chain_index) {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        while (encrypted.chain_index() != chain_index) {
            mod_switch_to_next_inplace(context, encrypted);
        }
    }

    // plaintext
    void mod_switch_to_next_inplace(const PhantomContext &context, PhantomPlaintext &plain);

    // plaintext
    inline auto mod_switch_to_next(const PhantomContext &context, const PhantomPlaintext &plain) {
        PhantomPlaintext destination = plain;
        mod_switch_to_next_inplace(context, destination);
        return destination;
    }

    // plaintext
    inline void mod_switch_to_inplace(const PhantomContext &context, PhantomPlaintext &plain, size_t chain_index) {
        if (plain.chain_index() > chain_index) {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        while (plain.chain_index() != chain_index) {
            mod_switch_to_next_inplace(context, plain);
        }
    }

    // plaintext
    inline auto mod_switch_to(const PhantomContext &context, const PhantomPlaintext &plain, size_t chain_index) {
        if (plain.chain_index() > chain_index) {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        PhantomPlaintext destination = plain;

        while (destination.chain_index() != chain_index) {
            mod_switch_to_next_inplace(context, destination);
        }

        return destination;
    }

    void apply_galois_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t galois_elt,
                              const PhantomGaloisKey &galois_keys);

    inline auto apply_galois(const PhantomContext &context, const PhantomCiphertext &encrypted, size_t galois_elt,
                             const PhantomGaloisKey &galois_keys) {
        PhantomCiphertext destination = encrypted;
        apply_galois_inplace(context, destination, galois_elt, galois_keys);
        return destination;
    }

    void rotate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, int step,
                        const PhantomGaloisKey &galois_key);

    inline auto rotate(const PhantomContext &context, const PhantomCiphertext &encrypted, int step,
                       const PhantomGaloisKey &galois_key) {
        PhantomCiphertext destination = encrypted;
        rotate_inplace(context, destination, step, galois_key);
        return destination;
    }

    void transform_to_ntt_inplace(const PhantomContext &context, PhantomPlaintext &plain, size_t chain_index);

    inline auto transform_to_ntt(const PhantomContext &context, const PhantomPlaintext &plain, size_t chain_index) {
        PhantomPlaintext destination = plain;
        transform_to_ntt_inplace(context, destination, chain_index);
        return destination;
    }

    void transform_to_ntt_inplace(const PhantomContext &context, PhantomCiphertext &encrypted);

    // Batch ciphertext NTT transform over all [batch * ciphertext_size] polys.
    void transform_to_ntt_inplace(const PhantomContext &context, PhantomBatchCiphertext &encrypteds);

    inline auto transform_to_ntt(const PhantomContext &context, const PhantomCiphertext &encrypted) {
        PhantomCiphertext destination = encrypted;
        transform_to_ntt_inplace(context, destination);
        return destination;
    }

    void transform_from_ntt_inplace(const PhantomContext &context, PhantomCiphertext &encrypted);

    // Batch ciphertext inverse NTT transform over all [batch * ciphertext_size] polys.
    void transform_from_ntt_inplace(const PhantomContext &context, PhantomBatchCiphertext &encrypteds);

    inline auto transform_from_ntt(const PhantomContext &context, const PhantomCiphertext &encrypted) {
        PhantomCiphertext destination = encrypted;
        transform_from_ntt_inplace(context, destination);
        return destination;
    }

/*************************************************** Advanced APIs ****************************************************/

    void hoisting_inplace(const PhantomContext &context, PhantomCiphertext &ct, const PhantomGaloisKey &glk,
                          const std::vector<int> &steps);

    inline auto hoisting(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomGaloisKey &glk,
                         const std::vector<int> &steps) {
        PhantomCiphertext destination = encrypted;
        hoisting_inplace(context, destination, glk, steps);
        return destination;
    }
}
