#include "evaluate.cuh"
#include "ntt.cuh"
#include "rns.cuh"
#include "rns_bconv.cuh"
#include "uintmodmath.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

namespace phantom {

    __global__ void key_switch_inner_prod_c2_and_evk(uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
                                                     const DModulus *modulus, size_t n, size_t size_QP,
                                                     size_t size_QP_n,
                                                     size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
                                                     size_t beta, size_t reduction_threshold) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size_QlP_n; tid += blockDim.x * gridDim.x) {
            size_t cnt = reduction_threshold; // FIX BUG
            size_t nid = tid / n;
            size_t twr = (nid >= size_Ql ? size_Q + (nid - size_Ql) : nid);
            // base_rns = {q0, q1, ..., qj, p}
            DModulus mod = modulus[twr];
            uint64_t evk_id = (tid % n) + twr * n;
            uint64_t c2_id = (tid % n) + nid * n;

            uint128_t prod0, prod1;
            uint128_t acc0, acc1;

            // ct^x = ( <RNS-Decomp(c*_2), evk_b> , <RNS-Decomp(c*_2), evk_a>
            // evk[key_index][rns]
            //
            // RNS-Decomp(c*_2)[key_index + rns_indx * twr] =
            //           ( {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod q0,
            //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod q1,
            //             ...
            //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod qj,
            //             {c*_2 mod q0, c*_2 mod q1, ..., c*_2 mod qj} mod p, )
            //
            // decomp_mod_size = number of evks

            // evk[0]_a
            acc0 = multiply_uint64_uint64_fp64(c2[c2_id], evks[0][evk_id]);
            // evk[0]_b
            acc1 = multiply_uint64_uint64_fp64(c2[c2_id], evks[0][evk_id + size_QP_n]);

            for (uint64_t i = 1; i < beta; i++) {
                if (i && cnt == 0 /* FIX BUG */) {
#ifdef FP64_MM_ARITH
                    acc0 = adjust_accum_int64_to_fp64(acc0);
                    // acc0 = adjust_accum_fp64_to_int64(acc0);
                    acc0.lo = barrett_reduce_uint128_uint64_fp64(acc0, mod.value(), mod.const_ratio_fp64());
                    acc0.hi = 0;

                    acc1 = adjust_accum_int64_to_fp64(acc1);
                    // acc1 = adjust_accum_fp64_to_int64(acc1);
                    acc1.lo = barrett_reduce_uint128_uint64_fp64(acc1, mod.value(), mod.const_ratio_fp64());
                    acc1.hi = 0;
#else
                    acc0.lo = barrett_reduce_uint128_uint64_fp64(acc0, mod.value(), mod.const_ratio());
                    acc0.hi = 0;

                    acc1.lo = barrett_reduce_uint128_uint64_fp64(acc1, mod.value(), mod.const_ratio());
                    acc1.hi = 0;
#endif
                    
                    cnt = reduction_threshold; // FIX BUG
                }

                prod0 = multiply_uint64_uint64_fp64(c2[c2_id + i * size_QlP_n], evks[i][evk_id]);
                add_uint128_uint128(acc0, prod0, acc0);

                prod1 = multiply_uint64_uint64_fp64(c2[c2_id + i * size_QlP_n], evks[i][evk_id + size_QP_n]);
                add_uint128_uint128(acc1, prod1, acc1);

                cnt --; // FIX BUG: reduction_threshold should be reduced by 1
            }

#ifdef FP64_MM_ARITH
            acc0 = adjust_accum_int64_to_fp64(acc0);
            // acc0 = adjust_accum_fp64_to_int64(acc0);
            uint64_t res0 = barrett_reduce_uint128_uint64_fp64(acc0, mod.value(), mod.const_ratio_fp64());
            dst[tid] = res0;

            acc1 = adjust_accum_int64_to_fp64(acc1);
            // acc1 = adjust_accum_fp64_to_int64(acc1);
            uint64_t res1 = barrett_reduce_uint128_uint64_fp64(acc1, mod.value(), mod.const_ratio_fp64());
            dst[tid + size_QlP_n] = res1;
#else
            uint64_t res0 = barrett_reduce_uint128_uint64_fp64(acc0, mod.value(), mod.const_ratio());
            dst[tid] = res0;

            uint64_t res1 = barrett_reduce_uint128_uint64_fp64(acc1, mod.value(), mod.const_ratio());
            dst[tid + size_QlP_n] = res1;
#endif
        }
    }

    __global__ void key_switch_inner_prod_c2_and_evk_batch(
        uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
        const DModulus *modulus, size_t n, size_t size_QP, size_t size_QP_n,
        size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
        size_t beta, size_t reduction_threshold, size_t c2_batch_stride,
        size_t dst_batch_stride) {
        const size_t batch_idx = blockIdx.y;
        const uint64_t *c2_batch = c2 + batch_idx * c2_batch_stride;
        uint64_t *dst_batch = dst + batch_idx * dst_batch_stride;

        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size_QlP_n; tid += blockDim.x * gridDim.x) {
            size_t cnt = reduction_threshold;
            size_t nid = tid / n;
            size_t twr = (nid >= size_Ql ? size_Q + (nid - size_Ql) : nid);
            DModulus mod = modulus[twr];
            uint64_t evk_id = (tid % n) + twr * n;
            uint64_t c2_id = (tid % n) + nid * n;

            uint128_t prod0, prod1;
            uint128_t acc0, acc1;

            acc0 = multiply_uint64_uint64_fp64(c2_batch[c2_id], evks[0][evk_id]);
            acc1 = multiply_uint64_uint64_fp64(c2_batch[c2_id], evks[0][evk_id + size_QP_n]);

            for (uint64_t i = 1; i < beta; i++) {
                if (i && cnt == 0) {
#ifdef FP64_MM_ARITH
                    acc0 = adjust_accum_int64_to_fp64(acc0);
                    acc0.lo = barrett_reduce_uint128_uint64_fp64(acc0, mod.value(), mod.const_ratio_fp64());
                    acc0.hi = 0;

                    acc1 = adjust_accum_int64_to_fp64(acc1);
                    acc1.lo = barrett_reduce_uint128_uint64_fp64(acc1, mod.value(), mod.const_ratio_fp64());
                    acc1.hi = 0;
#else
                    acc0.lo = barrett_reduce_uint128_uint64_fp64(acc0, mod.value(), mod.const_ratio());
                    acc0.hi = 0;

                    acc1.lo = barrett_reduce_uint128_uint64_fp64(acc1, mod.value(), mod.const_ratio());
                    acc1.hi = 0;
#endif
                    cnt = reduction_threshold;
                }

                prod0 = multiply_uint64_uint64_fp64(c2_batch[c2_id + i * size_QlP_n], evks[i][evk_id]);
                add_uint128_uint128(acc0, prod0, acc0);

                prod1 = multiply_uint64_uint64_fp64(c2_batch[c2_id + i * size_QlP_n], evks[i][evk_id + size_QP_n]);
                add_uint128_uint128(acc1, prod1, acc1);

                cnt--;
            }

#ifdef FP64_MM_ARITH
            acc0 = adjust_accum_int64_to_fp64(acc0);
            uint64_t res0 = barrett_reduce_uint128_uint64_fp64(acc0, mod.value(), mod.const_ratio_fp64());
            dst_batch[tid] = res0;

            acc1 = adjust_accum_int64_to_fp64(acc1);
            uint64_t res1 = barrett_reduce_uint128_uint64_fp64(acc1, mod.value(), mod.const_ratio_fp64());
            dst_batch[tid + size_QlP_n] = res1;
#else
            uint64_t res0 = barrett_reduce_uint128_uint64_fp64(acc0, mod.value(), mod.const_ratio());
            dst_batch[tid] = res0;

            uint64_t res1 = barrett_reduce_uint128_uint64_fp64(acc1, mod.value(), mod.const_ratio());
            dst_batch[tid + size_QlP_n] = res1;
#endif
        }
    }

    void
    key_switch_inner_prod(uint64_t *p_cx, const uint64_t *p_t_mod_up, const uint64_t *const *rlk,
                          const DRNSTool &rns_tool,
                          const DModulus *modulus_QP, size_t reduction_threshold, const cudaStream_t &stream) {

        const size_t size_QP = rns_tool.size_QP();
        const size_t size_P = rns_tool.size_P();
        const size_t size_Q = size_QP - size_P;

        const size_t size_Ql = rns_tool.base_Ql().size();
        const size_t size_QlP = size_Ql + size_P;

        const size_t n = rns_tool.n();
        const auto size_QP_n = size_QP * n;
        const auto size_QlP_n = size_QlP * n;

        const size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

        key_switch_inner_prod_c2_and_evk<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(
                p_cx, p_t_mod_up, rlk, modulus_QP, n, size_QP, size_QP_n, size_QlP, size_QlP_n, size_Q, size_Ql, beta,
                reduction_threshold);
    }

    // Add batched key-switch outputs back into ciphertext (c0/c1).
    // Grid mapping:
    //   blockIdx.y -> ciphertext index in batch
    //   blockIdx.z -> output component (0 for c0, 1 for c1)
    //   blockIdx.x/threadIdx.x -> coefficient index across one component
    __global__ static void add_to_ct_kernel_pair_batch(
        uint64_t *ct, const uint64_t *cx, const DModulus *modulus, size_t n,
        size_t dst_size_limb, size_t src_size_limb, size_t ct_poly_count,
        size_t batch_size) {
        const size_t batch_idx = blockIdx.y;
        const size_t component_idx = blockIdx.z;
        if (batch_idx >= batch_size || component_idx >= 2) {
            return;
        }

        const size_t src_batch_idx = batch_idx * 2 + component_idx;
        const size_t src_batch_offset = src_batch_idx * src_size_limb * n;
        const size_t dst_batch_offset =
            (batch_idx * ct_poly_count + component_idx) * dst_size_limb * n;

        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n * dst_size_limb;
             tid += blockDim.x * gridDim.x) {
            size_t twr = tid / n;
            DModulus mod = modulus[twr];
            ct[dst_batch_offset + tid] = add_uint64_uint64_mod(
                ct[dst_batch_offset + tid], cx[src_batch_offset + tid], mod.value());
        }
    }

// cks refers to cipher to be key-switched
    template <bool batch>
    void keyswitch_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, uint64_t *c2,
                           const PhantomRelinKey &relin_keys, bool is_relin, const cudaStream_t &stream) {
        const auto &s = stream;

        // Extract encryption parameters.
        auto &key_context_data = context.get_context_data(0);
        auto &key_parms = key_context_data.parms();
        auto scheme = key_parms.scheme();
        auto n = key_parms.poly_modulus_degree();
        auto mul_tech = key_parms.mul_tech();
        auto &key_modulus = key_parms.coeff_modulus();
        size_t size_P = key_parms.special_modulus_size();
        size_t size_QP = key_modulus.size();

        // HPS and HPSOverQ does not drop modulus
        uint32_t levelsDropped;

        if (scheme == scheme_type::bfv) {
            levelsDropped = 0;
            if (mul_tech == mul_tech_type::hps_overq_leveled) {
                size_t depth = encrypted.GetNoiseScaleDeg();
                bool isKeySwitch = !is_relin;
                bool is_Asymmetric = encrypted.is_asymmetric();
                size_t levels = depth - 1;
                auto dcrtBits = static_cast<double>(context.get_context_data(1).gpu_rns_tool().qMSB());

                // how many levels to drop
                levelsDropped = FindLevelsToDrop(context, levels, dcrtBits, isKeySwitch, is_Asymmetric);
            }
        } else if (scheme == scheme_type::bgv || scheme == scheme_type::ckks) {
            levelsDropped = encrypted.chain_index() - 1;
        } else {
            throw invalid_argument("unsupported scheme in keyswitch_inplace");
        }

        auto &rns_tool = context.get_context_data(1 + levelsDropped).gpu_rns_tool();

        auto modulus_QP = context.gpu_rns_tables().modulus();

        size_t size_Ql = rns_tool.base_Ql().size();
        size_t size_Q = size_QP - size_P;
        size_t size_QlP = size_Ql + size_P;

        auto size_Ql_n = size_Ql * n;
        // auto size_QP_n = size_QP * n;
        auto size_QlP_n = size_QlP * n;

        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
            auto t_cks = phantom::util::make_cuda_auto_ptr<uint64_t>(size_Q * n, s);
            cudaMemcpyAsync(t_cks.get(), c2, size_Q * n * sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, s);
            rns_tool.scaleAndRound_HPS_Q_Ql(c2, t_cks.get(), s);
        }

        // mod up
        size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();
        auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);
        rns_tool.modup(t_mod_up.get(), c2, context.gpu_rns_tables(), scheme, s);

        // key switch
        auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);
        auto reduction_threshold =
                (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;
        key_switch_inner_prod(cx.get(), t_mod_up.get(), relin_keys.public_keys_ptr(), rns_tool, modulus_QP,
                              reduction_threshold, s);

        // mod down
        if constexpr (batch) {
            rns_tool.moddown_from_NTT_batch(cx.get(), cx.get(), context.gpu_rns_tables(), 2, scheme, s);
        } else {
            for (size_t i = 0; i < 2; i++) {
                auto cx_i = cx.get() + i * size_QlP_n;
                rns_tool.moddown_from_NTT(cx_i, cx_i, context.gpu_rns_tables(), scheme, s);
            }
        }

        if constexpr (batch) {
            if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
                auto t_cx = make_cuda_auto_ptr<uint64_t>(2 * size_Q * n, s);
                rns_tool.ExpandCRTBasis_Ql_Q_batch(t_cx.get(), cx.get(), 2, s);
                add_to_ct_kernel_batch<<<dim3((size_Q * n) / blockDimGlb.x, 2), blockDimGlb, 0, s>>>(
                        encrypted.data(), t_cx.get(), rns_tool.base_Q().base(), n, size_Q, size_Q);
            } else {
                add_to_ct_kernel_batch<<<dim3(size_Ql_n / blockDimGlb.x, 2), blockDimGlb, 0, s>>>(
                        encrypted.data(), cx.get(), rns_tool.base_Ql().base(), n, size_Ql, size_QlP);
            }
        } else {
            for (size_t i = 0; i < 2; i++) {
                auto cx_i = cx.get() + i * size_QlP_n;

                if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
                    auto ct_i = encrypted.data() + i * size_Q * n;
                    auto t_cx = make_cuda_auto_ptr<uint64_t>(size_Q * n, s);
                    rns_tool.ExpandCRTBasis_Ql_Q(t_cx.get(), cx_i, s);
                    add_to_ct_kernel<<<(size_Q * n) / blockDimGlb.x, blockDimGlb, 0, s>>>(
                            ct_i, t_cx.get(), rns_tool.base_Q().base(), n, size_Q);
                } else {
                    auto ct_i = encrypted.data() + i * size_Ql_n;
                    add_to_ct_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
                            ct_i, cx_i, rns_tool.base_Ql().base(), n, size_Ql);
                }
            }
        }
    }

    template void keyswitch_inplace<false>(
        const PhantomContext &context, PhantomCiphertext &encrypted, uint64_t *c2,
        const PhantomRelinKey &relin_keys, bool is_relin, const cudaStream_t &stream);
    template void keyswitch_inplace<true>(
        const PhantomContext &context, PhantomCiphertext &encrypted, uint64_t *c2,
        const PhantomRelinKey &relin_keys, bool is_relin, const cudaStream_t &stream);

    // Batched key-switch / relinearization fusion flow for ciphertext size=3:
    //   1) Gather c2 from each ciphertext in the batch (optionally level-adjust first).
    //   2) Run modup_batch to lift c2 from Ql to QlP decomposition basis.
    //   3) Run one batched inner-product kernel against evaluation keys to produce
    //      two switched components per ciphertext.
    //   4) Run batched moddown from QlP to Ql.
    //   5) Add switched components back to ciphertext c0/c1 in place.
    // Caller then truncates ciphertext from size 3 to size 2.
    void keyswitch_inplace(const PhantomContext &context, PhantomBatchCiphertext &encrypted,
                           const PhantomRelinKey &relin_keys, bool is_relin,
                           const cudaStream_t &stream) {
        const auto &s = stream;
        const size_t batch_size = encrypted.batch_size();
        if (batch_size == 0) {
            return;
        }
        if (encrypted.size() != 3) {
            throw invalid_argument("keyswitch_inplace(batch): ciphertext size must be 3");
        }

        auto &key_context_data = context.get_context_data(0);
        auto &key_parms = key_context_data.parms();
        auto scheme = key_parms.scheme();
        auto n = key_parms.poly_modulus_degree();
        auto mul_tech = key_parms.mul_tech();
        auto &key_modulus = key_parms.coeff_modulus();
        size_t size_P = key_parms.special_modulus_size();
        size_t size_QP = key_modulus.size();

        uint32_t levelsDropped;
        if (scheme == scheme_type::bfv) {
            levelsDropped = 0;
            if (mul_tech == mul_tech_type::hps_overq_leveled) {
                size_t depth = encrypted.noiseScaleDeg();
                bool isKeySwitch = !is_relin;
                bool is_Asymmetric = encrypted.is_asymmetric();
                size_t levels = depth - 1;
                auto dcrtBits =
                    static_cast<double>(context.get_context_data(1).gpu_rns_tool().qMSB());
                levelsDropped =
                    FindLevelsToDrop(context, levels, dcrtBits, isKeySwitch, is_Asymmetric);
            }
        } else if (scheme == scheme_type::bgv || scheme == scheme_type::ckks) {
            levelsDropped = encrypted.chain_index() - 1;
        } else {
            throw invalid_argument("unsupported scheme in keyswitch_inplace(batch)");
        }

        auto &rns_tool = context.get_context_data(1 + levelsDropped).gpu_rns_tool();
        auto modulus_QP = context.gpu_rns_tables().modulus();

        const size_t size_Ql = rns_tool.base_Ql().size();
        const size_t size_Q = size_QP - size_P;
        const size_t size_QlP = size_Ql + size_P;

        const size_t size_Ql_n = size_Ql * n;
        const size_t size_Q_n = size_Q * n;
        const size_t size_QlP_n = size_QlP * n;
        const size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

        const size_t item_words = encrypted.item_data_count();
        const size_t coeff_words = encrypted.coeff_count();
        const size_t c2_offset = 2 * coeff_words;

        // Step 1: collect c2 from each batch item into contiguous [batch, Ql, n].
        auto c2_ql = make_cuda_auto_ptr<uint64_t>(batch_size * size_Ql_n, s);
        for (size_t b = 0; b < batch_size; ++b) {
            const uint64_t *c2_src = encrypted.data() + b * item_words + c2_offset;
            uint64_t *c2_dst = c2_ql.get() + b * size_Ql_n;
            if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
                rns_tool.scaleAndRound_HPS_Q_Ql(c2_dst, c2_src, s);
            } else {
                cudaMemcpyAsync(c2_dst, c2_src, size_Ql_n * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, s);
            }
        }

        // Step 2: batched ModUp: Ql -> QlP decomposition blocks.
        auto t_mod_up =
            make_cuda_auto_ptr<uint64_t>(batch_size * beta * size_QlP_n, s);
        rns_tool.modup_batch(
            t_mod_up.get(), c2_ql.get(), context.gpu_rns_tables(),
            batch_size, scheme, s);

        // Step 3: fused batched inner-product with relin key across all ciphertexts.
        auto cx = make_cuda_auto_ptr<uint64_t>(batch_size * 2 * size_QlP_n, s);
        auto reduction_threshold =
            (1 << (bits_per_uint64 -
                   static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) -
            1;
        dim3 gridDimGlb(size_QlP_n / blockDimGlb.x, batch_size);
        key_switch_inner_prod_c2_and_evk_batch<<<gridDimGlb, blockDimGlb, 0, s>>>(
            cx.get(), t_mod_up.get(), relin_keys.public_keys_ptr(), modulus_QP, n,
            size_QP, size_QP * n, size_QlP, size_QlP_n, size_Q, size_Ql, beta,
            reduction_threshold, beta * size_QlP_n, 2 * size_QlP_n);

        // Step 4: batched ModDown: QlP -> Ql (2 outputs per ciphertext).
        rns_tool.moddown_from_NTT_batch(cx.get(), cx.get(), context.gpu_rns_tables(),
                                        2 * batch_size, scheme, s);

        // Step 5: add switched outputs into c0/c1 in place.
        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
            auto t_cx = make_cuda_auto_ptr<uint64_t>(batch_size * 2 * size_Q_n, s);
            rns_tool.ExpandCRTBasis_Ql_Q_batch(t_cx.get(), cx.get(), 2 * batch_size, s);
            dim3 addGrid(size_Q_n / blockDimGlb.x, batch_size, 2);
            add_to_ct_kernel_pair_batch<<<addGrid, blockDimGlb, 0, s>>>(
                encrypted.data(), t_cx.get(), rns_tool.base_Q().base(), n, size_Q,
                size_Q, encrypted.size(), batch_size);
        } else {
            dim3 addGrid(size_Ql_n / blockDimGlb.x, batch_size, 2);
            add_to_ct_kernel_pair_batch<<<addGrid, blockDimGlb, 0, s>>>(
                encrypted.data(), cx.get(), rns_tool.base_Ql().base(), n, size_Ql,
                size_QlP, encrypted.size(), batch_size);
        }
    }
}
