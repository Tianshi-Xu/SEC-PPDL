#pragma once
#include "uintmath.cuh"

__global__ void bconv_mult_kernel(uint64_t *dst, const uint64_t *src, const uint64_t *scale,
                                  const uint64_t *scale_shoup, const DModulus *base, uint64_t base_size, uint64_t n);

__global__ void bconv_mult_unroll2_kernel(uint64_t *dst, const uint64_t *src, const uint64_t *scale,
                                          const uint64_t *scale_shoup, const DModulus *base, uint64_t base_size,
                                          uint64_t n);

__global__ void bconv_mult_unroll4_kernel(uint64_t *dst, const uint64_t *src, const uint64_t *scale,
                                          const uint64_t *scale_shoup, const DModulus *base, uint64_t base_size,
                                          uint64_t n);

__global__ void bconv_matmul_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *QHatModp,
                                    const DModulus *ibase, uint64_t ibase_size, const DModulus *obase,
                                    uint64_t obase_size, uint64_t n);

__global__ void bconv_matmul_unroll2_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *QHatModp,
                                            const DModulus *ibase, uint64_t ibase_size, const DModulus *obase,
                                            uint64_t obase_size, uint64_t n);

__global__ void bconv_matmul_unroll4_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *QHatModp,
                                            const DModulus *ibase, uint64_t ibase_size, const DModulus *obase,
                                            uint64_t obase_size, uint64_t n);

__forceinline__ __device__ auto base_convert_acc(const uint64_t *ptr, const uint64_t *QHatModp,
                                                 size_t out_prime_idx, size_t degree, size_t ibase_size,
                                                 size_t degree_idx) {
    phantom::arith::uint128_t accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const uint64_t op2 = QHatModp[out_prime_idx * ibase_size + i];
        phantom::arith::uint128_t out;

        uint64_t op1 = ptr[i * degree + degree_idx];
        out = phantom::arith::multiply_uint64_uint64(op1, op2);
        add_uint128_uint128(out, accum, accum);
    }
    return accum;
}

__forceinline__ __device__ auto base_convert_acc_unroll2(const uint64_t *ptr, const uint64_t *QHatModp,
                                                         size_t out_prime_idx, size_t degree, size_t ibase_size,
                                                         size_t degree_idx) {
    phantom::arith::uint128_t2 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const uint64_t op2 = QHatModp[out_prime_idx * ibase_size + i];
        phantom::arith::uint128_t2 out{};

        uint64_t op1_x, op1_y;
        phantom::arith::ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        out.x = phantom::arith::multiply_uint64_uint64_fp64(op1_x, op2);
        add_uint128_uint128(out.x, accum.x, accum.x);
        out.y = phantom::arith::multiply_uint64_uint64_fp64(op1_y, op2);
        add_uint128_uint128(out.y, accum.y, accum.y);
    }
    #ifdef FP64_MM_ARITH
    // For barrett_reduce_uint128_uint64_fp64
    accum.x = adjust_accum_int64_to_fp64(accum.x);
    accum.y = adjust_accum_int64_to_fp64(accum.y);

    // For barrett_reduce_uint128_uint64
    // accum.x = adjust_accum_fp64_to_int64(accum.x);
    // accum.y = adjust_accum_fp64_to_int64(accum.y);
    #endif
    return accum;
}

__forceinline__ __device__ auto base_convert_acc_unroll4(const uint64_t *ptr, const uint64_t *QHatModp,
                                                         size_t out_prime_idx, size_t degree, size_t ibase_size,
                                                         size_t degree_idx) {
    phantom::arith::uint128_t4 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const uint64_t op2 = QHatModp[out_prime_idx * ibase_size + i];
        phantom::arith::uint128_t4 out{};

        uint64_t op1_x, op1_y;
        phantom::arith::ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        out.x = phantom::arith::multiply_uint64_uint64(op1_x, op2);
        add_uint128_uint128(out.x, accum.x, accum.x);
        out.y = phantom::arith::multiply_uint64_uint64(op1_y, op2);
        add_uint128_uint128(out.y, accum.y, accum.y);

        uint64_t op1_z, op1_w;
        phantom::arith::ld_two_uint64(op1_z, op1_w, ptr + i * degree + degree_idx + 2);
        out.z = phantom::arith::multiply_uint64_uint64(op1_z, op2);
        add_uint128_uint128(out.z, accum.z, accum.z);
        out.w = phantom::arith::multiply_uint64_uint64(op1_w, op2);
        add_uint128_uint128(out.w, accum.w, accum.w);
    }
    return accum;
}

__forceinline__ __device__ double_t base_convert_acc_frac(const uint64_t *ptr, const double *qiInv, size_t degree,
                                                          size_t ibase_size, size_t degree_idx) {
    double_t accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const double op2 = qiInv[i];

        const uint64_t op1 = ptr[i * degree + degree_idx];
        accum += static_cast<double>(op1) * op2;
    }
    return accum;
}

__forceinline__ __device__ auto base_convert_acc_frac_unroll2(const uint64_t *ptr, const double *qiInv,
                                                              size_t degree, size_t ibase_size,
                                                              size_t degree_idx) {
    phantom::arith::double_t2 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const double op2 = qiInv[i];

        uint64_t op1_x, op1_y;
        phantom::arith::ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        accum.x += static_cast<double>(op1_x) * op2;
        accum.y += static_cast<double>(op1_y) * op2;
    }
    return accum;
}

__forceinline__ __device__ auto base_convert_acc_frac_unroll4(const uint64_t *ptr, const double *qiInv,
                                                              size_t degree, size_t ibase_size,
                                                              size_t degree_idx) {
    phantom::arith::double_t4 accum{0};
    for (int i = 0; i < ibase_size; i++) {
        const double op2 = qiInv[i];

        uint64_t op1_x, op1_y;
        phantom::arith::ld_two_uint64(op1_x, op1_y, ptr + i * degree + degree_idx);
        accum.x += static_cast<double>(op1_x) * op2;
        accum.y += static_cast<double>(op1_y) * op2;

        uint64_t op1_z, op1_w;
        phantom::arith::ld_two_uint64(op1_z, op1_w, ptr + i * degree + degree_idx + 2);
        accum.z += static_cast<double>(op1_z) * op2;
        accum.w += static_cast<double>(op1_w) * op2;
    }
    return accum;
}

__global__ void add_to_ct_kernel(uint64_t *ct, const uint64_t *cx, const DModulus *modulus, size_t n, size_t size_Ql);

__global__ void add_to_ct_kernel_batch(uint64_t *ct, const uint64_t *cx, const DModulus *modulus, size_t n, size_t dst_size_limb, size_t src_size_limb);
