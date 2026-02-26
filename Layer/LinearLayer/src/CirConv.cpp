/**
 * CirConv2D: Block Circulant Convolution with Nested Encoding
 *
 * Combines CirLinearNest's block circulant structure with Conv2DNest's
 * convolution encoding.
 *
 * Within each circulant block (b channels):
 *   Encoding follows the CirEncode for convolutions (Theorem in CirConv.latex):
 *     x̂[i·HW + j·W + k] = X[i, j, k]
 *     ŵ[i·HW + offset - m·W - n] = W[i, 0, m, n]
 *   Cyclic NTT gives circulant structure across block channels.
 *
 * Across circulant blocks:
 *   Same BSGS anti-diagonal encoding as CirLinearNest.
 *
 * When block_size=1, this reduces to Conv2DNest (with CyclicNTT instead of
 * negacyclic NTT, but functionally equivalent since padding prevents aliasing).
 */

#include <LinearLayer/Conv.h>
#include <Utils/CyclicNTT.h>
#include <cassert>

using namespace seal;
using namespace HE;
using namespace HE::unified;

namespace LinearLayer {

// ======================== CirConv2D ========================

CirConv2D::CirConv2D(uint64_t in_feature_size, uint64_t stride, uint64_t padding,
                     uint64_t block_size, const Tensor<uint64_t>& weight,
                     const Tensor<uint64_t>& bias, HEEvaluator* HE)
    : Conv2D(in_feature_size, stride, padding, weight, bias, HE),
      block_size(block_size)
{
    compute_he_params(in_feature_size);
    if (HE->server) {
        weight_pt = PackWeight();
    }
}

CirConv2D::CirConv2D(uint64_t in_feature_size, uint64_t in_channels,
                     uint64_t out_channels, uint64_t kernel_size,
                     uint64_t stride, uint64_t block_size, HEEvaluator* HE)
    : Conv2D(in_feature_size, in_channels, out_channels, kernel_size, stride, HE),
      block_size(block_size)
{
    compute_he_params(in_feature_size);
    if (HE->server) {
        weight_pt = PackWeight();
    }
}

void CirConv2D::compute_he_params(uint64_t in_feature_size) {
    assert(in_channels % block_size == 0 && "in_channels must be divisible by block_size");
    assert(out_channels % block_size == 0 && "out_channels must be divisible by block_size");

    int tmp_size = in_feature_size + 2 * this->padding - 1;
    for (int i = 0; i < 5; i++) {
        tmp_size |= tmp_size >> (1 << i);
    }
    padded_feature_size = tmp_size + 1;

    uint64_t padded_HW = padded_feature_size * padded_feature_size;
    ntt_size = block_size * padded_HW;

    num_blocks_in = in_channels / block_size;
    num_blocks_out = out_channels / block_size;

    tile_size = HE->polyModulusDegree / (2 * ntt_size);
    if (tile_size < 1) tile_size = 1;

    tiled_in_channels = (num_blocks_in + tile_size - 1) / tile_size;
    tiled_out_channels = (num_blocks_out + tile_size - 1) / tile_size;

    if (tile_size <= 1) {
        input_rot = 1;
    } else {
        input_rot = 1;
        while (input_rot * input_rot < tile_size) {
            input_rot++;
        }
    }

    out_feature_size = (in_feature_size + 2 * this->padding - kernel_size) / stride + 1;

    std::cout << "CirConv2D params: in_channels=" << in_channels
              << ", out_channels=" << out_channels
              << ", block_size=" << block_size
              << ", padded_feature_size=" << padded_feature_size
              << ", ntt_size=" << ntt_size
              << ", num_blocks=(" << num_blocks_in << "," << num_blocks_out << ")"
              << ", tile_size=" << tile_size
              << ", input_rot=" << input_rot
              << ", tiled=(" << tiled_in_channels << "," << tiled_out_channels << ")"
              << ", out_feature_size=" << out_feature_size << std::endl;
}

Tensor<UnifiedPlaintext> CirConv2D::PackWeight() {
    Utils::CyclicNTT cyclic_ntt(ntt_size, HE->plain_mod);
    uint64_t padded_HW = padded_feature_size * padded_feature_size;
    uint64_t offset = (kernel_size - 1) * (padded_feature_size + 1);

    Tensor<UnifiedPlaintext> wpt({tiled_in_channels, tiled_out_channels, tile_size}, HE->Backend());

    for (uint64_t ti = 0; ti < tiled_in_channels; ti++) {
        for (uint64_t tj = 0; tj < tiled_out_channels; tj++) {
            for (uint64_t k = 0; k < tile_size; k++) {
                std::vector<uint64_t> poly(HE->polyModulusDegree, 0);

                for (uint64_t l = 0; l < tile_size; l++) {
                    uint64_t in_blk, out_blk;

                    if (tile_size == 1) {
                        in_blk = ti;
                        out_blk = tj;
                    } else {
                        in_blk = ti * tile_size + (l + (input_rot - k % input_rot - 1)) % tile_size;
                        out_blk = tj * tile_size + (3 * tile_size - l - k - (input_rot - k % input_rot)) % tile_size;
                    }

                    if (in_blk >= num_blocks_in || out_blk >= num_blocks_out) continue;

                    std::vector<uint64_t> w_coef(ntt_size, 0);
                    for (uint64_t i = 0; i < block_size; i++) {
                        uint64_t out_ch = out_blk * block_size + i;
                        uint64_t in_ch = in_blk * block_size;
                        if (out_ch < out_channels && in_ch < in_channels) {
                            for (uint64_t m = 0; m < kernel_size; m++) {
                                for (uint64_t n = 0; n < kernel_size; n++) {
                                    w_coef[i * padded_HW + offset - m * padded_feature_size - n] =
                                        weight({out_ch, in_ch, m, n});
                                }
                            }
                        }
                    }

                    cyclic_ntt.ComputeForward(w_coef.data(), w_coef.data());

                    uint64_t slot_off = l * ntt_size;
                    for (uint64_t m = 0; m < ntt_size; m++) {
                        poly[slot_off + m] = w_coef[m];
                    }
                    if (tile_size > 1) {
                        uint64_t slot_off2 = l * ntt_size + HE->polyModulusDegree / 2;
                        for (uint64_t m = 0; m < ntt_size; m++) {
                            poly[slot_off2 + m] = w_coef[m];
                        }
                    }
                }

                bool all_zero = true;
                for (uint64_t m = 0; m < HE->polyModulusDegree && all_zero; m++) {
                    all_zero = (poly[m] == 0);
                }
                if (all_zero) {
                    poly[HE->polyModulusDegree - 1] = 1;
                }

                HE->encoder->encode(poly, wpt({ti, tj, k}));
            }
        }
    }

    return wpt;
}

Tensor<uint64_t> CirConv2D::PackActivation(Tensor<uint64_t> &x) {
    Utils::CyclicNTT cyclic_ntt(ntt_size, HE->plain_mod);
    uint64_t padded_HW = padded_feature_size * padded_feature_size;
    Tensor<uint64_t> ac_msg({tiled_in_channels, HE->polyModulusDegree});

    for (uint64_t ti = 0; ti < tiled_in_channels; ti++) {
        for (uint64_t l = 0; l < tile_size; l++) {
            uint64_t blk = ti * tile_size + l;
            if (blk >= num_blocks_in) continue;

            std::vector<uint64_t> x_coef(ntt_size, 0);
            for (uint64_t i = 0; i < block_size; i++) {
                uint64_t ch = blk * block_size + i;
                if (ch >= in_channels) continue;
                for (uint64_t j = 0; j < padded_feature_size; j++) {
                    for (uint64_t kk = 0; kk < padded_feature_size; kk++) {
                        if (j >= padding && j < padding + in_feature_size &&
                            kk >= padding && kk < padding + in_feature_size) {
                            x_coef[i * padded_HW + j * padded_feature_size + kk] =
                                x({ch, j - padding, kk - padding});
                        }
                    }
                }
            }

            cyclic_ntt.ComputeForward(x_coef.data(), x_coef.data());

            uint64_t slot_off = l * ntt_size;
            for (uint64_t m = 0; m < ntt_size; m++) {
                ac_msg({ti, slot_off + m}) = x_coef[m];
            }
            if (tile_size > 1) {
                uint64_t slot_off2 = l * ntt_size + HE->polyModulusDegree / 2;
                for (uint64_t m = 0; m < ntt_size; m++) {
                    ac_msg({ti, slot_off2 + m}) = x_coef[m];
                }
            }
        }
    }

    return ac_msg;
}

Tensor<UnifiedCiphertext> CirConv2D::HECompute(
    const Tensor<UnifiedPlaintext> &wpt,
    Tensor<UnifiedCiphertext> &ac_ct)
{
    const auto target = HE->server ? HE->Backend() : HOST;
    Tensor<UnifiedCiphertext> out_ct({tiled_out_channels}, HE->GenerateZeroCiphertext(target));

    if (!HE->server) return out_ct;

    UnifiedGaloisKeys* keys = HE->galoisKeys;

    if (tile_size == 1) {
        for (uint64_t tj = 0; tj < tiled_out_channels; tj++) {
            bool first = true;
            for (uint64_t ti = 0; ti < tiled_in_channels; ti++) {
                UnifiedCiphertext tmp(target);
                HE->evaluator->multiply_plain(ac_ct(ti), wpt({ti, tj, 0}), tmp);

                if (first) {
                    out_ct(tj) = tmp;
                    first = false;
                } else {
                    HE->evaluator->add_inplace(out_ct(tj), tmp);
                }
            }
        }
    } else {
        Tensor<UnifiedCiphertext> ac_rot({input_rot, tiled_in_channels},
                                          HE->GenerateZeroCiphertext(target));
        Tensor<UnifiedCiphertext> int_ct({tiled_out_channels, tile_size},
                                          HE->GenerateZeroCiphertext(target));

        for (uint64_t ti = 0; ti < tiled_in_channels; ti++) {
            ac_rot({0, ti}) = ac_ct(ti);
            for (uint64_t r = 1; r < input_rot; r++) {
                HE->evaluator->rotate_rows(ac_rot({r-1, ti}), ntt_size, *keys, ac_rot({r, ti}));
            }
        }

        for (uint64_t ti = 0; ti < tiled_in_channels; ti++) {
            for (uint64_t tj = 0; tj < tiled_out_channels; tj++) {
                for (uint64_t k = 0; k < tile_size; k++) {
                    uint64_t rot_idx = input_rot - 1 - k % input_rot;
                    UnifiedCiphertext tmp(target);
                    HE->evaluator->multiply_plain(ac_rot({rot_idx, ti}), wpt({ti, tj, k}), tmp);

                    if (ti == 0) {
                        int_ct({tj, k}) = tmp;
                    } else {
                        HE->evaluator->add_inplace(int_ct({tj, k}), tmp);
                    }
                }
            }
        }

        for (uint64_t tj = 0; tj < tiled_out_channels; tj++) {
            for (uint64_t k = 1; k < tile_size; k++) {
                if (k % input_rot != 0) {
                    HE->evaluator->add_inplace(int_ct({tj, k - k % input_rot}), int_ct({tj, k}));
                }
            }

            out_ct(tj) = int_ct({tj, 0});
            for (uint64_t g = 1; g < (tile_size + input_rot - 1) / input_rot; g++) {
                uint64_t k = g * input_rot;
                if (k < tile_size) {
                    HE->evaluator->rotate_rows(out_ct(tj), ntt_size * input_rot, *keys, out_ct(tj));
                    HE->evaluator->add_inplace(out_ct(tj), int_ct({tj, k}));
                }
            }
        }
    }

    return out_ct;
}

Tensor<uint64_t> CirConv2D::DepackResult(Tensor<uint64_t> &out_msg) {
    Utils::CyclicNTT cyclic_ntt(ntt_size, HE->plain_mod);
    uint64_t padded_HW = padded_feature_size * padded_feature_size;
    uint64_t offset = (kernel_size - 1) * (padded_feature_size + 1);
    Tensor<uint64_t> y({out_channels, out_feature_size, out_feature_size});

    for (uint64_t tj = 0; tj < tiled_out_channels; tj++) {
        for (uint64_t l = 0; l < tile_size; l++) {
            uint64_t out_blk;
            if (tile_size == 1) {
                out_blk = tj;
            } else {
                uint64_t G = (tile_size + input_rot - 1) / input_rot;
                out_blk = tj * tile_size + (3 * tile_size - l - G * input_rot) % tile_size;
            }

            if (out_blk >= num_blocks_out) continue;

            std::vector<uint64_t> y_ntt(ntt_size);
            uint64_t slot_off = l * ntt_size;
            for (uint64_t m = 0; m < ntt_size; m++) {
                y_ntt[m] = out_msg({tj, slot_off + m});
            }

            cyclic_ntt.ComputeInverse(y_ntt.data(), y_ntt.data());

            for (uint64_t i = 0; i < block_size; i++) {
                uint64_t out_ch = out_blk * block_size + i;
                if (out_ch >= out_channels) continue;

                for (uint64_t j = 0; j < out_feature_size; j++) {
                    for (uint64_t kk = 0; kk < out_feature_size; kk++) {
                        y({out_ch, j, kk}) = y_ntt[i * padded_HW + offset +
                            stride * j * padded_feature_size + stride * kk];
                    }
                }
            }
        }
    }

    return y;
}

Tensor<uint64_t> CirConv2D::operator()(Tensor<uint64_t> &x) {
    Tensor<uint64_t> ac_msg = PackActivation(x);
    Tensor<UnifiedCiphertext> ac_ct = Operator::SSToHE(ac_msg, HE);
    Tensor<UnifiedCiphertext> out_ct = HECompute(weight_pt, ac_ct);
    Tensor<uint64_t> out_msg = Operator::HEToSS(out_ct, HE);
    Tensor<uint64_t> y = DepackResult(out_msg);
    return y;
}

} // namespace LinearLayer
