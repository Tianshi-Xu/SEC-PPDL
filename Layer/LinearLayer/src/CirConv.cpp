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
#include "ConvWorkers.h"
#include <Utils/CyclicNTT.h>
#include <algorithm>
#include <stdexcept>

using namespace seal;
using namespace HE;
using namespace HE::unified;

namespace LinearLayer {

// ======================== CirConv2D ========================

CirConv2D::CirConv2D(uint64_t in_feature_size, uint64_t stride, uint64_t padding,
                     uint64_t block_size, const Tensor<uint64_t>& weight,
                     const Tensor<uint64_t>& bias, HEEvaluator* HE, size_t he_threads)
    : Conv2D(in_feature_size, stride, padding, weight, bias, HE),
      block_size(block_size), he_threads(HE->Backend() == HOST ? he_threads : 1)
{
    compute_he_params(in_feature_size);
    if (HE->server) {
        weight_pt = PackWeight();
    }
}

CirConv2D::CirConv2D(uint64_t in_feature_size, uint64_t in_channels,
                     uint64_t out_channels, uint64_t kernel_size,
                     uint64_t stride, uint64_t block_size, HEEvaluator* HE, size_t he_threads)
    : Conv2D(in_feature_size, in_channels, out_channels, kernel_size, stride, HE),
      block_size(block_size), he_threads(HE->Backend() == HOST ? he_threads : 1)
{
    compute_he_params(in_feature_size);
    if (HE->server) {
        weight_pt = PackWeight();
    }
}

void CirConv2D::compute_he_params(uint64_t in_feature_size) {
    if (block_size == 0 || in_channels % block_size != 0 || out_channels % block_size != 0) {
        throw std::invalid_argument("CirConv2D block size must divide the input and output channels");
    }

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
    if (tile_size == 0) {
        throw std::invalid_argument("CirConv2D NTT block exceeds the BFV row capacity");
    }

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
    nonzero_weight = Tensor<uint8_t>(wpt.shape());

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

                if (std::any_of(poly.begin(), poly.end(), [](uint64_t value) { return value != 0; })) {
                    HE->encoder->encode(poly, wpt({ti, tj, k}));
                    nonzero_weight({ti, tj, k}) = 1;
                }
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
    Tensor<UnifiedCiphertext> out_ct({tiled_out_channels}, target);
    if (!HE->server) return out_ct;

    const auto &keys = *HE->galoisKeys;
    Tensor<UnifiedCiphertext> ac_rot({input_rot, tiled_in_channels}, target);
    RunConvWorkers(tiled_in_channels, he_threads, [&](size_t begin, size_t end) {
        for (size_t ti = begin; ti < end; ++ti) {
            ac_rot({0, ti}) = ac_ct(ti);
            for (size_t r = 1; r < input_rot; ++r) {
                HE->evaluator->rotate_rows(ac_rot({r - 1, ti}), ntt_size, keys, ac_rot({r, ti}));
            }
        }
    });

    const auto accumulate = [&](size_t tj, size_t group, UnifiedCiphertext &output,
                                UnifiedCiphertext &product, bool has_product) {
        const size_t end = std::min(group + input_rot, tile_size);
        for (size_t k = group; k < end; ++k) {
            const size_t rot_idx = input_rot - 1 - (k - group);
            for (size_t ti = 0; ti < tiled_in_channels; ++ti) {
                if (nonzero_weight({ti, tj, k})) {
                    if (has_product) {
                        HE->evaluator->multiply_plain(ac_rot({rot_idx, ti}), wpt({ti, tj, k}), product);
                        HE->evaluator->add_inplace(output, product);
                    } else {
                        HE->evaluator->multiply_plain(ac_rot({rot_idx, ti}), wpt({ti, tj, k}), output);
                        has_product = true;
                    }
                }
            }
        }
    };

    if (he_threads <= 1) {
        UnifiedCiphertext product(target);
        for (size_t tj = 0; tj < tiled_out_channels; ++tj) {
            auto &output = out_ct(tj);
            output = HE->GenerateZeroCiphertext(target);
            for (size_t group = 0; group < tile_size; group += input_rot) {
                if (group != 0) {
                    HE->evaluator->rotate_rows(output, ntt_size * input_rot, keys, output);
                }
                accumulate(tj, group, output, product, true);
            }
        }
        return out_ct;
    }

    // The products within each giant-step group are independent. Merge the
    // group sums in the original rotation order, including empty groups.
    const size_t groups = (tile_size + input_rot - 1) / input_rot;
    Tensor<UnifiedCiphertext> sums({tiled_out_channels, groups}, target);
    RunConvWorkers(sums.size(), he_threads, [&](size_t begin, size_t end) {
        UnifiedCiphertext product(target);
        for (size_t task = begin; task < end; ++task) {
            accumulate(task / groups, (task % groups) * input_rot, sums(task), product, false);
        }
    });
    RunConvWorkers(tiled_out_channels, he_threads, [&](size_t begin, size_t end) {
        for (size_t tj = begin; tj < end; ++tj) {
            auto &output = out_ct(tj);
            output = HE->GenerateZeroCiphertext(target);
            for (size_t group = 0; group < groups; ++group) {
                if (group != 0) {
                    HE->evaluator->rotate_rows(output, ntt_size * input_rot, keys, output);
                }
                const auto &sum = sums({tj, group});
                if (sum.hcipher().size() != 0) {
                    HE->evaluator->add_inplace(output, sum);
                }
            }
        }
    });
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
