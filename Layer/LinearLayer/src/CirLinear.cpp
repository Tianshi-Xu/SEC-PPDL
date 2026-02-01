/**
 * CirLinearNest: Block Circulant Linear Layer with Nested Encoding
 * 
 * This is the block circulant version of LinearNest.
 * 
 * Key insight: Each circulant block is encoded using coefficient encoding (Theorem 1),
 * then transformed via cyclic NTT. Multiple blocks can be packed into one ciphertext
 * and processed with BSGS optimization.
 * 
 * Dimensions:
 *   Input: (dim_0, dim_1) = (batch, input_channels)
 *   Weight: (dim_1, dim_2) = (input_channels, output_channels), block circulant
 *   Output: (dim_0, dim_2) = (batch, output_channels)
 * 
 * Block structure:
 *   num_blocks_1 = dim_1 / block_size (input blocks)
 *   num_blocks_2 = dim_2 / block_size (output blocks)
 * 
 * Degenerate cases:
 *   - num_blocks_1 = num_blocks_2 = 1: Single block, no BSGS needed
 *   - tile_size = 1: Each ciphertext holds one block, simple accumulation
 *   - tile_size >= num_blocks: All blocks fit in one ciphertext per dimension
 */

#include <LinearLayer/CirLinear.h>
#include <Utils/CyclicNTT.h>
#include <cassert>
#include <hexl/hexl.hpp>
#include <chrono>

using namespace seal;
using namespace HE;
using namespace HE::unified;

namespace LinearLayer {

// ======================== CirLinearNest ========================

CirLinearNest::CirLinearNest(uint64_t dim_0, uint64_t block_size,
                             const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias,
                             HE::HEEvaluator* HE)
    : dim_0(dim_0),
      block_size(block_size),
      weight(weight),
      bias(bias),
      HE(HE)
{
    std::vector<size_t> weight_shape = weight.shape();
    dim_1 = weight_shape[0];
    dim_2 = weight_shape[1];
    
    assert(dim_1 % block_size == 0 && "dim_1 must be divisible by block_size");
    assert(dim_2 % block_size == 0 && "dim_2 must be divisible by block_size");
    
    num_blocks_1 = dim_1 / block_size;
    num_blocks_2 = dim_2 / block_size;
    
    compute_he_params();
    
    if (HE->server) {
        weight_pt = PackWeight();
    }
}

CirLinearNest::CirLinearNest(uint64_t dim_0, uint64_t dim_1, uint64_t dim_2, uint64_t block_size,
                             HE::HEEvaluator* HE)
    : dim_0(dim_0),
      dim_1(dim_1),
      dim_2(dim_2),
      block_size(block_size),
      HE(HE)
{
    assert(dim_1 % block_size == 0 && "dim_1 must be divisible by block_size");
    assert(dim_2 % block_size == 0 && "dim_2 must be divisible by block_size");
    
    num_blocks_1 = dim_1 / block_size;
    num_blocks_2 = dim_2 / block_size;
    
    if (HE->server) {
        this->weight = Tensor<uint64_t>({dim_1, dim_2});
        this->bias = Tensor<uint64_t>({dim_0, dim_2});
        this->weight.randomize(16);
        this->bias.randomize(16);
    }
    
    compute_he_params();
    
    if (HE->server) {
        weight_pt = PackWeight();
    }
}

void CirLinearNest::compute_he_params() {
    /**
     * Parameter computation for CirLinearNest.
     * 
     * Each circulant block needs ntt_size = padded_dim_0 * block_size slots.
     * We can pack multiple blocks per ciphertext (tile_size blocks per half).
     * 
     * For BSGS optimization:
     *   - tile_size blocks are arranged on an anti-diagonal
     *   - input_rot = sqrt(tile_size) rotations needed
     * 
     * Degenerate cases are handled by limiting tile_size appropriately.
     */
    
    // Pad dim_0 to power of 2
    padded_dim_0 = 1;
    while (padded_dim_0 < dim_0) {
        padded_dim_0 <<= 1;
    }
    
    // NTT size for one circulant block (per Theorem 1)
    ntt_size = padded_dim_0 * block_size;
    
    // Maximum blocks that can fit per ciphertext half
    uint64_t max_tile = HE->polyModulusDegree / (2 * ntt_size);
    if (max_tile < 1) max_tile = 1;
    
    // Actual tile_size: limited by the number of blocks we actually have
    // No point having tile_size larger than num_blocks
    tile_size = std::min(max_tile, std::max(num_blocks_1, num_blocks_2));
    
    // Tiled dimensions
    tiled_blocks_1 = (num_blocks_1 + tile_size - 1) / tile_size;
    tiled_blocks_2 = (num_blocks_2 + tile_size - 1) / tile_size;
    
    // Padded to tile boundaries
    padded_blocks_1 = tiled_blocks_1 * tile_size;
    padded_blocks_2 = tiled_blocks_2 * tile_size;
    
    // BSGS: input_rot = ceil(sqrt(tile_size))
    // Special case: tile_size=1 means no rotation needed
    if (tile_size <= 1) {
        input_rot = 1;
    } else {
        input_rot = 1;
        while (input_rot * input_rot < tile_size) {
            input_rot++;
        }
    }
    
    // Pad weight
    padded_weight = Tensor<uint64_t>({padded_blocks_1 * block_size, padded_blocks_2 * block_size});
    for (uint64_t i = 0; i < dim_1; i++) {
        for (uint64_t j = 0; j < dim_2; j++) {
            padded_weight({i, j}) = weight({i, j});
        }
    }
    
    std::cout << "CirLinearNest params: dim_0=" << dim_0 << " padded=" << padded_dim_0
              << ", block_size=" << block_size << ", ntt_size=" << ntt_size
              << ", num_blocks=(" << num_blocks_1 << "," << num_blocks_2 << ")"
              << ", tile_size=" << tile_size << ", input_rot=" << input_rot
              << ", tiled=(" << tiled_blocks_1 << "," << tiled_blocks_2 << ")" << std::endl;
}

Tensor<UnifiedPlaintext> CirLinearNest::PackWeight() {
    /**
     * Pack weights into plaintexts.
     * 
     * Convention (same as TestMatmul):
     *   weight shape: (dim_1=input, dim_2=output)
     *   W[out, in] = weight({in, out})
     *   First column of block W_block[:, 0]: W_block[i, 0] = weight({in_blk*b, out_blk*b + i})
     * 
     * For tile_size=1: simple encoding like CirLinearSimple
     * For tile_size>1: BSGS encoding with anti-diagonal pattern
     */
    Utils::CyclicNTT cyclic_ntt(ntt_size, HE->plain_mod);
    Tensor<UnifiedPlaintext> wpt({tiled_blocks_1, tiled_blocks_2, tile_size}, HE->Backend());
    
    for (uint64_t ti = 0; ti < tiled_blocks_1; ti++) {
        for (uint64_t tj = 0; tj < tiled_blocks_2; tj++) {
            for (uint64_t k = 0; k < tile_size; k++) {
                std::vector<uint64_t> poly(HE->polyModulusDegree, 0);
                
                for (uint64_t l = 0; l < tile_size; l++) {
                    uint64_t in_blk, out_blk;
                    
                    if (tile_size == 1) {
                        in_blk = ti;
                        out_blk = tj;
                    } else {
                        // BSGS pattern from LinearNest
                        in_blk = ti * tile_size + (l + input_rot - 1 - k % input_rot) % tile_size;
                        out_blk = tj * tile_size + (tile_size - 1 - l - k / input_rot + tile_size) % tile_size;
                    }
                    
                    if (in_blk >= num_blocks_1 || out_blk >= num_blocks_2) continue;
                    
                    // Coefficient encode the circulant block's first column
                    std::vector<uint64_t> w_coef(ntt_size, 0);
                    for (uint64_t i = 0; i < block_size; i++) {
                        uint64_t in_ch = in_blk * block_size;
                        uint64_t out_ch = out_blk * block_size + i;
                        if (in_ch < dim_1 && out_ch < dim_2) {
                            w_coef[i * padded_dim_0] = padded_weight({in_ch, out_ch});
                        }
                    }
                    
                    // Apply cyclic NTT (in-place)
                    cyclic_ntt.ComputeForward(w_coef.data(), w_coef.data());
                    
                    // Place in polynomial
                    // tile_size=1: only first half (like CirLinearSimple)
                    // tile_size>1: both halves for row batching
                    uint64_t offset = l * ntt_size;
                    for (uint64_t m = 0; m < ntt_size; m++) {
                        poly[offset + m] = w_coef[m];
                    }
                    if (tile_size > 1) {
                        uint64_t offset2 = l * ntt_size + HE->polyModulusDegree / 2;
                        for (uint64_t m = 0; m < ntt_size; m++) {
                            poly[offset2 + m] = w_coef[m];
                        }
                    }
                }
                
                HE->encoder->encode(poly, wpt({ti, tj, k}));
            }
        }
    }
    
    return wpt;
}

Tensor<uint64_t> CirLinearNest::PackActivation(Tensor<uint64_t> &x) {
    /**
     * Pack activations into message tensor for encryption.
     * 
     * For tile_size=1: like CirLinearSimple, only first half
     * For tile_size>1: both halves for row batching
     */
    Utils::CyclicNTT cyclic_ntt(ntt_size, HE->plain_mod);
    Tensor<uint64_t> ac_msg({tiled_blocks_1, HE->polyModulusDegree});
    
    for (uint64_t ti = 0; ti < tiled_blocks_1; ti++) {
        for (uint64_t l = 0; l < tile_size; l++) {
            uint64_t blk = ti * tile_size + l;
            if (blk >= num_blocks_1) continue;
            
            // Coefficient encode: x̂[i * padded_dim_0 + j] = X[i, j]
            std::vector<uint64_t> x_coef(ntt_size, 0);
            for (uint64_t i = 0; i < block_size; i++) {
                for (uint64_t j = 0; j < dim_0; j++) {
                    uint64_t ch = blk * block_size + i;
                    if (ch < dim_1) {
                        x_coef[i * padded_dim_0 + j] = x({j, ch});
                    }
                }
            }
            
            // Apply cyclic NTT (in-place)
            cyclic_ntt.ComputeForward(x_coef.data(), x_coef.data());
            
            // Place in polynomial
            uint64_t offset = l * ntt_size;
            for (uint64_t m = 0; m < ntt_size; m++) {
                ac_msg({ti, offset + m}) = x_coef[m];
            }
            if (tile_size > 1) {
                uint64_t offset2 = l * ntt_size + HE->polyModulusDegree / 2;
                for (uint64_t m = 0; m < ntt_size; m++) {
                    ac_msg({ti, offset2 + m}) = x_coef[m];
                }
            }
        }
    }
    
    return ac_msg;
}

Tensor<UnifiedCiphertext> CirLinearNest::HECompute(
    const Tensor<UnifiedPlaintext> &wpt,
    Tensor<UnifiedCiphertext> &ac_ct) 
{
    /**
     * HE computation with BSGS optimization.
     * 
     * Degenerate cases:
     *   - tile_size=1: No BSGS, simple multiply-accumulate
     *   - Otherwise: Full BSGS with rotations
     */
    const auto target = HE->server ? HE->Backend() : HOST;
    Tensor<UnifiedCiphertext> out_ct({tiled_blocks_2}, HE->GenerateZeroCiphertext(target));
    
    rotation_count = 0;  // Reset rotation count
    multiply_count = 0;  // Reset multiply count
    rotation_time_ms = 0;
    multiply_time_ms = 0;
    
    if (!HE->server) return out_ct;
    
    UnifiedGaloisKeys* keys = HE->galoisKeys;
    
    auto time_rotation = [&](auto&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        rotation_time_ms += std::chrono::duration<double, std::milli>(end - start).count();
        rotation_count++;
    };
    
    auto time_multiply = [&](auto&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        multiply_time_ms += std::chrono::duration<double, std::milli>(end - start).count();
        multiply_count++;
    };
    
    if (tile_size == 1) {
        // Simple case: no BSGS, just multiply and accumulate
        for (uint64_t tj = 0; tj < tiled_blocks_2; tj++) {
            bool first = true;
            for (uint64_t ti = 0; ti < tiled_blocks_1; ti++) {
                UnifiedCiphertext tmp(target);
                time_multiply([&]() {
                    HE->evaluator->multiply_plain(ac_ct(ti), wpt({ti, tj, 0}), tmp);
                });
                
                if (first) {
                    out_ct(tj) = tmp;
                    first = false;
                } else {
                    HE->evaluator->add_inplace(out_ct(tj), tmp);
                }
            }
        }
    } else {
        // Full BSGS
        Tensor<UnifiedCiphertext> ac_rot({input_rot, tiled_blocks_1}, 
                                          HE->GenerateZeroCiphertext(target));
        Tensor<UnifiedCiphertext> int_ct({tiled_blocks_2, tile_size}, 
                                          HE->GenerateZeroCiphertext(target));
        
        // Step 1: Precompute input rotations
        for (uint64_t ti = 0; ti < tiled_blocks_1; ti++) {
            ac_rot({0, ti}) = ac_ct(ti);
            for (uint64_t r = 1; r < input_rot; r++) {
                // Rotate by ntt_size (one block)
                time_rotation([&]() {
                    HE->evaluator->rotate_rows(ac_rot({r-1, ti}), ntt_size, *keys, ac_rot({r, ti}));
                });
            }
        }
        
        // Step 2: Multiply and accumulate
        for (uint64_t ti = 0; ti < tiled_blocks_1; ti++) {
            for (uint64_t tj = 0; tj < tiled_blocks_2; tj++) {
                for (uint64_t k = 0; k < tile_size; k++) {
                    uint64_t rot_idx = input_rot - 1 - k % input_rot;
                    UnifiedCiphertext tmp(target);
                    time_multiply([&]() {
                        HE->evaluator->multiply_plain(ac_rot({rot_idx, ti}), wpt({ti, tj, k}), tmp);
                    });
                    
                    if (ti == 0) {
                        int_ct({tj, k}) = tmp;
                    } else {
                        HE->evaluator->add_inplace(int_ct({tj, k}), tmp);
                    }
                }
            }
        }
        
        // Step 3: Reduce along BSGS dimension
        for (uint64_t tj = 0; tj < tiled_blocks_2; tj++) {
            // First reduce within each input_rot group
            for (uint64_t k = 1; k < tile_size; k++) {
                if (k % input_rot != 0) {
                    HE->evaluator->add_inplace(int_ct({tj, k - k % input_rot}), int_ct({tj, k}));
                }
            }
            
            // Then reduce across groups with output rotations
            out_ct(tj) = int_ct({tj, 0});
            for (uint64_t g = 1; g < (tile_size + input_rot - 1) / input_rot; g++) {
                uint64_t k = g * input_rot;
                if (k < tile_size) {
                    time_rotation([&]() {
                        HE->evaluator->rotate_rows(out_ct(tj), ntt_size * input_rot, *keys, out_ct(tj));
                    });
                    HE->evaluator->add_inplace(out_ct(tj), int_ct({tj, k}));
                }
            }
        }
    }
    
    return out_ct;
}

Tensor<uint64_t> CirLinearNest::DepackResult(Tensor<uint64_t> &out_msg) {
    /**
     * Depack results from HE output.
     * 
     * Apply cyclic iNTT to each block, then extract using Theorem 1:
     *   Y[batch, out_ch] = ŷ[i * padded_dim_0 + batch]
     * where out_ch = out_blk * block_size + i
     */
    Utils::CyclicNTT cyclic_ntt(ntt_size, HE->plain_mod);
    Tensor<uint64_t> y({dim_0, dim_2});
    
    for (uint64_t tj = 0; tj < tiled_blocks_2; tj++) {
        for (uint64_t l = 0; l < tile_size; l++) {
            // Determine output block index
            // For tile_size=1: out_blk = tj
            // Otherwise: reverse of BSGS pattern
            uint64_t out_blk;
            if (tile_size == 1) {
                out_blk = tj;
            } else {
                // Result is at slot l, corresponding to anti-diagonal position
                // When k=0, out_blk = tj * tile_size + (tile_size - 1 - l) % tile_size
                out_blk = tj * tile_size + (tile_size - 1 - l + tile_size) % tile_size;
            }
            
            if (out_blk >= num_blocks_2) continue;
            
            // Extract NTT values for this block
            std::vector<uint64_t> y_ntt(ntt_size);
            uint64_t offset = l * ntt_size;
            for (uint64_t m = 0; m < ntt_size; m++) {
                y_ntt[m] = out_msg({tj, offset + m});
            }
            
            // Apply cyclic iNTT (in-place)
            cyclic_ntt.ComputeInverse(y_ntt.data(), y_ntt.data());
            
            // Extract: Y[batch, out_ch] = ŷ[i * padded_dim_0 + batch]
            for (uint64_t i = 0; i < block_size; i++) {
                uint64_t out_ch = out_blk * block_size + i;
                if (out_ch >= dim_2) continue;
                
                for (uint64_t batch = 0; batch < dim_0; batch++) {
                    y({batch, out_ch}) = y_ntt[i * padded_dim_0 + batch];
                }
            }
        }
    }
    
    return y;
}

Tensor<uint64_t> CirLinearNest::operator()(Tensor<uint64_t> &x) {
    Tensor<uint64_t> ac_msg = PackActivation(x);
    Tensor<UnifiedCiphertext> ac_ct = Operator::SSToHE(ac_msg, HE);
    Tensor<UnifiedCiphertext> out_ct = HECompute(weight_pt, ac_ct);
    Tensor<uint64_t> out_msg = Operator::HEToSS(out_ct, HE);
    Tensor<uint64_t> y = DepackResult(out_msg);
    return y;
}

} // namespace LinearLayer
