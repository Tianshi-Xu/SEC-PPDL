#include <LinearLayer/Conv.h>

using namespace seal;
using namespace HE;
using namespace HE::unified;

namespace LinearLayer {
    
Conv2DNest::Conv2DNest(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HEEvaluator* HE)
    : Conv2D(in_feature_size, stride, padding, weight, bias, HE)
{
    // assert(padded_feature_size * padded_feature_size > HE->polyModulusDegree / 2 && "Input feature map is too big, please use Cheetah instead.");

    // Pad the feature size to PO2.
    int tmp_size = in_feature_size + 2 * padding - 1;
    for (int i = 0; i < 5; i++) {
        tmp_size |= tmp_size >> (1 << i);
    }
    padded_feature_size = tmp_size + 1;

    /* 
    The number of slots are divided into two identical parts in BFV, 
    so we process out_channels in parallel, or more concretely, we turn Ci*Co/N into Ci*(Co/2)/(N/2).
    */
    tile_size = HE->polyModulusDegree / (2 * padded_feature_size * padded_feature_size);
    out_channels /= 2;
    
    tiled_in_channels = in_channels / tile_size + (in_channels % tile_size != 0); // ceiling
    tiled_out_channels = out_channels / tile_size + (out_channels % tile_size != 0);
    input_rot = std::sqrt(tile_size);
    if (HE->server) {
        weight_pt = PackWeight();
    }
}

void Conv2DNest::compute_he_params(uint64_t in_feature_size) {
    // cout << "compute_he_params called" << endl;
    cout << "in_feature_size: " << in_feature_size << ", padding: " << this->padding << ", kernel_size: " << this->kernel_size << ", stride: " << this->stride << endl;
    int tmp_size = in_feature_size + 2 * this->padding - 1;
    for (int i = 0; i < 5; i++) {
        tmp_size |= tmp_size >> (1 << i);
    }
    cout << "tmp_size: " << tmp_size << endl;
    this->padded_feature_size = tmp_size + 1;
    this->tile_size = HE->polyModulusDegree / (2 * this->padded_feature_size * this->padded_feature_size);
    if (this->tile_size <=0) {
        cout << "tile_size is too small for: in_feature_size: " << in_feature_size << ", padding: " << this->padding << ", kernel_size: " << this->kernel_size << ", stride: " << this->stride << endl;
        exit(1);
    }
    this->out_channels /= 2;
    this->tiled_in_channels = in_channels / this->tile_size + (in_channels % this->tile_size != 0);
    this->tiled_out_channels = out_channels / this->tile_size + (out_channels % this->tile_size != 0);
    this->input_rot = std::sqrt(this->tile_size);  // to be checked
    this->out_feature_size = (in_feature_size + 2 * this->padding - kernel_size) / stride + 1;
    cout << "padded_feature_size: " << this->padded_feature_size << endl;
    cout << "tile_size: " << this->tile_size << endl;
    cout << "tiled_in_channels: " << this->tiled_in_channels << endl;
    cout << "tiled_out_channels: " << this->tiled_out_channels << endl;
    cout << "input_rot: " << this->input_rot << endl;
    cout << "out_feature_size: " << this->out_feature_size << endl;
}

Conv2DNest::Conv2DNest(uint64_t in_feature_size, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, HE::HEEvaluator* HE)
    : Conv2D(in_feature_size, in_channels, out_channels, kernel_size, stride, HE)
{
    compute_he_params(in_feature_size);
    // this->weight.print_shape();
    if(HE->server) {
        weight_pt = PackWeight();
    }
}

Tensor<UnifiedPlaintext> Conv2DNest::PackWeight() {
    intel::hexl::NTT ntt(padded_feature_size * padded_feature_size, HE->plain_mod);
    uint64_t offset = (kernel_size - 1) * (padded_feature_size + 1);
    Tensor<UnifiedPlaintext> weight_pt({tiled_in_channels, tiled_out_channels, tile_size}, HE->Backend());

    for (uint64_t i = 0; i < tiled_in_channels; i++) {
        for (uint64_t j = 0; j < tiled_out_channels; j++) {
            for (uint64_t k = 0; k < tile_size; k++) {
                /*
                Each ciphertext/plaintext executes tile_size convolutions in a SIMD manner.
                The remaining dimensions are tiled_in_channels, tiled_out_channels and tile_size.
                */
                std::vector<uint64_t> tmp_vec(HE->polyModulusDegree, 0);
                for (uint64_t l = 0; l < tile_size; l++) {
                    /*
                    We compute the indices of convolution kernels here.
                    We encode along anti-diagonals, but encoding along diagonals is also feasible.
                    The indices involve three parts: the indices of block i, j, the index in the
                    anti-diagonal l, and the preprocessed input rotation k.
                    */
                    uint64_t in_channel_idx = i * tile_size + (l + (input_rot - k % input_rot - 1)) % tile_size;
                    uint64_t out_channel_idx = j * tile_size + (3 * tile_size - l - k - (input_rot - k % input_rot)) % tile_size;
                    if (in_channel_idx < in_channels && out_channel_idx < out_channels) {
                        for (uint64_t m = 0; m < kernel_size; m++) {
                            for (uint64_t n = 0; n < kernel_size; n++) {
                                /*
                                We fill the weights into polynomials here. 
                                This is almost equivalent to the case of SISO convolution, where the offset guarantees that
                                the weights start from the 0th index.
                                */
                                uint64_t poly_idx = l * padded_feature_size * padded_feature_size + offset - m * padded_feature_size - n;
                                tmp_vec[poly_idx] = weight({out_channel_idx, in_channel_idx, m, n});
                                tmp_vec[poly_idx + HE->polyModulusDegree / 2] = weight({out_channel_idx + out_channels, in_channel_idx, m, n});
                            }
                        }
                    }
                }

                // We perform NTT independently for each block in the ciphertext. 
                for (uint64_t l = 0; l < 2 * tile_size; l++) {
                    std::vector<uint64_t> tmp_ntt(padded_feature_size * padded_feature_size, 0);
                    for (uint64_t m = 0; m < padded_feature_size * padded_feature_size; m++) {
                        tmp_ntt[m] = tmp_vec[l * padded_feature_size * padded_feature_size + m];
                    }
                    ntt.ComputeForward(tmp_ntt.data(), tmp_ntt.data(), 1, 1);
                    for (uint64_t m = 0; m < padded_feature_size * padded_feature_size; m++) {
                        tmp_vec[l * padded_feature_size * padded_feature_size + m] = tmp_ntt[m];
                    }
                }

                // We set unused slots to non-zero values to avoid transparent ciphertext error.
                bool zero_flag = 1;
                for (uint64_t l = 0; l < HE->polyModulusDegree; l++) {
                    zero_flag = zero_flag && (tmp_vec[l] == 0);
                }
                if (zero_flag) {
                    tmp_vec[HE->polyModulusDegree - 1] = 1; 
                }
                HE->encoder->encode(tmp_vec, weight_pt({i, j, k}));
            }
        }
    }

    return weight_pt;
}

Tensor<uint64_t> Conv2DNest::PackActivation(Tensor<uint64_t> &x) {
    intel::hexl::NTT ntt(padded_feature_size * padded_feature_size, HE->plain_mod);
    Tensor<uint64_t> ac_msg({tiled_in_channels, HE->polyModulusDegree});

    for (uint64_t i = 0; i < tiled_in_channels; i++) {
        /*
        Similarly, we initialize ciphertexts along the remaining dimension, tiled_in_channels.
        The encoding for the activation is straightforward. We just flatten the original tensor.
        */
        for (uint64_t j = 0; j < tile_size; j++) {
            if (i * tile_size + j < in_channels) {
                for (uint64_t k = 0; k < padded_feature_size; k++) {
                    for (uint64_t l = 0; l < padded_feature_size; l++) {
                        if (k >= padding && k < padding + in_feature_size && l >= padding && l < padding + in_feature_size) {
                            uint64_t idx = j * padded_feature_size * padded_feature_size + k * padded_feature_size + l;
                            ac_msg({i, idx}) = x({i * tile_size + j, k - padding, l - padding}); // dim(x) = {Ci, Hi, Wi}
                            ac_msg({i, idx + HE->polyModulusDegree / 2}) = x({i * tile_size + j, k - padding, l - padding});
                        }
                    }
                }
            }
        }

        for (uint64_t j = 0; j < 2 * tile_size; j++) {
            std::vector<uint64_t> tmp_ntt(padded_feature_size * padded_feature_size, 0);
            for (uint64_t k = 0; k < padded_feature_size * padded_feature_size; k++) {
                tmp_ntt[k] = ac_msg({i, j * padded_feature_size * padded_feature_size + k});
            }
            ntt.ComputeForward(tmp_ntt.data(), tmp_ntt.data(), 1, 1);
            for (uint64_t k = 0; k < padded_feature_size * padded_feature_size; k++) {
                ac_msg({i, j * padded_feature_size * padded_feature_size + k}) = tmp_ntt[k];
            }
        }
    }

    return ac_msg;
}

Tensor<UnifiedCiphertext> Conv2DNest::HECompute(const Tensor<UnifiedPlaintext> &weight_pt, Tensor<UnifiedCiphertext> &ac_ct) {
    /** 
     *  NOTE:
     *  Server computes on HE->Backend()
     *  Client does nothing
     */
    const auto target = HE->server ? HE->Backend() : HOST;
    Tensor<UnifiedCiphertext> out_ct({tiled_out_channels}, HE->GenerateZeroCiphertext(target));

    if (HE->server) {
        Tensor<UnifiedCiphertext> ac_rot_ct({input_rot, tiled_in_channels}, HE->GenerateZeroCiphertext(target));
        Tensor<UnifiedCiphertext> int_ct({tiled_out_channels, tile_size}, HE->GenerateZeroCiphertext(target));
        UnifiedGaloisKeys* keys = HE->galoisKeys;

        // First, complete the input rotation.
        for (uint64_t i = 0; i < input_rot; i++) {
            for (uint64_t j = 0; j < tiled_in_channels; j++) {
                if (i) {
                    // cout << "rot1:" << padded_feature_size * padded_feature_size << endl;
                    HE->evaluator->rotate_rows(ac_rot_ct({i - 1, j}), padded_feature_size * padded_feature_size, *keys, ac_rot_ct({i, j}));
                    
                }
                else {
                    ac_rot_ct({i, j}) = ac_ct(j);
                }
            }
        }
        // Then, complete all the multiplication, and reduce along the input channel dimension.
        for (uint64_t i = 0; i < tiled_in_channels; i++) {
            for (uint64_t j = 0; j < tiled_out_channels; j++) {
                for (uint64_t k = 0; k < tile_size; k++) {
                    UnifiedCiphertext tmp_ct(target);
                    HE->evaluator->multiply_plain(ac_rot_ct({input_rot - 1 - k % input_rot, i}), weight_pt({i, j, k}), tmp_ct);
                    if (i) {
                        HE->evaluator->add_inplace(int_ct({j, k}), tmp_ct);
                    }
                    else {
                        int_ct({j, k}) = tmp_ct;
                    }
                }
            }
        }
        for (uint64_t i = 0; i < tiled_out_channels; i++) {
            // Reduce along the input rotation dimension, since it has been completed.
            for (uint64_t j = 0; j < tile_size; j++) {
                if (j % input_rot) {
                    HE->evaluator->add_inplace(int_ct({i, j - j % input_rot}), int_ct({i, j}));
                }
            }
            out_ct(i) = int_ct({i, 0});
            // Complete output rotation to reduce along this dimension.
            for (uint64_t j = input_rot; j < tile_size; j += input_rot) {
                // cout << "rot2:" << padded_feature_size * padded_feature_size * input_rot << endl;
                HE->evaluator->rotate_rows(out_ct(i), padded_feature_size * padded_feature_size * input_rot, *keys, out_ct(i));
                
                HE->evaluator->add_inplace(out_ct(i), int_ct({i, j}));
            }
        }
    }

    return out_ct;
}

Tensor<uint64_t> Conv2DNest::DepackResult(Tensor<uint64_t> &out_msg) {
    out_channels *= 2;
    Tensor<uint64_t> y({out_channels, out_feature_size, out_feature_size});
    intel::hexl::NTT ntt(padded_feature_size * padded_feature_size, HE->plain_mod);

    // iNTT needs to be executed before depacking.
    for (uint64_t i = 0; i < tiled_out_channels; i++) {
        std::vector<uint64_t> tmp_ntt(padded_feature_size * padded_feature_size, 0);
        for (uint64_t j = 0; j < 2 * tile_size; j++) {
            for (uint64_t k = 0; k < padded_feature_size * padded_feature_size; k++) {
                tmp_ntt[k] = out_msg({i, j * padded_feature_size * padded_feature_size + k});
            }
            ntt.ComputeInverse(tmp_ntt.data(), tmp_ntt.data(), 1, 1);
            for (uint64_t k = 0; k < padded_feature_size * padded_feature_size; k++) {
                out_msg({i, j * padded_feature_size * padded_feature_size + k}) = tmp_ntt[k];
            }
        }
    }

    /*
    Depacking the results is almost the inverse of flattening, except considering the stride and 
    the weight offset. Also, since we use anti-diagonal, the blocks here are in the inversed order. 
    Note: The second half of output channels are stored in the second half of slots (offset by tile_size).
    */
    uint64_t half_out_channels = out_channels / 2;
    for (uint64_t i = 0; i < out_channels; i++) {
        for (uint64_t j = 0; j < out_feature_size; j++) {
            for (uint64_t k = 0; k < out_feature_size; k++) {
                uint64_t offset = stride * padded_feature_size * j + stride * k + (kernel_size - 1) * (padded_feature_size + 1);
                if (i < half_out_channels) {
                    // First half: use slot index directly (0 to tile_size-1)
                    uint64_t slot_idx = (tile_size - i % tile_size) % tile_size;
                    y({i, j, k}) = out_msg({i / tile_size, slot_idx * padded_feature_size * padded_feature_size + offset});
                }
                else {
                    // Second half: use slot index + tile_size to access second half of slots
                    uint64_t local_i = i - half_out_channels;
                    uint64_t slot_idx = (tile_size - local_i % tile_size) % tile_size + tile_size;
                    y({i, j, k}) = out_msg({local_i / tile_size, slot_idx * padded_feature_size * padded_feature_size + offset});
                }
            }
        }
    }

    return y;
}

Tensor<uint64_t> Conv2DNest::operator()(Tensor<uint64_t> &x) {  // x.shape = {Ci, H, W}
    // cout << "in Conv2D, x.shape:" << endl;
    // x.print_shape();
    // cout << "in Conv2DNest operator" << endl;
    // cout << "x.shape" << endl;
    // x.print_shape();
    // cout << "weight.shape" << endl;
    // this->weight.print_shape();
    if (this->in_feature_size != x.shape()[1]) {
        cout << "in_feature_size:" << this->in_feature_size << endl;
        cout << "wrong input feature size" << endl;
        exit(1);
    }
    // std::cout << "Conv2DNest operator called" << std::endl;
    Tensor<uint64_t> ac_msg = PackActivation(x);  // ac_msg.shape = {ci, N}
    // std::cout << "ac_msg generated" << std::endl;
    Tensor<UnifiedCiphertext> ac_ct = Operator::SSToHE(ac_msg, HE);  // ac_ct.shape = {ci}
    // std::cout << "ac_ct generated" << std::endl;
    Tensor<UnifiedCiphertext> out_ct = HECompute(weight_pt, ac_ct);  // out_ct.shape = {co}
    // std::cout << "out_ct generated: " << out_ct(0).location() << std::endl;

    
    Tensor<uint64_t> out_msg = Operator::HEToSS(out_ct, HE);  // out_msg.shape = {co, N}
    // std::cout << "out_msg generated" << std::endl;
    // out_msg.print_shape();
    Tensor<uint64_t> y = DepackResult(out_msg);  // y.shape = {Co, H, W}
    // std::cout << "y generated" << std::endl;
    return y;
};

} // namespace LinearLayer
