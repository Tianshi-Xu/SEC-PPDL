#include <Linear/Conv.h>
#include <cassert>

using namespace seal;

// Extract shared parameters. Let dim(w) = {Co, Ci, H, W}
Conv2D::Conv2D(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HEEvaluator* HE)
    : in_feature_size(in_feature_size), 
      weight(weight), 
      bias(bias), 
      HE(HE)
{
    std::vector<size_t> weight_shape = weight.shape();

    assert(weight_shape[0] == bias.shape()[0] && "Output channel does not match.");
    assert(weight_shape[2] == weight_shape[3] && "Input feature map is not a square.");
    assert(in_feature_size - weight_shape[2] + 2 * padding >= 0 && "Input feature map is too small.");

    out_channels = weight_shape[0];
    in_channels = weight_shape[1];
    kernel_size = weight_shape[2];
    out_feature_size = (in_feature_size + 2 * padding - kernel_size) / stride + 1;
};


Conv2DNest::Conv2DNest(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HEEvaluator* HE)
    : Conv2D(in_feature_size, stride, padding, weight, bias, HE)
{
    int tmp_size = in_feature_size + 2 * padding - 1;
    for (int i = 0; i < 5; i++) {
        tmp_size |= tmp_size >> (1 << i);
    }
    padded_feature_size = tmp_size + 1;
    tile_size = HE->polyModulusDegree / (2 * padded_feature_size * padded_feature_size);
    out_channels /= 2;
    tiled_in_channels = in_channels / tile_size + (in_channels % tile_size != 0);
    tiled_out_channels = out_channels / tile_size + (out_channels % tile_size != 0);
    input_rot = std::sqrt(tile_size);  // to be checked

    if (HE->server) {
        weight_pt = PackWeight();
    }
}

Tensor<Plaintext> Conv2DNest::PackWeight() {
    uint64_t offset = (kernel_size - 1) * (in_feature_size + 1);
    Tensor<Plaintext> weight_pt({tiled_in_channels, tiled_out_channels, tile_size});

    for (uint64_t i = 0; i < tiled_in_channels; i++) {
        for (uint64_t j = 0; j < tiled_out_channels; j++) {
            for (uint64_t k = 0; k < tile_size; k++) {
                std::vector<uint64_t> tmp_vec(HE->polyModulusDegree, 0);
                for (uint64_t l = 0; l < tile_size; l++) {
                    uint64_t in_channel_idx = i * tile_size + (l + (input_rot - k % input_rot - 1)) % tile_size;
                    uint64_t out_channel_idx = j * tile_size + (3 * tile_size - l - k - (input_rot - k % input_rot - 1)) % tile_size;
                    if (in_channel_idx < in_channels && out_channel_idx < out_channels) {
                        for (uint64_t m = 0; m < kernel_size; m++) {
                            for (uint64_t n = 0; n < kernel_size; n++) {
                                uint64_t poly_idx = l * padded_feature_size * padded_feature_size + offset - m * padded_feature_size - n;
                                tmp_vec[poly_idx] = weight({out_channel_idx, in_channel_idx, m, n});
                                tmp_vec[poly_idx + HE->polyModulusDegree / 2] = weight({out_channel_idx + out_channels, in_channel_idx, m, n});
                            }
                        }
                    }
                }
                // TODO: NTT
                HE->batchEncoder->encode(tmp_vec, weight_pt({i, j, k}));
            }
        }
    }

    return weight_pt;
}

Tensor<uint64_t> Conv2DNest::PackActivation(Tensor<uint64_t> x) {
    Tensor<uint64_t> ac_msg({tiled_in_channels, HE->polyModulusDegree});

    for (uint64_t i = 0; i < tiled_in_channels; i++) {
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
    }
    // TODO: NTT

    return ac_msg;
}

Tensor<Ciphertext> Conv2DNest::HECompute(Tensor<Plaintext> weight_pt, Tensor<Ciphertext> ac_ct) {
    Tensor<Ciphertext> out_ct({tiled_out_channels}, HE->GenerateZeroCiphertext());
    Tensor<Ciphertext> ac_rot_ct({input_rot, tiled_in_channels}, HE->GenerateZeroCiphertext());
    Tensor<Ciphertext> int_ct({tiled_out_channels, tile_size}, HE->GenerateZeroCiphertext());
    GaloisKeys* keys = HE->galoisKeys;

    // First complete the input rotation
    for (uint64_t i = 0; i < input_rot; i++) {
        for (uint64_t j = 0; j < tiled_in_channels; j++) {
            if (i) {
                HE->evaluator->rotate_rows(ac_rot_ct({i - 1, j}), padded_feature_size * padded_feature_size, *keys, ac_rot_ct({i, j}));
            }
            else {
                ac_rot_ct({i, j}) = ac_ct(j);
            }
        }
    }
    // Complete all the multiplication, and reduce along the input channel dimension
    for (uint64_t i = 0; i < tiled_in_channels; i++) {
        for (uint64_t j = 0; j < tiled_out_channels; j++) {
            for (uint64_t k = 0; k < tile_size; k++) {
                Ciphertext tmp_ct;
                HE->evaluator->multiply_plain(ac_rot_ct({input_rot - 1 - k % input_rot, i}), weight_pt({i, j, k}), tmp_ct);
                if (i) {
                    HE->evaluator->add_inplace(int_ct({j, k}), tmp_ct);
                }
            }
        }
    }
    for (uint64_t i = 0; i < tiled_out_channels; i++) {
        // Reduce along the input rotation dimension, since it has been completed
        for (uint64_t j = 0; j < tile_size; j++) {
            if (j % input_rot) {
                HE->evaluator->add_inplace(int_ct({i, j - j % input_rot}), int_ct({i, j}));
            }
        }
        out_ct(i) = int_ct({i, 0});
        // Complete output rotation to reduce along this dimension
        for (uint64_t j = input_rot; j < tile_size; j += input_rot) {
            HE->evaluator->rotate_rows(out_ct(i), padded_feature_size * padded_feature_size * input_rot, *keys, out_ct(i));
            HE->evaluator->add_inplace(out_ct(i), int_ct({i, j}));
        }
    }

    return out_ct;
}

Tensor<uint64_t> Conv2DNest::DepackResult(Tensor<uint64_t> out_msg) {
    Tensor<uint64_t> y({out_channels, out_feature_size, out_feature_size});

    // TODO: iNTT
    out_channels *= 2;
    for (uint64_t i = 0; i < out_channels; i++) {
        for (uint64_t j = 0; j < out_feature_size; j++) {
            for (uint64_t k = 0; k < out_feature_size; k++) {
                uint64_t offset = stride * out_feature_size * j + stride * k + (kernel_size - 1) * (out_feature_size + 1);
                if (i < out_channels / 2) {
                    y({i, j, k}) = out_msg({i / tile_size, ((tile_size * out_channels - i) % tile_size) * padded_feature_size * padded_feature_size + offset});
                }
                else {
                    y({i, j, k}) = out_msg({(i - out_channels / 2) / tile_size, ((tile_size * out_channels - i + out_channels / 2) % tile_size) * padded_feature_size * padded_feature_size + offset});
                }
            }
        }
    }

    return y;
}

Tensor<uint64_t> Conv2DNest::operator()(Tensor<uint64_t> x) {  // x.shape = {Ci, H, W}
    Tensor<uint64_t> ac_msg = PackActivation(x);  // ac_msg.shape = {ci, N}
    Tensor<Ciphertext> ac_ct = SSToHE(ac_msg, HE);  // ac_ct.shape = {ci}
    Tensor<Ciphertext> out_ct = HECompute(weight_pt, ac_ct);  // out_ct.shape = {co}
    Tensor<uint64_t> out_msg = HEToSS(out_ct, HE);  // out_msg.shape = {co, N}
    Tensor<uint64_t> y = DepackResult(out_msg);  // y.shape = {Co, H, W}

    return y;
};