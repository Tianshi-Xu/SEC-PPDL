#include <LinearLayer/Conv.h>
#include <cassert>

using namespace seal;
using namespace LinearLayer;
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
    //Tensor<uint64_t> ac_msg = PackActivation(x);  // ac_msg.shape = {ci, N}
    //Tensor<Ciphertext> ac_ct = SSToHE(ac_msg, HE);  // ac_ct.shape = {ci}
    //Tensor<Ciphertext> out_ct = HECompute(weight_pt, ac_ct);  // out_ct.shape = {co}
    //Tensor<uint64_t> out_msg = HEToSS(out_ct, HE);  // out_msg.shape = {co, N}
    //Tensor<uint64_t> y = DepackResult(out_msg);  // y.shape = {Co, H, W}
    //mistake
    return x;
};






// // 计算上取整除法
// int Conv2DCheetah::div_upper(int a, int b) {
//     return ((a + b - 1) / b);
// }

// // 计算计算开销
// int Conv2DCheetah::calculate_cost(int H, int W, int h, int Hw, int Ww, int C, int N) {
//     return (int)ceil((double)C / (N / (Hw * Ww))) *
//            (int)ceil((double)(H - h + 1) / (Hw - h + 1)) *
//            (int)ceil((double)(W - h + 1) / (Ww - h + 1));
// }

// // 查找最佳分块方式
// void Conv2DCheetah::find_optimal_partition(int H, int W, int h, int C, int N, int* optimal_Hw, int* optimal_Ww) {
//     int min_cost = (1 << 30);
//     for (int Hw = h; Hw <= H; Hw++) {
//         for (int Ww = h; Ww <= W; Ww++) {
//             if (Hw * Ww > N) continue;
//             int cost = calculate_cost(H, W, h, Hw, Ww, C, N);
//             if (cost < min_cost) {
//                 min_cost = cost;
//                 *optimal_Hw = Hw;
//                 *optimal_Ww = Ww;
//             }
//         }
//     }
// }

// // 构造函数
// Conv2DCheetah::Conv2DCheetah(int in_channels, int out_channels, int kernel_size, int stride, int padding, HEEvaluator* he, Tensor<int> inputTensor, Tensor<int> kernel)
//     : Conv2D(in_channels, out_channels, kernel_size, stride, padding, he, inputTensor, kernel) {

//     C = out_channels;
//     M = in_channels;
//     s = stride;
//     h = kernel_size;

//     const std::vector<size_t>& inputTensorShape = inputTensor.shape();
//     H = inputTensorShape[0];
//     W = inputTensorShape[1];

//     int optimal_Hw = 0, optimal_Ww = 0;
//     find_optimal_partition(H, W, h, C, N, &optimal_Hw, &optimal_Ww);
    
//     HW = optimal_Hw;
//     WW = optimal_Ww;
//     CW = 2;
//     MW = 2;
//     dM = div_upper(M, MW);
//     dC = div_upper(C, CW);
//     dH = div_upper(H - h + 1, HW - h + 1);
//     dW = div_upper(W - h + 1, WW - h + 1);
//     OW = HW * WW * (MW * CW - 1) + WW * (h - 1) + h - 1;
//     Hprime = (H - h + s) / s;
//     Wprime = (W - h + s) / s;
//     HWprime = (HW - h + s) / s;
//     WWprime = (WW - h + s) / s;
//     polyModulusDegree = he->polyModulusDegree;
//     plain = he->plain;
// }

// // 加密张量
// Tensor<seal::Ciphertext> Conv2DCheetah::EncryptTensor(Tensor<seal::Plaintext> plainTensor) {
//     std::vector<size_t> shapeTab = {dC, dH, dW};
//     Tensor<seal::Ciphertext> TalphabetaCipher(shapeTab);

//     for (unsigned long gama = 0; gama < dC; gama++) {
//         for (unsigned long alpha = 0; alpha < dH; alpha++) {
//             for (unsigned long beta = 0; beta < dW; beta++) {
//                 he->encryptor->encrypt(plainTensor({gama, alpha, beta}), TalphabetaCipher({gama, alpha, beta}));
//             }
//         }
//     }
//     return TalphabetaCipher;
// }

// // 计算输入张量的 Pack 版本
// Tensor<seal::Plaintext> Conv2DCheetah::PackTensor(Tensor<int> x) {
//     std::vector<size_t> shapeTab = {dC, dH, dW};
//     Tensor<seal::Plaintext> Talphabeta(shapeTab);
    
//     for (unsigned long gama = 0; gama < dC; gama++) {
//         for (unsigned long alpha = 0; alpha < dH; alpha++) {
//             for (unsigned long beta = 0; beta < dW; beta++) {
//                 Talphabeta({gama, alpha, beta}).resize(polyModulusDegree);
//                 seal::util::modulo_poly_coeffs(x.flatten().data(), CW * HW * WW, plain, Talphabeta({gama, alpha, beta}).data());
//                 std::fill_n(Talphabeta({gama, alpha, beta}).data() + CW * HW * WW, polyModulusDegree - CW * HW * WW, 0);
//             }
//         }
//     }
//     return Talphabeta;
// }

// // 计算卷积核的 Pack 版本
// Tensor<seal::Plaintext> Conv2DCheetah::PackKernel(Tensor<int> x) {
//     std::vector<size_t> shapeTab = {dM, dC};
//     Tensor<seal::Plaintext> Ktg(shapeTab);

//     for (unsigned long theta = 0; theta < dM; theta++) {
//         for (unsigned long gama = 0; gama < dC; gama++) {
//             Ktg({theta, gama}).resize(polyModulusDegree);
//             seal::util::modulo_poly_coeffs(x.flatten().data(), MW * CW * h * h, plain, Ktg({theta, gama}).data());
//             std::fill_n(Ktg({theta, gama}).data() + MW * CW * h * h, polyModulusDegree - MW * CW * h * h, 0);
//         }
//     }
//     return Ktg;
// }

// // 计算同态卷积
// Tensor<seal::Ciphertext> Conv2DCheetah::Conv(Tensor<seal::Ciphertext> T, Tensor<seal::Plaintext> K) {
//     std::vector<size_t> shapeTab = {dM, dH, dW};
//     Tensor<seal::Ciphertext> ConvRe(shapeTab);
//     seal::Ciphertext interm;

//     for (size_t theta = 0; theta < dM; theta++) {
//         for (size_t alpha = 0; alpha < dH; alpha++) {
//             for (size_t beta = 0; beta < dW; beta++) {
//                 he->evaluator->multiply_plain(T({0, alpha, beta}), K({theta, 0}), ConvRe({theta, alpha, beta}));
//                 for (size_t gama = 1; gama < dC; gama++) {
//                     he->evaluator->multiply_plain(T({gama, alpha, beta}), K({theta, gama}), interm);
//                     he->evaluator->add_inplace(ConvRe({theta, alpha, beta}), interm);
//                 }
//             }
//         }
//     }
//     return ConvRe;
// }

//  // namespace LinearLayer