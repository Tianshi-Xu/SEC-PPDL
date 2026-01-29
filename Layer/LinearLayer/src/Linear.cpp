#include <LinearLayer/Linear.h>
#include <cassert>

using namespace seal;
using namespace HE;
using namespace HE::unified;

namespace LinearLayer {

// Extract shared parameters.
Linear::Linear(uint64_t dim_0, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HE::HEEvaluator* HE)
    : dim_0(dim_0), 
      weight(weight), 
      bias(bias), 
      HE(HE)
{
    std::vector<size_t> weight_shape = weight.shape();
    dim_1 = weight_shape[0];
    dim_2 = weight_shape[1];
};

Linear::Linear(uint64_t dim_0, uint64_t dim_1, uint64_t dim_2, HE::HEEvaluator* HE)
    : dim_0(dim_0),
      dim_1(dim_1),
      dim_2(dim_2),
      HE(HE)
      {
        cout << "---------in MatmulCtpt constructor-----------" << endl;
        cout << "dim_0, dim_1, dim_2: " << dim_0 << ", " << dim_1 << ", " << dim_2 << endl;
        
        if(HE->server) {
            this->weight = Tensor<uint64_t>({dim_1, dim_2});
            this->bias = Tensor<uint64_t>({dim_0, dim_2});
            this->weight.randomize(16);
            this->bias.randomize(16);
        }
      }

LinearBolt::LinearBolt(uint64_t dim_0, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HE::HEEvaluator* HE)
    : Linear(dim_0, weight, bias, HE)
{
    padded_dim_0 = dim_0 - 1;
    for (int i = 0; i < 5; i++) {
        padded_dim_0 |= padded_dim_0 >> (1 << i);
    }
    padded_dim_0 += 1;
    tile_size = HE->polyModulusDegree / padded_dim_0;
    padded_dim_1 = dim_1 - 1;
    padded_dim_1 = padded_dim_1 + tile_size - padded_dim_1 % tile_size;
    tiled_dim_1 = padded_dim_1 / tile_size;
    padded_dim_2 = dim_2 - 1;
    padded_dim_2 = padded_dim_2 + tile_size - padded_dim_2 % tile_size;
    tiled_dim_2 = padded_dim_2 / tile_size;
    input_rot = std::sqrt(tile_size);  // to be checked

    padded_weight = Tensor<uint64_t>({padded_dim_1, padded_dim_2});
    for (uint64_t i = 0; i < weight.size(); i++) {
        padded_weight.data()[(i / dim_2) * padded_dim_2 + i % dim_2] = weight.data()[i];
    }

    if (HE->server) {
        weight_pt = PackWeight();
    }
}

LinearBolt::LinearBolt(uint64_t dim_0, uint64_t dim_1, uint64_t dim_2, HE::HEEvaluator* HE)
    : Linear(dim_0, dim_1, dim_2, HE)
{
    padded_dim_0 = dim_0 - 1;
    for (int i = 0; i < 5; i++) {
        padded_dim_0 |= padded_dim_0 >> (1 << i);
    }
    padded_dim_0 += 1;
    tile_size = HE->polyModulusDegree / padded_dim_0;
    padded_dim_1 = dim_1 - 1;
    padded_dim_1 = padded_dim_1 + tile_size - padded_dim_1 % tile_size;
    tiled_dim_1 = padded_dim_1 / tile_size;
    padded_dim_2 = dim_2 - 1;
    padded_dim_2 = padded_dim_2 + tile_size - padded_dim_2 % tile_size;
    tiled_dim_2 = padded_dim_2 / tile_size;
    input_rot = std::sqrt(tile_size);  // to be checked

    padded_weight = Tensor<uint64_t>({padded_dim_1, padded_dim_2});
    for (uint64_t i = 0; i < weight.size(); i++) {
        padded_weight.data()[(i / dim_2) * padded_dim_2 + i % dim_2] = weight.data()[i];
    }

    if (HE->server) {
        weight_pt = PackWeight();
    }
}

Tensor<UnifiedPlaintext> LinearBolt::PackWeight() {
    Tensor<UnifiedPlaintext> weight_pt({tiled_dim_1, tiled_dim_2, tile_size}, HE->Backend());

    for (uint64_t i = 0; i < tiled_dim_1; i++) {
        for (uint64_t j = 0; j < tiled_dim_2; j++) {
            for (uint64_t k = 0; k < tile_size; k++) {
                std::vector<uint64_t> tmp_vec(HE->polyModulusDegree, 0);
                for (uint64_t l = 0; l < HE->polyModulusDegree / 2; l++) {
                    uint64_t idx_0 = i * tile_size + (l / padded_dim_0 + input_rot - 1 - (k % input_rot)) % tile_size;
                    uint64_t idx_1 = (3 * tile_size - l / padded_dim_0 - input_rot + (k % input_rot) - k) % tile_size;
                    tmp_vec[l] = padded_weight({idx_0, j * tile_size + idx_1});  // pd1 * pd2
                    tmp_vec[l + HE->polyModulusDegree / 2] = padded_weight({idx_0, j * tile_size + idx_1});
                }
                
                bool zero_flag = 1;
                for (uint64_t l = 0; l < HE->polyModulusDegree; l++) {
                    zero_flag = zero_flag && (tmp_vec[l] == 0);
                }
                if (zero_flag) {
                    tmp_vec[HE->polyModulusDegree - 1] = 1; // set unused slots to non-zero values to avoid transparent ciphertext error
                }
                HE->encoder->encode(tmp_vec, weight_pt({i, j, k}));
            }
        }
    }

    return weight_pt;
}

Tensor<uint64_t> LinearBolt::PackActivation(Tensor<uint64_t> &x) {
    Tensor<uint64_t> ac_msg({tiled_dim_1, HE->polyModulusDegree});

    for (uint64_t i = 0; i < tiled_dim_1; i++) {
        for (uint64_t j = 0; j < HE->polyModulusDegree / 2; j++) {
            uint64_t idx = i * tile_size + j / (padded_dim_0 / 2);
            if (idx < dim_1) {
                ac_msg({i, j}) = x({j % (padded_dim_0 / 2), idx});  // to be checked
                if (j % (padded_dim_0 / 2) + padded_dim_0 / 2 < dim_0) {
                    ac_msg({i, j + HE->polyModulusDegree / 2}) = x({j % (padded_dim_0 / 2) + padded_dim_0 / 2, i * tile_size + j / (padded_dim_0 / 2)});
                }
            }
        }
    }

    return ac_msg;
}

Tensor<UnifiedCiphertext> LinearBolt::HECompute(const Tensor<UnifiedPlaintext> &weight_pt, Tensor<UnifiedCiphertext> &ac_ct) {
    /** 
     *  NOTE:
     *  Server computes on HE->Backend()
     *  Client does nothing
     */
    const auto target = HE->server ? HE->Backend() : HOST;
    Tensor<UnifiedCiphertext> out_ct({tiled_dim_2}, HE->GenerateZeroCiphertext(target));

    if (HE->server) {
        Tensor<UnifiedCiphertext> ac_rot_ct({input_rot, tiled_dim_1}, HE->GenerateZeroCiphertext(target));
        Tensor<UnifiedCiphertext> int_ct({tiled_dim_2, tile_size}, HE->GenerateZeroCiphertext(target));
        UnifiedGaloisKeys* keys = HE->galoisKeys;

        // First complete the input rotation
        for (uint64_t i = 0; i < input_rot; i++) {
            for (uint64_t j = 0; j < tiled_dim_1; j++) {
                if (i) {
                    HE->evaluator->rotate_rows(ac_rot_ct({i - 1, j}), padded_dim_0 / 2, *keys, ac_rot_ct({i, j}));
                    
                }
                else {
                    ac_rot_ct({i, j}) = ac_ct(j);
                }
            }
        }
        // Complete all the multiplication, and reduce along the input channel dimension
        for (uint64_t i = 0; i < tiled_dim_1; i++) {
            for (uint64_t j = 0; j < tiled_dim_2; j++) {
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
        for (uint64_t i = 0; i < tiled_dim_2; i++) {
            // Reduce along the input rotation dimension, since it has been completed
            for (uint64_t j = 0; j < tile_size; j++) {
                if (j % input_rot) {
                    HE->evaluator->add_inplace(int_ct({i, j - j % input_rot}), int_ct({i, j}));
                }
            }
            out_ct(i) = int_ct({i, 0});
            // Complete output rotation to reduce along this dimension
            for (uint64_t j = input_rot; j < tile_size; j += input_rot) {
                HE->evaluator->rotate_rows(out_ct(i), padded_dim_0 * input_rot / 2, *keys, out_ct(i));
                
                HE->evaluator->add_inplace(out_ct(i), int_ct({i, j}));
            }
        }
    }

    return out_ct;
}

Tensor<uint64_t> LinearBolt::DepackResult(Tensor<uint64_t> &out_msg) {
    Tensor<uint64_t> y({dim_0, dim_2});
    for (uint64_t i = 0; i < tiled_dim_2; i++) {
        for (uint64_t j = 0; j < HE->polyModulusDegree; j++) {
            // Fix: Add modulo tile_size to ensure slot index is in range [0, tile_size)
            uint64_t idx = i * tile_size + ((tile_size - j / (padded_dim_0 / 2) % tile_size) % tile_size);
            if (idx < dim_2) {
                if (j < HE->polyModulusDegree / 2) {
                    y({j % (padded_dim_0 / 2), idx}) = out_msg({i, j});
                }
                else if (j % (padded_dim_0 / 2) + padded_dim_0 / 2 < dim_0) {
                    y({j % (padded_dim_0 / 2) + padded_dim_0 / 2, idx}) = out_msg({i, j});
                }
            }
        }
    }

    return y;
}

Tensor<uint64_t> LinearBolt::operator()(Tensor<uint64_t> &x) {
    // std::cout << "MatmulCtptBolt operator called" << std::endl;
    Tensor<uint64_t> ac_msg = PackActivation(x);
    // std::cout << "ac_msg generated" << std::endl;
    Tensor<UnifiedCiphertext> ac_ct = Operator::SSToHE(ac_msg, HE);
    // std::cout << "ac_ct generated" << std::endl;
    Tensor<UnifiedCiphertext> out_ct = HECompute(weight_pt, ac_ct);
    Tensor<uint64_t> out_msg = Operator::HEToSS(out_ct, HE);
    // std::cout << "out_msg generated" << std::endl;
    Tensor<uint64_t> y = DepackResult(out_msg);
    // std::cout << "y generated" << std::endl;

    return y;
};

LinearNest::LinearNest(uint64_t dim_0, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HE::HEEvaluator* HE)
    : Linear(dim_0, weight, bias, HE)
{
    compute_he_params();
    if (HE->server) {
        weight_pt = PackWeight();
    }
}

LinearNest::LinearNest(uint64_t dim_0, uint64_t dim_1, uint64_t dim_2, HE::HEEvaluator* HE)
    : Linear(dim_0, dim_1, dim_2, HE)
{
    compute_he_params();
    if (HE->server) {
        weight_pt = PackWeight();
    }
}

void LinearNest::compute_he_params() {
    padded_dim_0 = dim_0 - 1;
    for (int i = 0; i < 5; i++) {
        padded_dim_0 |= padded_dim_0 >> (1 << i);
    }
    padded_dim_0 += 1;
    tile_size = HE->polyModulusDegree / padded_dim_0;
    padded_dim_1 = dim_1 - 1;
    padded_dim_1 = padded_dim_1 + tile_size - padded_dim_1 % tile_size;
    tiled_dim_1 = padded_dim_1 / tile_size;
    padded_dim_2 = dim_2 - 1;
    padded_dim_2 = padded_dim_2 + tile_size - padded_dim_2 % tile_size;
    tiled_dim_2 = padded_dim_2 / tile_size;
    input_rot = std::sqrt(tile_size);

    padded_weight = Tensor<uint64_t>({padded_dim_1, padded_dim_2});
    for (uint64_t i = 0; i < weight.size(); i++) {
        padded_weight.data()[(i / dim_2) * padded_dim_2 + i % dim_2] = weight.data()[i];
    }
}

Tensor<UnifiedPlaintext> LinearNest::PackWeight() {
    Tensor<UnifiedPlaintext> weight_pt({tiled_dim_1, tiled_dim_2, tile_size}, HE->Backend());

    for (uint64_t i = 0; i < tiled_dim_1; i++) {
        for (uint64_t j = 0; j < tiled_dim_2; j++) {
            for (uint64_t k = 0; k < tile_size; k++) {
                std::vector<uint64_t> tmp_vec(HE->polyModulusDegree, 0);
                for (uint64_t l = 0; l < HE->polyModulusDegree / 2; l++) {
                    uint64_t idx_0 = i * tile_size + (l / padded_dim_0 + input_rot - 1 - (k % input_rot)) % tile_size;
                    uint64_t idx_1 = (3 * tile_size - l / padded_dim_0 - input_rot + (k % input_rot) - k) % tile_size;
                    tmp_vec[l] = padded_weight({idx_0, j * tile_size + idx_1});
                    tmp_vec[l + HE->polyModulusDegree / 2] = padded_weight({idx_0, j * tile_size + idx_1});
                }

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

Tensor<uint64_t> LinearNest::PackActivation(Tensor<uint64_t> &x) {
    Tensor<uint64_t> ac_msg({tiled_dim_1, HE->polyModulusDegree});

    for (uint64_t i = 0; i < tiled_dim_1; i++) {
        for (uint64_t j = 0; j < HE->polyModulusDegree / 2; j++) {
            uint64_t idx = i * tile_size + j / (padded_dim_0 / 2);
            if (idx < dim_1) {
                ac_msg({i, j}) = x({j % (padded_dim_0 / 2), idx});
                if (j % (padded_dim_0 / 2) + padded_dim_0 / 2 < dim_0) {
                    ac_msg({i, j + HE->polyModulusDegree / 2}) = x({j % (padded_dim_0 / 2) + padded_dim_0 / 2, i * tile_size + j / (padded_dim_0 / 2)});
                }
            }
        }
    }

    return ac_msg;
}

Tensor<UnifiedCiphertext> LinearNest::HECompute(const Tensor<UnifiedPlaintext> &weight_pt, Tensor<UnifiedCiphertext> &ac_ct) {
    const auto target = HE->server ? HE->Backend() : HOST;
    Tensor<UnifiedCiphertext> out_ct({tiled_dim_2}, HE->GenerateZeroCiphertext(target));

    if (HE->server) {
        Tensor<UnifiedCiphertext> ac_rot_ct({input_rot, tiled_dim_1}, HE->GenerateZeroCiphertext(target));
        Tensor<UnifiedCiphertext> int_ct({tiled_dim_2, tile_size}, HE->GenerateZeroCiphertext(target));
        UnifiedGaloisKeys* keys = HE->galoisKeys;

        for (uint64_t i = 0; i < input_rot; i++) {
            for (uint64_t j = 0; j < tiled_dim_1; j++) {
                if (i) {
                    HE->evaluator->rotate_rows(ac_rot_ct({i - 1, j}), padded_dim_0 / 2, *keys, ac_rot_ct({i, j}));
                }
                else {
                    ac_rot_ct({i, j}) = ac_ct(j);
                }
            }
        }
        for (uint64_t i = 0; i < tiled_dim_1; i++) {
            for (uint64_t j = 0; j < tiled_dim_2; j++) {
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
        for (uint64_t i = 0; i < tiled_dim_2; i++) {
            for (uint64_t j = 0; j < tile_size; j++) {
                if (j % input_rot) {
                    HE->evaluator->add_inplace(int_ct({i, j - j % input_rot}), int_ct({i, j}));
                }
            }
            out_ct(i) = int_ct({i, 0});
            for (uint64_t j = input_rot; j < tile_size; j += input_rot) {
                HE->evaluator->rotate_rows(out_ct(i), padded_dim_0 * input_rot / 2, *keys, out_ct(i));
                HE->evaluator->add_inplace(out_ct(i), int_ct({i, j}));
            }
        }
    }

    return out_ct;
}

Tensor<uint64_t> LinearNest::DepackResult(Tensor<uint64_t> &out_msg) {
    Tensor<uint64_t> y({dim_0, dim_2});
    
    for (uint64_t i = 0; i < tiled_dim_2; i++) {
        for (uint64_t j = 0; j < HE->polyModulusDegree; j++) {
            // Fix: Add modulo tile_size to ensure slot index is in range [0, tile_size)
            uint64_t idx = i * tile_size + ((tile_size - j / (padded_dim_0 / 2) % tile_size) % tile_size);
            if (idx < dim_2) {
                if (j < HE->polyModulusDegree / 2) {
                    y({j % (padded_dim_0 / 2), idx}) = out_msg({i, j});
                }
                else if (j % (padded_dim_0 / 2) + padded_dim_0 / 2 < dim_0) {
                    y({j % (padded_dim_0 / 2) + padded_dim_0 / 2, idx}) = out_msg({i, j});
                }
            }
        }
    }

    return y;
}

Tensor<uint64_t> LinearNest::operator()(Tensor<uint64_t> &x) {
    // std::cout << "MatmulCtptNest operator called" << std::endl;
    Tensor<uint64_t> ac_msg = PackActivation(x);
    Tensor<UnifiedCiphertext> ac_ct = Operator::SSToHE(ac_msg, HE);
    Tensor<UnifiedCiphertext> out_ct = HECompute(weight_pt, ac_ct);
    Tensor<uint64_t> out_msg = Operator::HEToSS(out_ct, HE);
    Tensor<uint64_t> y = DepackResult(out_msg);

    return y;
};

MatmulCtctBumble::MatmulCtctBumble(HE::HEEvaluator* HE)
    : HE(HE) {};

Tensor<uint64_t> MatmulCtctBumble::Transpose(Tensor<uint64_t> &x) {
    std::vector<uint64_t> shape = x.shape();
    Tensor<uint64_t> x_t({shape[1], shape[0]});

    for (uint64_t i = 0; i < shape[0]; i++) {
        for (uint64_t j = 0; j < shape[1]; j++) {
            x_t({j, i}) = x({i, j});
        }
    }

    return x_t;
}

Tensor<uint64_t> MatmulCtctBumble::MatmulPtpt(Tensor<uint64_t> &x1, Tensor<uint64_t> &x2) {
    uint64_t dim_0 = x1.shape()[0];
    uint64_t dim_1 = x1.shape()[1];
    uint64_t dim_2 = x2.shape()[1];
    Tensor<uint64_t> y({dim_0, dim_2});

    for (uint64_t i = 0; i < dim_0; i++) {
        for (uint64_t j = 0; j < dim_2; j++) {
            for (uint64_t k = 0; k < dim_1; k++) {
                y({i, j}) += x1({i, k}) * x2({k, j});
            }
        }
    }

    return y;
}

Tensor<uint64_t> MatmulCtctBumble::operator()(Tensor<uint64_t> &x1, Tensor<uint64_t> &x2) {
    // std::cout << "MatmulCtctBumble operator called" << std::endl;

    Tensor<uint64_t> local_term = MatmulPtpt(x1, x2);
    Tensor<uint64_t> x1_t = this->Transpose(x1);
    Tensor<uint64_t> x2_t = this->Transpose(x2);
    // std::cout << "Plaintext mult completed" << std::endl;
    LinearBolt* ctpt_mult_1 = new LinearBolt(x1.shape()[0], x2, x2, this->HE);
    Tensor<uint64_t> cross_term_1 = ctpt_mult_1->operator()(x1);
    // std::cout << "Cross term 1 generated" << std::endl;
    LinearBolt* ctpt_mult_2 = new LinearBolt(x2_t.shape()[0], x1_t, x1_t, this->HE);
    Tensor<uint64_t> cross_term_2_t = ctpt_mult_2->operator()(x2_t);
    // std::cout << "Cross term 2 generated" << std::endl;
    Tensor<uint64_t> cross_term_2 = this->Transpose(cross_term_2_t);
    Tensor<uint64_t> y(local_term.shape());
    for (uint64_t i = 0; i < local_term.size(); i++) {
        if (HE->server) {
            y(i) = cross_term_1(i) + cross_term_2(i) - local_term(i);
        }
        else {
            y(i) = cross_term_1(i) + cross_term_2(i) + local_term(i);
        }
    }
    std::cout << "y generated" << std::endl;

    return y;
};

} // namespace LinearLayer

