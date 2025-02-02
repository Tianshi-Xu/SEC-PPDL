#include <seal/seal.h>
#include "datatype/Tensor.h"
#include "HE/HE.h"
#include "utils/io.h"


// let the last dimension of x be N, the polynomial degree
Tensor<Ciphertext> SSToHE(Tensor<uint64_t> x, HEEvaluator* HE) {
    std::vector<size_t> scalar_shape = x.shape();
    uint64_t poly_degree = scalar_shape[scalar_shape.size() - 1];
    std::vector<size_t> poly_shape(scalar_shape.begin(), scalar_shape.end() - 1);
    Tensor<Plaintext> ac_pt(poly_shape);
    Tensor<Ciphertext> ac_ct(poly_shape);
    std::vector<uint64_t> tmp_vec(poly_degree);

    // encoding
    for (size_t i = 0; i < ac_pt.size(); i++) {
        for (size_t j = 0; j < poly_degree; j++) {
            tmp_vec[j] = x({i * poly_degree + j});
        }
        HE->batchEncoder->encode(tmp_vec, ac_pt(i));
    }

    // communication and addition
    if (HE->server){
        HE->ReceiveEncVec(ac_ct);
        assert(ac_pt.size() == ac_ct.size() && "Number of polys does not match.");
        for (size_t i = 0; i < ac_ct.size(); i++) {
            HE->evaluator->add_plain_inplace(ac_ct(i), ac_pt(i));
        }
    } 
    else{
        for (size_t i = 0; i < ac_pt.size(); i++) {
            HE->encryptor->encrypt(ac_pt(i), ac_ct(i));
        }
        HE->SendEncVec(ac_ct);
    }

    return ac_ct;
};


Tensor<uint64_t> HEToSS(Tensor<Ciphertext> out_ct, HEEvaluator* HE) {
    std::vector<size_t> scalar_shape = out_ct.shape();
    scalar_shape.push_back(HE->polyModulusDegree);
    Tensor<uint64_t> x(scalar_shape);
    Tensor<Plaintext> out_share(out_ct.shape());
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> distrib(0, HE->plain_mod - 1);

    // mask generation and communication
    if (HE->server) {
        for (size_t i = 0; i < out_ct.size(); i++){
            std::vector<uint64_t> pos_mask(HE->polyModulusDegree, 0);
            std::vector<uint64_t> neg_mask(HE->polyModulusDegree, 0);
            for (size_t j = 0; j < pos_mask.size(); j++) {
                pos_mask[j] = distrib(gen);
                neg_mask[j] = HE->plain_mod - pos_mask[j];
            }
            // TODO: noise flooding (add freshly encrypted zero), refer to Cheetah
            Plaintext tmp_pos, tmp_neg;
            HE->batchEncoder->encode(pos_mask, tmp_pos);
            HE->batchEncoder->encode(neg_mask, tmp_neg);
            HE->evaluator->add_plain_inplace(out_ct(i), tmp_neg);
            out_share(i) = tmp_pos;
        }
        HE->SendEncVec(out_ct);
    }
    else {
        HE->ReceiveEncVec(out_ct);
    }

    // decoding and decryption
    std::vector<uint64_t> tmp_vec(HE->polyModulusDegree);
    if (HE->server) {
        for (size_t i = 0; i < out_share.size(); i++) {
            HE->batchEncoder->decode(out_share(i), tmp_vec);
            for (size_t j = 0; j < HE->polyModulusDegree; j++) {
                x(i * HE->polyModulusDegree + j) = tmp_vec[j];
            }
        }
    }
    else {
        for (size_t i = 0; i < out_ct.size(); i++) {
            Plaintext out_pt;
            HE->decryptor->decrypt(out_ct(i), out_pt);
            HE->batchEncoder->decode(out_pt, tmp_vec);
            for (size_t j = 0; j < HE->polyModulusDegree; j++) {
                x(i * HE->polyModulusDegree + j) = tmp_vec[j];
            }
        }
    }

    x.reshape(scalar_shape);
    return x;
};