#include <seal/seal.h>
#include "datatype/Tensor.h"
#include "HE/HE.h"
#include "utils/io.h"

/*
 * dependency on io and he:
 * io: IO.send_cipher_vector/recv_cipher_vector
 * he: HE.encryptor/decryptor/batch_encoder/poly_degree/prime_mod
*/


// let the last dimension of x be N, the polynomial degree
Tensor<Ciphertext> SSToHE(Tensor<int> x) {
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
        HE.batch_encoder.encode(tmp_vec, ac_pt(i));
    }

    // communication and addition
    if (party == ALICE){
        ac_ct = IO.recv_cipher_vector();
        assert(ac_pt.size() == ac_ct.size() && "Number of polys does not match.");
        for (size_t i = 0; i < ac_ct.size(); i++) {
            HE.evaluator.add_plain_inplace(ac_ct(i), ac_pt(i));
        }
    } 
    else{
        for (size_t i = 0; i < ac_pt.size(); i++) {
            HE.encryptor.encrypt(ac_pt(i), ac_ct(i));
        }
        IO.send_cipher_vector(ac_ct);
    }

    return ac_ct;
};


Tensor<int> HEToSS(Tensor<Ciphertext> out_ct) {
    std::vector<size_t> scalar_shape = out_ct.shape().push_back(HE.poly_degree);
    Tensor<int> x(scalar_shape);
    Tensor<Plaintext> out_share(out_ct.shape());
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> distrib(0, HE.prime_mod - 1);

    // mask generation and communication
    if (party == ALICE){
        for (size_t i = 0; i < out_ct.size(); i++){
            std::vector<uint64_t> pos_mask(HE.poly_degree, 0);
            std::vector<uint64_t> neg_mask(HE.poly_degree, 0);
            for (size_t j = 0; j < pos_mask.size(); j++) {
                pos_mask[j] = distrib(gen);
                neg_mask[j] = HE.prime_mod - pos_mask[j];
            }
            // TODO: noise flooding (add freshly encrypted zero), refer to Cheetah
            Plaintext temp_buffer_pos, temp_buffer_neg;
            HE.batch_encoder.encode(pos_mask, temp_buffer_pos);
            HE.batch_encoder.encode(neg_mask, temp_buffer_neg);
            HE.evaluator.add_plain_inplace(out_ct(i), temp_buffer_neg);
            out_share(i) = temp_buffer_pos;
        }
        IO.send_cipher_vector(out_ct);
    }
    else {
        out_ct = IO.recv_cipher_vector();
    }

    // decoding and decryption
    std::vector<int> tmp_vec(HE.poly_degree);
    if (party == ALICE) {
        for (size_t i = 0; i < out_share.size(); i++) {
            HE.batch_encoder.decode(out_share(i), tmp_vec);
            for (size_t j = 0; j < HE.poly_degree; j++) {
                x(i * HE.poly_degree + j) = tmp_vec[j];
            }
        }
    }
    else {
        for (size_t i = 0; i < out_ct.size(); i++) {
            Plaintext out_pt;
            HE.decryptor.decrypt(out_ct(i), out_pt);
            HE.batch_encoder.decode(out_pt, tmp_vec);
            for (size_t j = 0; j < HE.poly_degree; j++) {
                x(i * HE.poly_degree + j) = tmp_vec[j];
            }
        }
    }
    
    x.reshape(scalar_shape);
    return x;
};