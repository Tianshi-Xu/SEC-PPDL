#include <LinearOperator/Conversion.h>

using namespace HE::unified;

namespace Operator {

// input mod prime, output mod q, need a ring2field conversion before it
Tensor<UnifiedCiphertext> SSToHE(const Tensor<uint64_t> &x, HE::HEEvaluator* HE) {
    std::vector<size_t> scalar_shape = x.shape();
    uint64_t poly_degree = scalar_shape[scalar_shape.size() - 1];
    std::vector<size_t> poly_shape(scalar_shape.begin(), scalar_shape.end() - 1);
    std::vector<uint64_t> tmp_vec(poly_degree,0ULL);
    // encoding
    Tensor<UnifiedPlaintext> ac_pt(poly_shape, HE->server ? HE->Backend() : HOST);
    Tensor<UnifiedCiphertext> ac_ct(poly_shape, HOST /* Communication can be only done on HOST */);
    HE->encoder->encode(tmp_vec, ac_pt(0));
    for (size_t i = 0; i < ac_pt.size(); i++) {
        for (size_t j = 0; j < poly_degree; j++) {
            tmp_vec[j] = x(i * poly_degree + j);
        }
        HE->encoder->encode(tmp_vec, ac_pt(i));
    }
    if (HE->server){
        HE->ReceiveEncVec(ac_ct);
        if (HE->Backend() == DEVICE){
            ac_ct.apply([HE](UnifiedCiphertext &ct){
                ct.to_device(*HE->context);
            });
        }
        assert(ac_pt.size() == ac_ct.size() && "Number of polys does not match.");
        for (size_t i = 0; i < ac_ct.size(); i++) {
            HE->evaluator->add_plain_inplace(ac_ct(i), ac_pt(i));
        }
    } 
    else { /* client */
        for (size_t i = 0; i < ac_pt.size(); i++) {
            HE->encryptor->encrypt(ac_pt(i), ac_ct(i));
        }
        HE->SendEncVec(ac_ct);
    }
    return ac_ct;
};

// input mod q, output mod prime, need a field2ring conversion after it to support ring MPC protocols
Tensor<uint64_t> HEToSS(Tensor<UnifiedCiphertext> out_ct, HE::HEEvaluator* HE) {
    std::vector<size_t> scalar_shape = out_ct.shape();
    scalar_shape.push_back(HE->polyModulusDegree);
    Tensor<uint64_t> x(scalar_shape);
    Tensor<UnifiedPlaintext> out_share(out_ct.shape(), HOST);
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
                if (HE->server) {
                    x(i * HE->polyModulusDegree + j) = pos_mask[j];
                }
            }
            // TODO: noise flooding (add freshly encrypted zero), refer to Cheetah
            UnifiedPlaintext tmp_pos(HOST);
            UnifiedPlaintext tmp_neg(HOST);
            HE->encoder->encode(pos_mask, tmp_pos);
            HE->encoder->encode(neg_mask, tmp_neg);
            // HE->evaluator->add_plain_inplace(out_ct(i), tmp_neg);  // annotate this when testing
            out_share(i) = tmp_pos;
        }
        out_ct.apply([HE](UnifiedCiphertext &ct){
            if (HE->Backend() == DEVICE) {
                ct.to_host(*HE->context);
            }
        });
        HE->SendEncVec(out_ct);
    }
    else {
        HE->ReceiveEncVec(out_ct);
    }

    // decoding and decryption
    std::vector<uint64_t> tmp_vec(HE->polyModulusDegree);
    if (HE->server) {
        // for (size_t i = 0; i < out_share.size(); i++) {
        //     // HE->batchEncoder->decode(out_share(i), tmp_vec);     * SEAL does not allow adjacent encoding and decoding?
        //     for (size_t j = 0; j < HE->polyModulusDegree; j++) {
        //         x(i * HE->polyModulusDegree + j) = tmp_vec[j];
        //     }
        // }
    }
    else {
        for (size_t i = 0; i < out_ct.size(); i++) {
            Plaintext out_pt;
            HE->decryptor->decrypt(out_ct(i), out_pt);
            HE->encoder->decode(out_pt, tmp_vec);
            for (size_t j = 0; j < HE->polyModulusDegree; j++) {
                x(i * HE->polyModulusDegree + j) = tmp_vec[j];
            }
        }
    }

    x.reshape(scalar_shape);
    return x;
};


Tensor<HE::unified::UnifiedCiphertext> SSToHE_coeff(const Tensor<uint64_t> &x, HE::HEEvaluator* HE)
{
    std::vector<size_t> shapeTab = x.shape();
    auto polyModulusDegree = HE->polyModulusDegree;
    auto plain = HE->plain_mod;
    size_t numPoly = 1;
    for (int num : shapeTab) {
        numPoly *= num;
    }

    int len = shapeTab.back(); 
    shapeTab.pop_back();
    numPoly /= len;
    Tensor<UnifiedPlaintext> T(shapeTab,Datatype::HOST);
    for (size_t i = 0; i < numPoly; i++){
        vector<uint64_t> Tsubv(len, 0);
        for (size_t j = 0; j < len; j++){
            Tsubv[j] = x(i * len + j);
        }
        T(i).hplain().resize(polyModulusDegree);
        seal::util::modulo_poly_coeffs(Tsubv, len, plain, T(i).hplain().data());
        std::fill_n(T(i).hplain().data() + len, polyModulusDegree - len, 0);
    }
    Tensor<UnifiedCiphertext> finalpack(shapeTab, HOST);
    if (!HE->server){
        //客户端
        for (size_t i = 0; i < numPoly; i++){
            HE->encryptor->encrypt(T(i), finalpack(i));
        }
        // enc.flatten();
        HE->SendEncVec(finalpack);
    }else{
        //服务器端
        HE->ReceiveEncVec(finalpack);
        if (HE->Backend() == DEVICE){
            finalpack.apply([HE](UnifiedCiphertext &ct){
                ct.to_device(*HE->context);
            });
            T.apply([HE](UnifiedPlaintext &pt){
                pt.to_device(*HE->context);
            });
        }
        for (size_t i = 0; i < numPoly; i++){
            HE->evaluator->add_plain_inplace(finalpack(i), T(i));
        }
    }
    return finalpack;
}


Tensor<uint64_t> HEToSS_coeff(Tensor<HE::unified::UnifiedCiphertext> out_ct, HE::HEEvaluator* HE)
{
    auto shapeTab = out_ct.shape();
    Tensor<UnifiedCiphertext> cipherMask(shapeTab,HE->GenerateZeroCiphertext());
    Tensor<UnifiedPlaintext> plainMask(shapeTab,HOST);
    size_t numPoly = 1;
    for (int num : shapeTab) {
        numPoly *= num;
    }
    auto tensorShapeTab = shapeTab;
    tensorShapeTab.push_back(HE->polyModulusDegree);

    Tensor<uint64_t> tensorMask(tensorShapeTab, 0);
    UnifiedPlaintext plainMaskInv(HOST);
    if (HE->server) {
        int64_t mask;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int64_t> dist(0, HE->plain_mod - 1);
        for (size_t i = 0; i < numPoly; i++){
            plainMask(i).hplain().resize(HE->polyModulusDegree);
            plainMaskInv.hplain().resize(HE->polyModulusDegree);
            for (size_t l = 0; l < HE->polyModulusDegree; l++){
                mask = dist(gen);
                *(plainMask(i).hplain().data() + l) = mask;
                tensorMask((i) * HE->polyModulusDegree + l) = mask;
                mask = HE->plain_mod - mask;
                *(plainMaskInv.hplain().data() + l) = mask;   
            }
            HE->evaluator->add_plain(out_ct(i), plainMaskInv, cipherMask(i));
        }
        cipherMask.flatten();
        if (HE->Backend() == DEVICE){
            cipherMask.apply([HE](UnifiedCiphertext &ct){
                ct.to_host(*HE->context);
            });
        }
        HE->SendEncVec(cipherMask);
        return tensorMask;

    }else{
        HE->ReceiveEncVec(cipherMask);
        for (size_t i = 0; i < numPoly; i++){
            HE->decryptor->decrypt(cipherMask(i), plainMask(i));
            for (size_t j = 0; j < HE->polyModulusDegree; j++){
                tensorMask(i * HE->polyModulusDegree + j) = *(plainMask(i).hplain().data() + j);
            }
        }
        return tensorMask;
    }
}

} // namespace Operator
