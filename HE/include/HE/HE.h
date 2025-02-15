#include <seal/seal.h>
#include <vector>
#include <HE/NetIO.h>
#include <HE/unified/UnifiedEvk.h>
#include "HE/unified/UnifiedEncoder.h"
#include <HE/unified/UnifiedEvaluator.h>

#pragma once

using namespace std;
using namespace seal;
using namespace seal::util;
using namespace Datatype;
namespace HE {
class HEEvaluator {
    public:
    LOCATION backend = LOCATION::UNDEF;
    unified::UnifiedContext *context = nullptr;
    Encryptor *encryptor = nullptr;
    Decryptor *decryptor = nullptr;
    unified::UnifiedBatchEncoder *encoder = nullptr;
    unified::UnifiedEvaluator *evaluator = nullptr;
    RelinKeys *relinKeys = nullptr;
    unified::UnifiedGaloisKeys *galoisKeys = nullptr;
    unified::UnifiedBatchEncoder *batchEncoder = nullptr;
    PublicKey *publicKeys = nullptr;
    SecretKey *secretKeys= nullptr;
    bool server = false;
    NetIO *IO = nullptr;
    uint64_t polyModulusDegree = 8192;
    uint64_t plain_mod = 0;

    HEEvaluator(
        NetIO &IO,
        bool server,
        LOCATION backend = HOST
    ){
        this->IO = &IO;
        this->server = server;
        if (backend == LOCATION::HOST_AND_DEVICE) {
            throw std::invalid_argument("Currently not supported");
        }
        this->backend = backend;
    }

    void GenerateNewKey() {
        publicKeys = new PublicKey();
        secretKeys = new SecretKey();
        relinKeys  = new RelinKeys();
        galoisKeys = new unified::UnifiedGaloisKeys(HOST);
        EncryptionParameters parms(scheme_type::bfv);
        context = new unified::UnifiedContext(polyModulusDegree, 20, backend);
        encoder = new unified::UnifiedBatchEncoder(*context);
        evaluator = new unified::UnifiedEvaluator(*context);
        batchEncoder = new unified::UnifiedBatchEncoder(*context);
        plain_mod = parms.plain_modulus().value();
        if (server) {
            uint64_t pk_sze{0};
            uint64_t gk_sze{0};
            this->IO->recv_data(&pk_sze, sizeof(uint64_t));
            this->IO->recv_data(&gk_sze, sizeof(uint64_t));
            char *key_buf = new char[pk_sze + gk_sze];
            this->IO->recv_data(key_buf, pk_sze + gk_sze);
            std::stringstream is;
            is.write(key_buf, pk_sze);
            publicKeys->load(context->hcontext(), is);
            is.write(key_buf + pk_sze, gk_sze);
            galoisKeys->hgalois().load(context->hcontext(), is);

            if (IsGPUenable()) {
                // Load Galois Keys to GPU
                galoisKeys->to_device(*context);
            }

            std::cout << "Server received: " << key_buf << "\n";
            std::cout << "Server received: " << pk_sze << "\n";
            encryptor = new Encryptor(*context, *publicKeys);
            delete[] key_buf;
        } else {
            //send the key
            KeyGenerator keygen(*context);
            *secretKeys = keygen.secret_key();
            keygen.create_relin_keys(*relinKeys);
            keygen.create_galois_keys(*galoisKeys);
            keygen.create_public_key(*publicKeys);
            encryptor = new Encryptor(*context, *publicKeys);
            decryptor = new Decryptor(*context, *secretKeys);
            uint64_t plain_mod = parms.plain_modulus().value(); 
            std::stringstream os;
            publicKeys->save(os);
            uint64_t pk_sze = static_cast<uint64_t>(os.tellp());
            galoisKeys->save(os);
            uint64_t gk_size = (uint64_t)os.tellp() - pk_sze;
            const std::string &keys_str = os.str();
            this->IO->send_data(&pk_sze, sizeof(uint64_t));
            this->IO->send_data(&gk_size,sizeof(uint64_t));
            this->IO->send_data(keys_str.c_str(),pk_sze + gk_size);
            std::cout << "Client send: " << keys_str.c_str() << "\n";
            std::cout << "Client send: " << pk_sze << "\n";
        }
    }

    void FreeKey(){
        auto safe_delete = [](auto *&ptr) {
            if (ptr) {
                delete ptr;
                ptr = nullptr;
            }
        };

        safe_delete(batchEncoder);
        safe_delete(context);
        safe_delete(encryptor);
        safe_delete(decryptor);
        safe_delete(encoder);
        safe_delete(evaluator);
        safe_delete(relinKeys);
        safe_delete(galoisKeys);
        safe_delete(publicKeys);
        safe_delete(secretKeys);
    }

    void SendCipherText(const unified::UnifiedCiphertext &ct){
        std::stringstream os;
        if (ct.is_device()) {
            /**
               [Important]:
               1. There are two forms of ciphertext bit streams (SEAL and Phantom).
               2. Forms of ciphertext bit streams are dependent on evaluator->backend()
             */
            throw std::invalid_argument("SendCipherText: Need to explicitly transfer `ct` to HOST");
        }
        ct.save(os);
        uint64_t ct_sze = static_cast<uint64_t>(os.tellp());
        const std::string &ct_str = os.str();
        this->IO->send_data(&ct_sze, sizeof(uint64_t));
        this->IO->send_data(ct_str.c_str(), ct_sze);
    }

    void SendEncVec(const Tensor<unified::UnifiedCiphertext> &ct_vec){
        uint64_t vec_size = static_cast<uint64_t>(ct_vec.size());
        this->IO->send_data(&vec_size, sizeof(uint64_t));

        // Send each Ciphertext in the vector using SendCipherText
        for (size_t i = 0; i < vec_size; i++) {
            SendCipherText(ct_vec({i}));
        }
    }

    void ReceiveCipherText(unified::UnifiedCiphertext &ct){
        uint64_t ct_sze{0};
        this->IO->recv_data(&ct_sze,sizeof(uint64_t));
        char *char_buf = new char[ct_sze];
        this->IO->recv_data(char_buf,ct_sze);
        std::stringstream is;
        is.write(char_buf,ct_sze);
        ct.load(*context,is);
        delete[] char_buf;
    }

    void ReceiveEncVec(Tensor<unified::UnifiedCiphertext> &ct_vec){
        uint64_t vec_size{0};
        this->IO->recv_data(&vec_size,sizeof(uint64_t));
        assert(vec_size == ct_vec.size() && "Number of ciphertexts does not match.");

        for (size_t i = 0; i < vec_size; ++i){
            ReceiveCipherText(ct_vec({i}));
        }
    }

    unified::UnifiedCiphertext GenerateZeroCiphertext() {
        unified::UnifiedPlaintext zeros_pt(HOST);
        unified::UnifiedCiphertext zeros_ct(HOST);

        std::vector<uint64_t> zeros(this->polyModulusDegree, 0);
        this->batchEncoder->encode(zeros, zeros_pt);
        this->encryptor->encrypt(zeros_pt, zeros_ct);

        if (IsGPUenable()) {
            zeros_ct.to_device(*context);
        }

        return zeros_ct;
    }

    bool IsGPUenable() {
        return evaluator->backend() == LOCATION::DEVICE;
    }
};

}