#include <seal/seal.h>
#include <seal/util/uintarith.h>
#include "NetIO.h"

#include <vector>

using namespace std;
using namespace seal;
using namespace seal::util;

class HEEvaluator {
    public:
    SEALContext *context = nullptr;
    Encryptor *encryptor = nullptr;
    Decryptor *decryptor = nullptr;
    BatchEncoder *encoder = nullptr;
    Evaluator *evaluator = nullptr;
    RelinKeys *relinKeys = nullptr;
    GaloisKeys *galoisKeys = nullptr;
    BatchEncoder *batchEncoder = nullptr;
    PublicKey *publicKeys = nullptr;
    SecretKey *secretKeys= nullptr;
    size_t slotCount = 0;
    size_t rowSize = 0;
    bool server = false;
    NetIO *IO = nullptr;
    size_t polyModulusDegree = 8192;
    uint64_t plain = 0;
    vector<int> COEFF_MODULI = {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58};


    HEEvaluator(
        NetIO &IO,
        bool server
    ){
        this->IO = &IO;
        this->server = server;
        // if (server){
        //     uint64_t pk_sze{0};
        //     this->IO->recv_data(&pk_sze, sizeof(uint64_t));
        //     char *key_buf = new char[pk_sze];
        //     this->IO->recv_data(key_buf, pk_sze);
        //     std::stringstream is;
        //     is.write(key_buf,pk_sze);
        //     publicKeys->load(*context,is);
        //     delete[] key_buf;
        //     std::cout << "Server received: " << key_buf << "\n";
        //     std::cout << "Server received: " << pk_sze << "\n";
        // }else{
        //     //send the key
        //     EncryptionParameters parms(scheme_type::bfv);
        //     parms.set_poly_modulus_degree(poly_modulus_degree);
        //     parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        //     parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        //     context = new SEALContext(parms);
        //     encoder = new BatchEncoder(*context);
        //     evaluator = new Evaluator(*context);
        //     KeyGenerator keygen(*context);
        //     *secretKeys = keygen.secret_key();
        //     encryptor = new Encryptor(*context, *publicKeys);
        //     decryptor = new Decryptor(*context, *secretKeys);
        //     keygen.create_relin_keys(*relinKeys);
        //     keygen.create_galois_keys(*galoisKeys);
        //     keygen.create_public_key(*publicKeys);
        //     batchEncoder = new  BatchEncoder(*context);
        //     slotCount = batchEncoder->slot_count();
        //     rowSize = slotCount / 2;
        //     uint64_t plain = parms.plain_modulus().value(); std::stringstream os;
        //     publicKeys->save(os);
        //     //Encryptor encryptor(*context, public_key);
        //     uint64_t pk_sze = static_cast<uint64_t>(os.tellp());
        //     const std::string &keys_str = os.str();
        //     this->IO->send_data(&pk_sze, sizeof(uint64_t));
        //     this->IO->send_data(keys_str.c_str(),pk_sze);
        //     std::cout << "Client send: " << keys_str.c_str() << "\n";
        //     std::cout << "Client send: " << pk_sze << "\n";
        // }
    }
    void GenerateNewKey(
        uint64_t slotCnt
    ){
        slotCount = slotCnt;
        polyModulusDegree = slotCount;
        publicKeys = new PublicKey();
        secretKeys = new SecretKey();
        relinKeys  = new RelinKeys();
        galoisKeys = new GaloisKeys();
        EncryptionParameters parms(scheme_type::bfv);
        parms.set_poly_modulus_degree(polyModulusDegree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(polyModulusDegree));
        parms.set_plain_modulus(PlainModulus::Batching(polyModulusDegree, 20));
        context = new SEALContext(parms);
        encoder = new BatchEncoder(*context);
        evaluator = new Evaluator(*context);
        batchEncoder = new  BatchEncoder(*context);
        plain = parms.plain_modulus().value();
        if (server){
            uint64_t pk_sze{0};
            uint64_t gk_sze{0};
            this->IO->recv_data(&pk_sze, sizeof(uint64_t));
            this->IO->recv_data(&gk_sze, sizeof(uint64_t));
            char *key_buf = new char[pk_sze + gk_sze];
            this->IO->recv_data(key_buf, pk_sze + gk_sze);
            std::stringstream is;
            is.write(key_buf,pk_sze);
            publicKeys->load(*context,is);
            is.write(key_buf + pk_sze, gk_sze);
            galoisKeys->load(*context,is);
            std::cout << "Server received: " << key_buf << "\n";
            std::cout << "Server received: " << pk_sze << "\n";
            encryptor = new Encryptor(*context, *publicKeys);
            delete[] key_buf;
        }else{
            //send the key
            KeyGenerator keygen(*context);
            *secretKeys = keygen.secret_key();
            keygen.create_relin_keys(*relinKeys);
            keygen.create_galois_keys(*galoisKeys);
            keygen.create_public_key(*publicKeys);
            encryptor = new Encryptor(*context, *publicKeys);
            decryptor = new Decryptor(*context, *secretKeys);
            slotCount = batchEncoder->slot_count();
            rowSize = slotCount / 2;
            uint64_t plain = parms.plain_modulus().value(); 
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

    void SendCipherText(const Ciphertext &ct){
        std::stringstream os;
        ct.save(os);
        uint64_t ct_sze = static_cast<uint64_t>(os.tellp());
        const std::string &ct_str = os.str();
        this->IO->send_data(&ct_sze,sizeof(uint64_t));
        this->IO->send_data(ct_str.c_str(),ct_sze);
    }

    void SendEncVec(const std::vector<seal::Ciphertext> &ct_vec){
        uint64_t vec_size = static_cast<uint64_t>(ct_vec.size());
        this->IO->send_data(&vec_size, sizeof(uint64_t));

        // Send each Ciphertext in the vector using SendCipherText
        for (const auto &ct : ct_vec) {
            SendCipherText(ct);
        }
    }

    void ReceiveCipherText(seal::Ciphertext &ct){
        uint64_t ct_sze{0};
        this->IO->recv_data(&ct_sze,sizeof(uint64_t));
        char *char_buf = new char[ct_sze];
        this->IO->recv_data(char_buf,ct_sze);
        std::stringstream is;
        is.write(char_buf,ct_sze);
        ct.load(*context,is);
        delete[] char_buf;
    }

    void ReceiveEncVec(std::vector<seal::Ciphertext> &ct_vec){
        ct_vec.clear();
        uint64_t vec_size{0};
        this->IO->recv_data(&vec_size,sizeof(uint64_t));
        ct_vec.resize(vec_size);
        for(uint64_t i = 0; i < vec_size; ++i){
            ReceiveCipherText(ct_vec[i]);
        }
    }
};
