//#include "NetIO.h"
#include "HE.h"
#include <cstring>
#include "seal/seal.h"
#include <seal/secretkey.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/rlwe.h>
#include <seal/secretkey.h>
#include <seal/serializable.h>
using namespace seal;
using namespace seal::util;
using namespace std;



int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <server/client> <IP> <port>\n";
        return 1;
    }

    bool is_server = std::string(argv[1]) == "server";
    const char* ip = argv[2];
    int port = std::stoi(argv[3]);
    NetIO netio(ip, port, is_server);
    // const char* msg = "Hello from client!";
    // size_t poly_modulus_degree = 8192;
    // EncryptionParameters parms(scheme_type::bfv);
    // vector<int> COEFF_MODULI = {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58};
    // vector<int> MM_COEFF_MODULI = {60, 40, 60};
    // parms.set_poly_modulus_degree(poly_modulus_degree);
    // parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    // seal::SEALContext* context;
    // std::vector<int> moduli_bits{60,49};
    // parms.set_poly_modulus_degree(poly_modulus_degree);
    // parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    // parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
    // context = new SEALContext(parms, true, sec_level_type::tc128);
    // KeyGenerator keygen(*context);
    // SecretKey secret_key = keygen.secret_key();
    // RelinKeys relin_keys;
    // keygen.create_relin_keys(relin_keys);
    // GaloisKeys galois_keys;
    // keygen.create_galois_keys(galois_keys);
    // BatchEncoder encoder(*context);
    // //Evaluator evaluator(*context);
    // Decryptor decryptor(*context, secret_key);
    // if (is_server) {
    //     // 接收数据
    //     auto pk_ = std::make_shared<PublicKey>();
    //     uint64_t pk_sze{0};
    //     netio.recv_data(&pk_sze, sizeof(uint64_t));
    //     char *key_buf = new char[pk_sze];
    //     netio.recv_data(key_buf, pk_sze);
    //     std::stringstream is;
    //     is.write(key_buf,pk_sze);
    //     pk_->load(*context,is);
    //     delete[] key_buf;
    //     std::cout << "Server received: " << key_buf << "\n";
    //     std::cout << "Server received: " << pk_sze << "\n";
    // } else {
    //     // 发送数据
    //     std::stringstream os;
    //     PublicKey public_key;
    //     keygen.create_public_key(public_key);
    //     public_key.save(os);
    //     //Encryptor encryptor(*context, public_key);
    //     uint64_t pk_sze = static_cast<uint64_t>(os.tellp());
    //     const std::string &keys_str = os.str();
    //     netio.send_data(&pk_sze, sizeof(uint64_t));
    //     netio.send_data(keys_str.c_str(),pk_sze);
    //     std::cout << "Client send: " << keys_str.c_str() << "\n";
    //     std::cout << "Server send: " << pk_sze << "\n";
    // }

    //start test!
    std::cout << "start test" << std::endl;
    HEEvaluator HE(netio,is_server);
    std::cout << "build";
    HE.GenerateNewKey(8192);
    std::cout << "gen";
    return 0;
}