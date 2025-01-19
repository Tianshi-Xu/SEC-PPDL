#include "layer/linear/MatCheetah.h"

inline std::string uint64_to_hex_string(std::uint64_t value)
{
    return seal::util::uint_to_hex_string(&value, std::size_t(1));
}

inline void print_line(int line_number)
{
    std::cout << "Line " << " " << line_number << " --> ";
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <server/client> <IP> <port>\n";
        return 1;
    }

    bool is_server = std::string(argv[1]) == "server";
    const char* ip = argv[2];
    int port = std::stoi(argv[3]);
    NetIO netio(ip, port, is_server);
    std::cout << "start test" << std::endl;
    HEEvaluator HE(netio,is_server);
    std::cout << "build";
    HE.GenerateNewKey(8192);
    std::cout << "gen";
    if (!is_server){
        print_line(__LINE__);
        uint64_t x = 6;
        Plaintext x_plain(uint64_to_hex_string(x));
        cout << "Express x = " + to_string(x) + " as a plaintext polynomial 0x" + x_plain.to_string() + "." << endl;
        print_line(__LINE__);
        Ciphertext x_encrypted;
        cout << "Encrypt x_plain to x_encrypted." << endl;
        
        HE.encryptor->encrypt(x_plain, x_encrypted);
        cout << "    + size of freshly encrypted x: " << x_encrypted.size() << endl;
        cout << "    + noise budget in freshly encrypted x: " << HE.decryptor->invariant_noise_budget(x_encrypted) << " bits"
            << endl;
        Plaintext x_decrypted;
        cout << "    + decryption of x_encrypted: ";
        HE.decryptor->decrypt(x_encrypted, x_decrypted);
        cout << "0x" << x_decrypted.to_string() << " ...... Correct." << endl;
        print_line(__LINE__);
        cout << "Compute x_sq_plus_one (x^2+1)." << endl;
        Ciphertext x_sq_plus_one;
        HE.evaluator->square(x_encrypted, x_sq_plus_one);
        Plaintext plain_one("1");
        HE.evaluator->add_plain_inplace(x_sq_plus_one, plain_one);

        Plaintext plain_poly("1");
        plain_poly.resize(HE.polyModulusDegree);
        std::vector<uint64_t> origin{1,2,3,4,5,6,7,8,9,10};
        const uint64_t plain = HE.plain;
        int len = 10;
        seal::util::modulo_poly_coeffs(origin, len, plain, plain_poly.data());
        std::fill_n(plain_poly.data() + len, HE.polyModulusDegree - len, 0);
        Ciphertext x_enc;
        HE.encryptor->encrypt(plain_poly,x_enc);
    }

    

    return 0;
}