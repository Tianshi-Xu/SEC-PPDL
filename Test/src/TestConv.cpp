#include <LinearLayer/Conv.h>

int main(int argc, char **argv){
    bool party = std::stoi(argv[1]);
    const char* address = "127.0.0.1";
    int port = 32001;
    std::cout << party << std::endl;
    HE::NetIO netio(address, port, party);
    std::cout << "netio generated" << std::endl;
    HE::HEEvaluator HE(netio, party);
    HE.GenerateNewKey();
    Ciphertext ct = Ciphertext();
    
    uint64_t Ci = 16; uint64_t Co = 16; uint64_t H = 19; uint64_t W = 19;
    uint64_t p = 1; uint64_t s = 2; uint64_t k = 5; uint64_t Ho = 9; uint64_t Wo = 9;
    Tensor<uint64_t> input({Ci, H, W}); 
    Tensor<uint64_t> weight({Co, Ci, k, k});
    Tensor<uint64_t> bias({Co});
    for(uint32_t i = 0; i < Co; i++){
        for(uint32_t j = 0; j < Ci; j++){
            for(uint32_t k = 0; k < k; k++){
                for(uint32_t l = 0; l < k; l++){
                    weight({i, j, k, l}) = party;
                }
            }
        }
    }
    for(uint32_t i = 0; i < Ci; i++){
        for(uint32_t j = 0; j < H; j++){
            for(uint32_t k = 0; k < W; k++){
                input({i, j, k}) = party;
            }
        }
    }
    cout << "input generated" << endl;
    LinearLayer::Conv2DNest conv(H, s, p, weight, bias, &HE);
    Tensor<uint64_t> output = conv(input);
    for (uint64_t i = 0; i < Co; i++){
        std::cout << i << std::endl;
        for (uint64_t j = 0; j < Ho; j++){
            for (uint64_t l = 0; l < Wo; l++){
                std::cout << output({i, j, l}) << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}