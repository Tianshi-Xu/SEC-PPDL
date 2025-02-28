#include <LinearLayer/Conv.h>

using namespace LinearLayer;
int main(int argc, char **argv){
    bool party = std::stoi(argv[1]);
    const char* address = "127.0.0.1";
    int port = 32000;
    Utils::NetIO netio(party==0?nullptr:address, port);
    std::cout << "netio generated" << std::endl;
    HE::HEEvaluator HE(netio, party);
    HE.GenerateNewKey();
    
    uint64_t Ci = 32; uint64_t Co = 64; uint64_t H = 16; uint64_t W = 16;
    uint64_t p = 1; uint64_t s = 1; uint64_t k = 1; uint64_t Ho = 16; uint64_t Wo = 16;
    Tensor<uint64_t> input({Ci, H, W}); 
    Tensor<uint64_t> weight({Co, Ci, k, k});
    Tensor<uint64_t> bias({Co});
    for(uint32_t i = 0; i < Co; i++){
        for(uint32_t j = 0; j < Ci; j++){
            for(uint32_t p = 0; p < k; p++){
                for(uint32_t q = 0; q < k; q++){
                    weight({i, j, p, q}) = 1;
                }
            }
        }
    }
    for(uint32_t i = 0; i < Ci; i++){
        for(uint32_t j = 0; j < H; j++){
            for(uint32_t p = 0; p < W; p++){
                input({i, j, p}) = 1;
            }
        }
    }
    cout << "input generated" << endl;
    Conv2D* conv1 = new Conv2DNest(H, Ci, Co, k, s, &HE);
    // Conv2DNest conv(H, s, p, weight, bias, &HE);
    conv1->weight.print_shape();
    Tensor<uint64_t> output = conv1->operator()(input);
    if (!party) {
        for (uint64_t i = 0; i < Co; i++){
            std::cout << i << std::endl;
            for (uint64_t j = 0; j < Ho; j++){
                for (uint64_t l = 0; l < Wo; l++){
                    std::cout << output({i, j, l}) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    return 0;
}