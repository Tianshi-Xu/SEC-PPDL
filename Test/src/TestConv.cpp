#include <LinearLayer/Conv.h>

int main(int argc, char **argv){
    bool party = std::stoi(argv[1]);
    const char* address = "127.0.0.1";
    int port = 32000;
    std::cout << party << std::endl;
    NetIO netio(address, port, party);
    std::cout << "netio generated" << std::endl;
    HEEvaluator HE(netio, party);
    HE.GenerateNewKey();
    std::cout << "in HE" << HE.polyModulusDegree << std::endl;
    
    uint64_t Ci = 50; uint64_t Co = 160; uint64_t H = 16; uint64_t W = 16;
    uint64_t p = 1; uint64_t s = 2; uint64_t k = 7; uint64_t Ho = 9; uint64_t Wo = 9;
    Tensor<int64_t> input({Ci, H, W}); 
    Tensor<int64_t> weight({Co, Ci, k, k});
    Tensor<uint64_t> bias({Co});
    for (uint64_t i = 0; i < Co * Ci * k * k; i++) {
        weight(i) = 1;
    }
    for (uint64_t i = 0; i < Ci * H * W ; i++){
        input(i) = 1;
    }

    //Conv2DNest conv(H, s, p, weight, bias, &HE);
    LinearLayer::Conv2DCheetah conv(H,W, &HE, weight);
    std::cout << "CW HW WW: " << conv.CW << conv.HW << conv.WW << conv.HW << conv.dC << std::endl;
    std::cout << conv.polyModulusDegree << std::endl;

    auto pack = conv.PackTensor(input);
    auto pack2 = conv.PackKernel(weight);
    auto enc = conv.EncryptTensor(pack);
    //Tensor<uint64_t> output = conv(input);
    for (uint64_t i = 0; i < Co; i++){
        std::cout << i << std::endl;
        for (uint64_t j = 0; j < Ho; j++){
            for (uint64_t l = 0; l < Wo; l++){
                //std::cout << output({i, j, l}) << " ";
            }
            //std::cout << std::endl;
        }
    }

    return 0;
}