// #include <LinearLayer/Conv.h>
#include <Model/ResNet.h>
#include <fstream>
#include <iostream>

int party, port = 32000;
int num_threads = 2;
string address = "127.0.0.1";

using namespace std;
using namespace LinearLayer;
int main(int argc, char **argv){
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2"); // 1 is server, 2 is client
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.parse(argc, argv);
    
    Utils::NetIO* netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
    std::cout << "netio generated" << std::endl;
    HE::HEEvaluator HE(netio, party, 8192,60,Datatype::HOST);
    HE.GenerateNewKey();
    
    // return 0;
    uint64_t Ci = 16; uint64_t Co = 3; uint64_t H =32; uint64_t W = 32;
    uint64_t p = 1; uint64_t s = 1; uint64_t k = 3;
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
    // Conv2D* conv1 = new Conv2DNest(H, Ci, Co, k, s, &HE);
    Conv2D* conv1 = new Conv2DNest(H,s,p,weight,bias,&HE);
    // Conv2D* conv1 = new Conv2DCheetah(H, Ci, Co, k, s, &HE);
    // Conv2D* conv1 = new Conv2DCheetah(H,s,p,weight,bias,&HE);
    // conv1->weight.print_shape();
    Tensor<uint64_t> output = conv1->operator()(input);
    output.print_shape();
    output.print();
    cout << HE.plain_mod << endl;
    cout << pow(2,40) << endl;

    return 0;
}