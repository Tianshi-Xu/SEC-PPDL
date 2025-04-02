#include <LinearLayer/Linear.h>
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
    HE::HEEvaluator HE(netio, party, 8192, 60, Datatype::HOST);
    HE.GenerateNewKey();
    
    uint64_t d0 = 8; uint64_t d1 = 8; uint64_t d2 = 8;
    Tensor<uint64_t> input({d0, d1}); 
    Tensor<uint64_t> weight({d1, d2});
    Tensor<uint64_t> bias({d0, d2});
    if(party == ALICE){
        for(uint32_t i = 0; i < d0; i++){
            for(uint32_t j = 0; j < d1; j++){
                input({i, j}) = 1;
            }
        }
    }
    for(uint32_t i = 0; i < d1; i++){
        for(uint32_t j = 0; j < d2; j++){
            weight({i, j}) = 1;
        }
    }
    if(party == ALICE){
        input.print();
        weight.print();
    }

    cout << "input generated" << endl;
    LinearBolt* matmul1 = new LinearBolt(d0, weight, weight, &HE);
    Tensor<uint64_t> output1 = matmul1->operator()(input);
    output1.print(10);
    // MatmulCtctBumble* matmul2 = new MatmulCtctBumble(&HE);
    // Tensor<uint64_t> output2 = matmul2->operator()(input, weight);
    // output2.print();

    return 0;
}