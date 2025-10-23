#include <LinearLayer/Conv.h>
// #include <Model/ResNet.h>
#include <fstream>
#include <iostream>

int party, port = 32000;
int num_threads = 2;
string address = "127.0.0.11";

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
    uint64_t Ci = 2; uint64_t Co = 2; uint64_t H = 2; uint64_t W = 2;
    uint64_t padding = 0; uint64_t s = 1; uint64_t kernelSize = 1;
    Tensor<uint64_t> input1({Ci, H, W}); 
    Tensor<uint64_t> input2({Ci, H, W}); 
    Tensor<uint64_t> weight({Co, Ci, kernelSize, kernelSize});
    Tensor<uint64_t> bias({Co});
    for(uint32_t i = 0; i < Co; i++){
        for(uint32_t j = 0; j < Ci; j++){
            for(uint32_t p = 0; p < kernelSize; p++){
                for(uint32_t q = 0; q < kernelSize; q++){
                    weight({i, j, p, q}) = i + j + p + q;
                }
            }
        }
    }

    for(uint32_t i = 0; i < Ci; i++){
        for(uint32_t j = 0; j < H; j++){
            for(uint32_t p = 0; p < W; p++){
                input1({i, j, p}) = i + j + p;
                input2({i, j, p}) = i + j + p;
            }
        }
    }


    cout << "input generated" << endl;
    // Conv2D* conv1 = new Conv2DNest(H, Ci, Co, kernelSize, s, &HE);
    Conv2D* conv1 = new Conv2DNest(H,s,padding,weight,bias,&HE);
    // Conv2D* conv1 = new Conv2DCheetah(H, Ci, Co, k, s, &HE);
    // Conv2D* conv1 = new Conv2DCheetah(H,s,padding,weight,bias,&HE);
    // conv1->weight.print_shape();
    if (party == 1){
        Tensor<uint64_t> output = conv1->operator()(input1);
        output.print();
    }else{
        Tensor<uint64_t> output = conv1->operator()(input2);
        output.print();
    }
    input1.print();
    weight.print();
    cout << HE.plain_mod << endl;
    cout << pow(2,40) << endl;



    size_t H_out = (H - kernelSize + 2 * padding) / s + 1;
    size_t W_out = (W - kernelSize + 2 * padding) / s + 1;

    Tensor<int64_t> O({Co,H_out,W_out});

    

    for (size_t m = 0; m < Co; m++){
        for (size_t i = 0; i < H_out; i++){
            for (size_t j = 0; j < W_out; j++){
                int64_t sum = 0;
                int in_i = i * s;
                int in_j = j * s;

                for (size_t c = 0; c < Ci; c++){
                    for (size_t p = 0; p < kernelSize; p++){
                        for (size_t q = 0; q < kernelSize; q++){
                            sum += (input1({c,in_i + p, in_j + q}) + input2({c,in_i + p, in_j + q})) * weight({m, c, p, q});
                        }
                    }
                }
                O({m, i, j}) = sum;
            }
        }
    }

    O.print();



    return 0;
}


// #include <LinearLayer/Conv.h>
// #include <Model/ResNet.h>
// #include <fstream>
// #include <iostream>

// int party, port = 32000;
// int num_threads = 2;
// string address = "127.0.0.1";

// using namespace std;
// using namespace LinearLayer;
// int main(int argc, char **argv){
//     ArgMapping amap;
//     amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2"); // 1 is server, 2 is client
//     amap.arg("p", port, "Port Number");
//     amap.arg("ip", address, "IP Address of server (ALICE)");
//     amap.parse(argc, argv);
    
//     Utils::NetIO* netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
//     std::cout << "netio generated" << std::endl;
//     HE::HEEvaluator HE(netio, party, 8192,60,Datatype::HOST);
//     HE.GenerateNewKey();
    
//     // return 0;
//     uint64_t Ci = 4; uint64_t Co = 4; uint64_t H =1; uint64_t W = 1;
//     uint64_t p = 0; uint64_t s = 1; uint64_t k = 1;
//     Tensor<uint64_t> input({Ci, H, W}); 
//     Tensor<uint64_t> weight({Co, Ci, k, k});
//     Tensor<uint64_t> bias({Co});
//     for(uint32_t i = 0; i < Co; i++){
//         for(uint32_t j = 0; j < Ci; j++){
//             for(uint32_t p = 0; p < k; p++){
//                 for(uint32_t q = 0; q < k; q++){
//                     weight({i, j, p, q}) = 1;
//                 }
//             }
//         }
//     }
//     for(uint32_t i = 0; i < Ci; i++){
//         for(uint32_t j = 0; j < H; j++){
//             for(uint32_t p = 0; p < W; p++){
//                 input({i, j, p}) = 1;
//             }
//         }
//     }
//     cout << "input generated" << endl;
//     // Conv2D* conv1 = new Conv2DNest(H, Ci, Co, k, s, &HE);
//     Conv2D* conv1 = new Conv2DNest(H,s,p,weight,bias,&HE);
//     // Conv2D* conv1 = new Conv2DCheetah(H, Ci, Co, k, s, &HE);
//     // Conv2D* conv1 = new Conv2DCheetah(H,s,p,weight,bias,&HE);
//     // conv1->weight.print_shape();
//     Tensor<uint64_t> output = conv1->operator()(input);
//     output.print_shape();
//     output.print();
//     cout << HE.plain_mod << endl;
//     cout << pow(2,40) << endl;

//     return 0;
// }
