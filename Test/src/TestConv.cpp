#include <LinearLayer/Conv.h>
// #include <Model/ResNet.h>
#include <Utils/ArgMapping/ArgMapping.h>
#include <iostream>
#include <vector>

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
    HE::HEEvaluator HE(netio, party, 8192, 60, Datatype::HOST);
    HE.GenerateNewKey();
    struct Case {
        uint64_t Ci;
        uint64_t Co;
        uint64_t H;
        uint64_t kernel;
        uint64_t stride;
        uint64_t padding;
    };

    std::vector<Case> cases = {
        {3, 4, 4, 3, 1, 1},
        {16, 16, 32, 3, 1, 1},
        {16, 32, 32, 3, 2, 1},
        {32, 64, 16, 3, 1, 1},
        {64, 64, 16, 1, 1, 0},
        {64, 128, 16, 3, 2, 1},
        {128, 128, 8, 3, 1, 1}
    };

    std::vector<double> case_ratios(cases.size(), -1.0);

    for (size_t case_idx = 0; case_idx < cases.size(); case_idx++) {
        const auto &tc = cases[case_idx];
        uint64_t Ci = tc.Ci;
        uint64_t Co = tc.Co;
        uint64_t H = tc.H;
        uint64_t W = tc.H;
        uint64_t kernelSize = tc.kernel;
        uint64_t s = tc.stride;
        uint64_t padding = tc.padding;

        std::cout << "[Case " << case_idx << "] start Ci=" << Ci
              << " Co=" << Co
              << " H=" << H
              << " K=" << kernelSize
              << " S=" << s
              << " P=" << padding << std::endl;

        Tensor<uint64_t> input({Ci, H, W});
        Tensor<uint64_t> weight({Co, Ci, kernelSize, kernelSize});
        Tensor<uint64_t> bias({Co});

        if (party == ALICE) {
            for (uint32_t i = 0; i < Co; i++) {
                for (uint32_t j = 0; j < Ci; j++) {
                    for (uint32_t p = 0; p < kernelSize; p++) {
                        for (uint32_t q = 0; q < kernelSize; q++) {
                            weight({i, j, p, q}) = (i + j + p + q) % 7;
                        }
                    }
                }
            }
            for (uint32_t i = 0; i < Ci; i++) {
                for (uint32_t j = 0; j < H; j++) {
                    for (uint32_t p = 0; p < W; p++) {
                        input({i, j, p}) = (i + j + p + 1) % 5;
                    }
                }
            }
        } else {
            for (uint32_t i = 0; i < Co; i++) {
                for (uint32_t j = 0; j < Ci; j++) {
                    for (uint32_t p = 0; p < kernelSize; p++) {
                        for (uint32_t q = 0; q < kernelSize; q++) {
                            weight({i, j, p, q}) = 0;
                        }
                    }
                }
            }
            for (uint32_t i = 0; i < Ci; i++) {
                for (uint32_t j = 0; j < H; j++) {
                    for (uint32_t p = 0; p < W; p++) {
                        input({i, j, p}) = 0;
                    }
                }
            }
        }

        Conv2D* conv1 = new Conv2DNest(H, s, padding, weight, bias, &HE);
        size_t H_out = (H - kernelSize + 2 * padding) / s + 1;
        size_t W_out = (W - kernelSize + 2 * padding) / s + 1;
        Tensor<uint64_t> output({Co, H_out, W_out});
        output = conv1->operator()(input);
        if (party == ALICE) {
            Tensor<uint64_t> output_peer(output.shape());
            netio->recv_data(output_peer.data().data(), output_peer.size() * sizeof(uint64_t));

            Tensor<uint64_t> padded_input({Ci, H + 2 * padding, W + 2 * padding});
            for (uint64_t c = 0; c < Ci; c++) {
                for (uint64_t i = 0; i < H; i++) {
                    for (uint64_t j = 0; j < W; j++) {
                        padded_input({c, i + padding, j + padding}) = input({c, i, j});
                    }
                }
            }

            Tensor<int64_t> expected({Co, H_out, W_out});
            for (size_t m = 0; m < Co; m++) {
                for (size_t i = 0; i < H_out; i++) {
                    for (size_t j = 0; j < W_out; j++) {
                        int64_t sum = 0;
                        uint64_t in_i = i * s;
                        uint64_t in_j = j * s;

                        for (size_t c = 0; c < Ci; c++) {
                            for (size_t p = 0; p < kernelSize; p++) {
                                for (size_t q = 0; q < kernelSize; q++) {
                                    sum += padded_input({c, in_i + p, in_j + q}) * weight({m, c, p, q});
                                }
                            }
                        }
                        expected({m, i, j}) = sum;
                    }
                }
            }
            cout << "expected:" << endl;
            expected.print();
            Tensor<uint64_t> reconstructed(output.shape());
            uint64_t mismatches = 0;
            for (uint64_t i = 0; i < output.size(); i++) {
                uint64_t recon = (output(i) + output_peer(i)) % HE.plain_mod;
                reconstructed(i) = recon;
                uint64_t exp = static_cast<uint64_t>(expected(i) % static_cast<int64_t>(HE.plain_mod));
                mismatches += (recon != exp);
            }
            cout << "reconstructed:" << endl;
            reconstructed.print();

            double ratio = output.size() ? static_cast<double>(mismatches) / static_cast<double>(output.size()) : 0.0;
            std::cout << "[Case " << case_idx << "] Ci=" << Ci
                      << " Co=" << Co
                      << " H=" << H
                      << " K=" << kernelSize
                      << " S=" << s
                      << " P=" << padding
                      << " mismatch_ratio=" << ratio << std::endl;
            std::cout << "[Case " << case_idx << "] done" << std::endl;

            case_ratios[case_idx] = ratio;
        } else {
            netio->send_data(output.data().data(), output.size() * sizeof(uint64_t));
            std::cout << "[Case " << case_idx << "] sent" << std::endl;
        }

        delete conv1;
    }

    if (party == ALICE) {
        std::cout << "[Summary] per-case results:" << std::endl;
        for (size_t i = 0; i < cases.size(); i++) {
            const auto &tc = cases[i];
            std::cout << "  Case " << i
                      << " (Ci=" << tc.Ci
                      << ", Co=" << tc.Co
                      << ", H=" << tc.H
                      << ", K=" << tc.kernel
                      << ", S=" << tc.stride
                      << ", P=" << tc.padding
                      << ") mismatch_ratio=" << case_ratios[i] << std::endl;
        }
    }
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
