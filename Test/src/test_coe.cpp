// #include <LinearLayer/Conv.h>
#include <Model/ResNet.h>
#include <fstream>

int party, port = 32000;
int num_threads = 8;
string address = "127.0.0.1";

int rand_int(int min_val = -10, int max_val = 10){
    return min_val + rand() % (max_val - min_val + 1);
}

int div_upper(
    int a,
    int b
){
    return ((a + b - 1) / b);
}

int main(int argc, char **argv){
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2"); // 1 is server, 2 is client
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.parse(argc, argv);
    size_t padding = 3;

    size_t len = 56 * 56 * 64;

    std::cout << party << std::endl;
    Utils::NetIO* netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
    std::cout << "netio generated" << std::endl;
    HEEvaluator HE(netio, party, 8192, 20);
    HE.GenerateNewKey();

    uint64_t C = 1024; uint64_t M = 256; uint64_t H = 14; uint64_t W = 14;
    uint64_t s = 2; uint64_t h = 3; 
    Tensor<uint64_t> T1({C, H, W}); 
    Tensor<uint64_t> T2({C, H, W}); 
    Tensor<uint64_t>  T({C, H, W}); 
    Tensor<uint64_t> T_padding({C, H + 2 * padding, W + 2 * padding}, 0);
    Tensor<uint64_t> K({M, C, h, h});
    Tensor<uint64_t> KF({M, C, h, h});
    for (uint64_t i = 0; i < C; i++) {
        for (size_t j = 0; j < H; j++) {
            for (size_t k = 0; k < W; k++){
                T1({i, j, k}) = rand_int(-3,3);
                T2({i, j, k}) = rand_int(-3,3);
                T({i, j, k}) = T1({i, j, k}) + T2({i, j, k});
                T_padding({i, j + padding, k + padding}) = T({i, j, k});
            }
        }
    }

    Tensor<uint64_t> beta({M}, 0);
    Tensor<uint64_t> gamma({M}, 0);

    for (size_t i = 0; i < M; i++){
        beta({i}) = rand_int(-3,3);
        //gamma({i}) = rand_int(-3,3);
        gamma({i}) = rand_int(1, 10);
    }

    for (uint64_t i = 0; i < M; i++){
        for (size_t j = 0; j < C; j++){
            for (size_t k = 0; k < h; k++){
                for (size_t l = 0; l < h; l++){
                    K({i,j,k,l})  = rand_int(-3,3);
                }
            }
        }
    }

    for (uint64_t i = 0; i < M; i++){
        for (size_t j = 0; j < C; j++){
            for (size_t k = 0; k < h; k++){
                for (size_t l = 0; l < h; l++){
                    KF({i,j,k,l})  = K({i,j,k,l}) * gamma({i});
                }
            }
        }
    }

    size_t H_out = (H - h + 2 * padding) / s + 1;
    size_t W_out = (W - h + 2 * padding) / s + 1;

    Tensor<int64_t> O({M,H_out,W_out});

    for (size_t m = 0; m < M; m++){
        for (size_t i = 0; i < H_out; i++){
            for (size_t j = 0; j < W_out; j++){
                int sum = 0;
                int in_i = i * s;
                int in_j = j * s;

                for (size_t c = 0; c < C; c++){
                    for (size_t p = 0; p < h; p++){
                        for (size_t q = 0; q < h; q++){
                            sum += T_padding({c,in_i + p, in_j + q}) * KF({m, c, p, q});
                        }
                    }
                }
                O({m, i, j}) = sum;
            }
        }
    }

    size_t HWprime = (H + 2 * padding - h + s) / s;
    size_t WWprime = (W + 2 * padding - h + s) / s;
    std::cout << "HWprime:" << HWprime << std::endl;
    std::cout << "WWprime:" << WWprime << std::endl; 


    Tensor<uint64_t> KC({M, HWprime, WWprime},0);




    //Conv2DNest conv(H, s, p, weight, bias, &HE);
    LinearLayer::Conv2DCheetah conv(H,s,padding,K,KC,&HE, &gamma, &beta);
    if (party){
        auto rrrr = conv.operator()(T1);

        std::string filename = "tensorData1.bin";
        std::ofstream outfile(filename, std::ios::binary);


        for (size_t i = 0; i < M; i++){
            for (size_t j = 0; j < H_out; j++){
                for (size_t k = 0; k < W_out; k++){
                    int64_t value;
                    value = rrrr({i,j,k});
                    outfile << value << " ";
                }
            }
        }
        
        outfile.close();


    }else{
        auto rrrr = conv.operator()(T2);

        std::string filename = "tensorData0.bin";
        std::ofstream outfile(filename, std::ios::binary);


        for (size_t i = 0; i < M; i++){
            for (size_t j = 0; j < H_out; j++){
                for (size_t k = 0; k < W_out; k++){
                    int64_t value;
                    value = rrrr({i,j,k});
                    outfile << value << " ";
                }
            }
        }
        
        outfile.close();
    }

    
    
    std::cout << "CW HW WW: " << conv.CW << conv.HW << conv.WW << conv.HW << conv.dC << std::endl;
    std::cout << conv.polyModulusDegree << std::endl;


    return 0;
}