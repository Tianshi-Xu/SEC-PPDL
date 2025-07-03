#include <Model/ResNet.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace HE;

int party, port = 32000;
int num_threads = 2;
string address = "127.0.0.1";

int main(int argc, char* argv[]) {
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2"); // 1 is server, 2 is client
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.parse(argc, argv);
    
    Utils::NetIO* netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
    std::cout << "netio generated" << std::endl;
    HE::HEEvaluator HE(netio, party, 8192,60,Datatype::DEVICE);
    HE.GenerateNewKey();
    UnifiedCiphertext ct1_r(HE.GenerateZeroCiphertext(HE.Backend()));
    UnifiedCiphertext temp_add_ct(HE.GenerateZeroCiphertext(HE.Backend()));
    UnifiedCiphertext ct1_l(HE.GenerateZeroCiphertext(HE.Backend()));
    Tensor<UnifiedCiphertext> temp_results1({2, 64}, HE.GenerateZeroCiphertext(HE.Backend()));
    Tensor<UnifiedCiphertext> rotatedIR({2, 64}, HE.GenerateZeroCiphertext(HE.Backend()));
    Tensor<UnifiedPlaintext> encodeMatrix_p1({128, 2}, DEVICE);
    for (size_t i = 0; i < 1000; i++){
        HE.evaluator->multiply_plain(rotatedIR({0, 0}),          encodeMatrix_p1({0, 0}), ct1_l);
        HE.evaluator->multiply_plain(rotatedIR({0, 32}), encodeMatrix_p1({0, 0}), ct1_r);
        HE.evaluator->add(ct1_l, ct1_r, temp_results1({0, 0}));
    }
    return 0;
}
