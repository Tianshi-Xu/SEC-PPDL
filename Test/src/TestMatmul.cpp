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
    HE::HEEvaluator HE(netio, party, 8192, 60, Datatype::HOST);
    HE.GenerateNewKey();
    
    // Use matrix dimensions where tile_size is a perfect square
    // With padded_dim_0=128, tile_size = 8192/128 = 64, input_rot = 8, and 8*8=64
    uint64_t d0 = 128; uint64_t d1 = 64; uint64_t d2 = 64;
    Tensor<uint64_t> input({d0, d1}); 
    Tensor<uint64_t> weight({d1, d2});
    Tensor<uint64_t> bias({d0, d2});
    if(party == ALICE){
        cout << "-----I'm ALICE------" << endl;
        for(uint32_t i = 0; i < d0; i++){
            for(uint32_t j = 0; j < d1; j++){
                input({i, j}) = 1;
            }
        }
    } else {
        // cout << "-----I'm BOB------" << endl;
        for (uint32_t i = 0; i < d0; i++) {
            for (uint32_t j = 0; j < d1; j++) {
                input({i, j}) = 0;
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

    // cout << "input generated" << endl;
    LinearBolt* matmul1 = new LinearBolt(d0, weight, weight, &HE);
    Tensor<uint64_t> output1 = matmul1->operator()(input);
    cout << "--------output LinearBolt--------" << endl;
    output1.print(10);
    LinearNest* matmul_nest = new LinearNest(d0, weight, weight, &HE);
    Tensor<uint64_t> output_nest = matmul_nest->operator()(input);
    cout << (998251184618198176 + 154670319988632417)%1152921504606830593 << endl;
    cout << "--------output LinearNest--------" << endl;
    output_nest.print(10);
    // MatmulCtctBumble* matmul2 = new MatmulCtctBumble(&HE);
    // Tensor<uint64_t> output2 = matmul2->operator()(input, weight);
    // output2.print();

    if (party == ALICE) {
        Tensor<uint64_t> output_bolt_peer(output1.shape());
        netio->recv_data(output_bolt_peer.data().data(), output_bolt_peer.size() * sizeof(uint64_t));
        
        Tensor<uint64_t> output_nest_peer(output_nest.shape());
        netio->recv_data(output_nest_peer.data().data(), output_nest_peer.size() * sizeof(uint64_t));

        Tensor<uint64_t> expected({d0, d2});
        for (uint64_t i = 0; i < d0; i++) {
            for (uint64_t j = 0; j < d2; j++) {
                uint64_t acc = 0;
                for (uint64_t k = 0; k < d1; k++) {
                    acc = (acc + (input({i, k}) * weight({k, j})) % HE.plain_mod) % HE.plain_mod;
                }
                expected({i, j}) = acc;
            }
        }

        auto verify = [&](const char* name, Tensor<uint64_t>& local_share, Tensor<uint64_t>& peer_share) {
            uint64_t mismatches = 0;
            for (uint64_t i = 0; i < local_share.size(); i++) {
                uint64_t recon = (local_share(i) + peer_share(i)) % HE.plain_mod;
                uint64_t exp = expected(i);
                if (recon != exp) {
                    if (mismatches < 8) {
                        std::cout << "[" << name << "] mismatch idx=" << i
                                  << " recon=" << recon
                                  << " expected=" << exp << std::endl;
                    }
                    mismatches++;
                }
            }
            if (mismatches == 0) {
                std::cout << "[" << name << "] PASS: all outputs match" << std::endl;
            } else {
                std::cout << "[" << name << "] FAIL: mismatches=" << mismatches << " total:" << local_share.size() << std::endl;
            }
        };
        verify("LinearBolt", output1, output_bolt_peer);
        verify("LinearNest", output_nest, output_nest_peer);
    } else {
        netio->send_data(output1.data().data(), output1.size() * sizeof(uint64_t));
        netio->send_data(output_nest.data().data(), output_nest.size() * sizeof(uint64_t));
    }

    return 0;
}