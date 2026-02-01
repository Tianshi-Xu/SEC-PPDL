/**
 * TestCirLinear: Benchmark for Block Circulant Linear Layer
 * 
 * Tests CirLinearNest correctness and benchmarks HE operation counts/time.
 */

#include <LinearLayer/CirLinear.h>
#include <Model/ResNet.h>
#include <iostream>
#include <iomanip>

int party, port = 32000;
string address = "127.0.0.1";

using namespace std;
using namespace LinearLayer;

/**
 * Verify correctness against pre-computed expected output.
 */
bool verify_result(
    const Tensor<uint64_t>& output,
    const Tensor<uint64_t>& output_peer,
    const Tensor<uint64_t>& expected,
    uint64_t plain_mod)
{
    for (uint64_t i = 0; i < output.size(); i++) {
        uint64_t recon = (output(i) + output_peer(i)) % plain_mod;
        if (recon != expected(i)) return false;
    }
    return true;
}

/**
 * Compute expected output: Y[i,j] = sum_k weight[k,j] * input[i,k]
 */
Tensor<uint64_t> compute_expected(
    const Tensor<uint64_t>& input,
    const Tensor<uint64_t>& weight,
    uint64_t dim_0, uint64_t dim_1, uint64_t dim_2,
    uint64_t plain_mod)
{
    Tensor<uint64_t> expected({dim_0, dim_2});
    for (uint64_t i = 0; i < dim_0; i++) {
        for (uint64_t j = 0; j < dim_2; j++) {
            __uint128_t acc = 0;
            for (uint64_t k = 0; k < dim_1; k++) {
                acc += (__uint128_t)weight({k, j}) * input({i, k});
            }
            expected({i, j}) = acc % plain_mod;
        }
    }
    return expected;
}

/**
 * Create a block circulant weight matrix (all 1s in first column).
 */
Tensor<uint64_t> create_block_circulant_weight(
    uint64_t dim_1, uint64_t dim_2, uint64_t block_size)
{
    Tensor<uint64_t> weight({dim_1, dim_2});
    uint64_t num_blocks_1 = dim_1 / block_size;
    uint64_t num_blocks_2 = dim_2 / block_size;
    
    for (uint64_t bi = 0; bi < num_blocks_1; bi++) {
        for (uint64_t bj = 0; bj < num_blocks_2; bj++) {
            for (uint64_t i = 0; i < block_size; i++) {
                for (uint64_t j = 0; j < block_size; j++) {
                    weight({bi * block_size + i, bj * block_size + j}) = 1;
                }
            }
        }
    }
    return weight;
}

int main(int argc, char **argv) {
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.parse(argc, argv);
    
    Utils::NetIO* netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
    HE::HEEvaluator HE(netio, party, 8192, 60, Datatype::HOST);
    HE.GenerateNewKey();
    
    // Benchmark dimensions
    uint64_t dim_0 = 64, dim_1 = 256, dim_2 = 256;
    
    if (party == ALICE) {
        cout << "\n========== CirLinearNest Benchmark ==========\n";
        cout << "dim=(" << dim_0 << "," << dim_1 << "," << dim_2 << "), poly_degree=8192\n";
        cout << "-------------------------------------------------------------------\n";
        cout << setw(6) << "blk" << setw(8) << "ntt" << setw(6) << "tile"
             << setw(6) << "rots" << setw(6) << "muls"
             << setw(10) << "rot_ms" << setw(10) << "mul_ms"
             << setw(8) << "status\n";
        cout << "-------------------------------------------------------------------\n";
    }
    
    for (uint64_t block_size : {1, 2, 4, 8, 16, 32, 64}) {
        Tensor<uint64_t> input({dim_0, dim_1});
        Tensor<uint64_t> weight = create_block_circulant_weight(dim_1, dim_2, block_size);
        Tensor<uint64_t> bias({dim_0, dim_2});
        
        if (party == ALICE) {
            for (uint64_t i = 0; i < input.size(); i++) input(i) = i % 2;
        } else {
            for (uint64_t i = 0; i < input.size(); i++) input(i) = 0;
        }
        for (uint64_t i = 0; i < bias.size(); i++) bias(i) = 0;
        
        Tensor<uint64_t> expected = compute_expected(input, weight, dim_0, dim_1, dim_2, HE.plain_mod);
        
        CirLinearNest layer(dim_0, block_size, weight, bias, &HE);
        Tensor<uint64_t> output = layer(input);
        
        bool pass = true;
        if (party == ALICE) {
            Tensor<uint64_t> output_peer(output.shape());
            netio->recv_data(output_peer.data().data(), output_peer.size() * sizeof(uint64_t));
            pass = verify_result(output, output_peer, expected, HE.plain_mod);
            
            cout << setw(6) << block_size
                 << setw(8) << layer.ntt_size
                 << setw(6) << layer.tile_size
                 << setw(6) << layer.getRotationCount()
                 << setw(6) << layer.getMultiplyCount()
                 << setw(10) << fixed << setprecision(2) << layer.getRotationTimeMs()
                 << setw(10) << fixed << setprecision(2) << layer.getMultiplyTimeMs()
                 << setw(8) << (pass ? "PASS" : "FAIL") << "\n";
        } else {
            netio->send_data(output.data().data(), output.size() * sizeof(uint64_t));
        }
    }
    
    if (party == ALICE) {
        cout << "-------------------------------------------------------------------\n";
    }
    
    delete netio;
    return 0;
}
