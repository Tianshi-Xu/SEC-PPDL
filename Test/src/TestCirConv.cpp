/**
 * TestCirConv: Correctness test for Block Circulant Convolution Layer
 *
 * Tests CirConv2D with varying block sizes against brute-force convolution.
 */

#include <LinearLayer/Conv.h>
#include <Model/ResNet.h>
#include <iostream>
#include <iomanip>

int party, port = 32000;
string address = "127.0.0.1";

using namespace std;
using namespace LinearLayer;

/**
 * Create a block circulant convolution weight.
 * The weight is block circulant in (Co, Ci) with block size b:
 *   W[bo*b+i, bi*b+j, m, n] = W[bo*b+((i-j+b)%b), bi*b, m, n]
 */
Tensor<uint64_t> create_block_circulant_conv_weight(
    uint64_t Co, uint64_t Ci, uint64_t kernel_size, uint64_t block_size)
{
    Tensor<uint64_t> weight({Co, Ci, kernel_size, kernel_size});
    uint64_t num_blocks_out = Co / block_size;
    uint64_t num_blocks_in = Ci / block_size;

    for (uint64_t bo = 0; bo < num_blocks_out; bo++) {
        for (uint64_t bi = 0; bi < num_blocks_in; bi++) {
            for (uint64_t i = 0; i < block_size; i++) {
                for (uint64_t m = 0; m < kernel_size; m++) {
                    for (uint64_t n = 0; n < kernel_size; n++) {
                        weight({bo * block_size + i, bi * block_size, m, n}) =
                            (i + m + n + bo + bi + 1) % 5;
                    }
                }
            }
            for (uint64_t i = 0; i < block_size; i++) {
                for (uint64_t j = 1; j < block_size; j++) {
                    for (uint64_t m = 0; m < kernel_size; m++) {
                        for (uint64_t n = 0; n < kernel_size; n++) {
                            weight({bo * block_size + i, bi * block_size + j, m, n}) =
                                weight({bo * block_size + (i - j + block_size) % block_size,
                                        bi * block_size, m, n});
                        }
                    }
                }
            }
        }
    }

    return weight;
}

/**
 * Compute expected convolution output via brute force.
 */
Tensor<uint64_t> compute_expected_conv(
    const Tensor<uint64_t>& input,
    const Tensor<uint64_t>& weight,
    uint64_t Ci, uint64_t Co,
    uint64_t H, uint64_t kernel_size,
    uint64_t stride, uint64_t pad,
    uint64_t plain_mod)
{
    uint64_t H_out = (H + 2 * pad - kernel_size) / stride + 1;
    uint64_t W_out = H_out;
    Tensor<uint64_t> expected({Co, H_out, W_out});

    Tensor<uint64_t> padded_input({Ci, H + 2 * pad, H + 2 * pad});
    for (uint64_t c = 0; c < Ci; c++)
        for (uint64_t i = 0; i < H; i++)
            for (uint64_t j = 0; j < H; j++)
                padded_input({c, i + pad, j + pad}) = input({c, i, j});

    for (uint64_t co = 0; co < Co; co++) {
        for (uint64_t i = 0; i < H_out; i++) {
            for (uint64_t j = 0; j < W_out; j++) {
                __uint128_t acc = 0;
                for (uint64_t ci = 0; ci < Ci; ci++) {
                    for (uint64_t m = 0; m < kernel_size; m++) {
                        for (uint64_t n = 0; n < kernel_size; n++) {
                            acc += (__uint128_t)padded_input({ci, i * stride + m, j * stride + n})
                                   * weight({co, ci, m, n});
                        }
                    }
                }
                expected({co, i, j}) = acc % plain_mod;
            }
        }
    }

    return expected;
}

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

int main(int argc, char **argv) {
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.parse(argc, argv);

    Utils::NetIO* netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
    HE::HEEvaluator HE(netio, party, 8192, 60, Datatype::HOST);
    HE.GenerateNewKey();

    struct Case {
        uint64_t Ci, Co, H, kernel, stride, padding;
    };

    std::vector<Case> cases = {
        {4, 4, 4, 3, 1, 1},
        {8, 8, 4, 3, 1, 1},
    };

    if (party == ALICE) {
        cout << "\n========== CirConv2D Test ==========\n";
        cout << "-----------------------------------------------------------\n";
        cout << setw(4) << "Ci" << setw(4) << "Co" << setw(4) << "H"
             << setw(4) << "K" << setw(6) << "blk" << setw(8) << "ntt"
             << setw(6) << "tile" << setw(8) << "status" << "\n";
        cout << "-----------------------------------------------------------\n";
    }

    for (const auto& tc : cases) {
        uint64_t Ci = tc.Ci, Co = tc.Co, H = tc.H;
        uint64_t kernelSize = tc.kernel, s = tc.stride, pad = tc.padding;

        for (uint64_t block_size = 1; block_size <= Ci; block_size *= 2) {
            if (Ci % block_size != 0 || Co % block_size != 0) continue;

            Tensor<uint64_t> input({Ci, H, H});
            Tensor<uint64_t> weight = create_block_circulant_conv_weight(Co, Ci, kernelSize, block_size);
            Tensor<uint64_t> bias({Co});

            if (party == ALICE) {
                for (uint64_t c = 0; c < Ci; c++)
                    for (uint64_t i = 0; i < H; i++)
                        for (uint64_t j = 0; j < H; j++)
                            input({c, i, j}) = (c + i + j + 1) % 5;
            }
            for (uint64_t i = 0; i < bias.size(); i++) bias(i) = 0;

            if (party != ALICE) {
                for (uint64_t i = 0; i < input.size(); i++) input(i) = 0;
            }

            Tensor<uint64_t> expected = compute_expected_conv(
                input, weight, Ci, Co, H, kernelSize, s, pad, HE.plain_mod);

            CirConv2D layer(H, s, pad, block_size, weight, bias, &HE);
            Tensor<uint64_t> output = layer(input);

            if (party == ALICE) {
                Tensor<uint64_t> output_peer(output.shape());
                netio->recv_data(output_peer.data().data(), output_peer.size() * sizeof(uint64_t));
                bool pass = verify_result(output, output_peer, expected, HE.plain_mod);

                cout << setw(4) << Ci << setw(4) << Co << setw(4) << H
                     << setw(4) << kernelSize << setw(6) << block_size
                     << setw(8) << layer.ntt_size << setw(6) << layer.tile_size
                     << setw(8) << (pass ? "PASS" : "FAIL") << "\n";

                if (!pass) {
                    uint64_t H_out = (H + 2 * pad - kernelSize) / s + 1;
                    cout << "  Expected vs Reconstructed (first mismatches):\n";
                    uint64_t shown = 0;
                    for (uint64_t i = 0; i < output.size() && shown < 5; i++) {
                        uint64_t recon = (output(i) + output_peer(i)) % HE.plain_mod;
                        if (recon != expected(i)) {
                            cout << "    [" << i << "] expected=" << expected(i)
                                 << " got=" << recon << "\n";
                            shown++;
                        }
                    }
                }
            } else {
                netio->send_data(output.data().data(), output.size() * sizeof(uint64_t));
            }
        }
    }

    if (party == ALICE) {
        cout << "-----------------------------------------------------------\n";
    }

    delete netio;
    return 0;
}
