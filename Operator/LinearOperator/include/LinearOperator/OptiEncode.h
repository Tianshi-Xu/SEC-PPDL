#pragma once
#include <Datatype/Tensor.h>
#include <cstdint>
#include <vector>

namespace Operator {
// Public layout only; all values are residues modulo the BFV plaintext modulus.
// Output tiles include their full input halo, including virtual zero padding.
struct OptiEncodeLayout {
    size_t degree, height, width, input_channels, output_channels;
    size_t kernel, stride, padding, output_height, output_width;
    size_t tile_height, tile_width, patch_height, patch_width;
    size_t channels, input_blocks, output_blocks, tile_rows, tile_columns;
    size_t baby_steps, giant_steps;
    uint32_t generator;

    OptiEncodeLayout(size_t degree, size_t height, size_t width, size_t input_channels,
                     size_t output_channels, size_t kernel, size_t stride, size_t padding,
                     size_t requested_channels = 0, size_t requested_tile_size = 0);
    size_t TileCount() const;
    size_t InputCiphertexts() const;
    size_t OutputCiphertexts() const;
    size_t OutputIndex(size_t row, size_t column, size_t channel) const;
    Datatype::Tensor<uint64_t> PackActivation(const Datatype::Tensor<uint64_t> &input,
                                            uint64_t modulus, size_t threads) const;
    Datatype::Tensor<uint64_t> DepackResult(const Datatype::Tensor<uint64_t> &polynomials) const;
    // Returns Q[input_block, output_block, automorphism] after inverse giant
    // step and C^-1 normalization. For j=b+a, this is
    // C^-1 sigma^(-b)(sum_o X^o sigma^j(w_o)). The caller chooses coefficient/NTT storage.
    std::vector<uint64_t> PackWeight(const Datatype::Tensor<uint64_t> &weights,
                                    size_t input_block, size_t output_block,
                                    size_t automorphism, uint64_t modulus,
                                    bool normalize = true) const;
    std::vector<uint32_t> GaloisElements() const;
};
uint64_t OptiEncodeMultiplyMod(uint64_t a, uint64_t b, uint64_t modulus);
uint64_t OptiEncodePowerMod(uint64_t base, uint64_t power, uint64_t modulus);
} // namespace Operator
