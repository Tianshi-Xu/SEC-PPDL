#include <LinearOperator/OptiEncode.h>
#include <Utils/ParallelFor.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace Operator {
namespace {
size_t DivideUp(size_t a, size_t b) { return a / b + (a % b != 0); }
size_t PowerOfTwoFloor(size_t value) {
    size_t result = 1;
    while (result <= value / 2) result *= 2;
    return result;
}
void AddTerm(std::vector<uint64_t> &poly, int64_t exponent, uint64_t value, uint64_t modulus) {
    const int64_t period = 2 * static_cast<int64_t>(poly.size());
    exponent = (exponent % period + period) % period;
    const size_t index = static_cast<size_t>(exponent) % poly.size();
    if (exponent >= static_cast<int64_t>(poly.size()) && value) value = modulus - value;
    poly[index] = (poly[index] + value) % modulus;
}
}
uint64_t OptiEncodeMultiplyMod(uint64_t a, uint64_t b, uint64_t modulus) {
    return static_cast<uint64_t>(static_cast<__uint128_t>(a) * b % modulus);
}
uint64_t OptiEncodePowerMod(uint64_t base, uint64_t power, uint64_t modulus) {
    uint64_t result = 1;
    for (; power; power >>= 1, base = OptiEncodeMultiplyMod(base, base, modulus))
        if (power & 1) result = OptiEncodeMultiplyMod(result, base, modulus);
    return result;
}
OptiEncodeLayout::OptiEncodeLayout(size_t n, size_t h, size_t w, size_t ci, size_t co,
                                 size_t k, size_t s, size_t p, size_t requested_c, size_t tile)
    : degree(n), height(h), width(w), input_channels(ci), output_channels(co),
      kernel(k), stride(s), padding(p) {
    if (n < 8 || (n & (n - 1)) || n > (size_t(1) << 20) || !h || !w || !ci || !co || !k || !s)
        throw std::invalid_argument("Invalid OptiEncode dimensions");
    if (p > (std::numeric_limits<size_t>::max() - std::max(h, w)) / 2 ||
        h + 2 * p < k || w + 2 * p < k || k > n / k)
        throw std::invalid_argument("OptiEncode kernel/padding exceeds capacity");
    output_height = (h + 2 * p - k) / s + 1;
    output_width = (w + 2 * p - k) / s + 1;
    size_t limit_c = 1;
    while (limit_c * 2 <= (2 * n) / (limit_c * 2)) limit_c *= 2;
    size_t useful_c = 1;
    while (useful_c < std::max(ci, co) && useful_c < limit_c) useful_c *= 2;
    if (requested_c && ((requested_c & (requested_c - 1)) || requested_c > limit_c))
        throw std::invalid_argument("OptiEncode channel block must satisfy C^2 <= 2N");
    const size_t minimum_c = requested_c ? requested_c : 1;
    size_t side = 1;
    while (side + 1 <= (n / minimum_c) / (side + 1)) ++side;
    if (k > side) throw std::invalid_argument("Kernel does not fit requested channel block");
    const size_t tile_limit = (side - k) / s + 1;
    tile_height = std::min(output_height, tile ? std::min(tile, tile_limit) : tile_limit);
    tile_width = std::min(output_width, tile ? std::min(tile, tile_limit) : tile_limit);
    patch_height = (tile_height - 1) * s + k;
    patch_width = (tile_width - 1) * s + k;
    const size_t capacity = n / (patch_height * patch_width);
    channels = requested_c ? requested_c : PowerOfTwoFloor(std::min(capacity, useful_c));
    input_blocks = DivideUp(ci, channels);
    output_blocks = DivideUp(co, channels);
    tile_rows = DivideUp(output_height, tile_height);
    tile_columns = DivideUp(output_width, tile_width);
    baby_steps = 1;
    while (baby_steps * baby_steps < channels) baby_steps *= 2;
    giant_steps = channels / baby_steps;
    generator = channels == 1 ? 1 : static_cast<uint32_t>(1 + 2 * n / channels);
}
size_t OptiEncodeLayout::TileCount() const { return tile_rows * tile_columns; }
size_t OptiEncodeLayout::InputCiphertexts() const { return TileCount() * input_blocks; }
size_t OptiEncodeLayout::OutputCiphertexts() const { return TileCount() * output_blocks; }
size_t OptiEncodeLayout::OutputIndex(size_t row, size_t column, size_t channel) const {
    return channels * (row * stride * patch_width + column * stride) + channel;
}
Datatype::Tensor<uint64_t> OptiEncodeLayout::PackActivation(
    const Datatype::Tensor<uint64_t> &input, uint64_t modulus, size_t threads) const {
    if (input.shape() != std::vector<size_t>{input_channels, height, width})
        throw std::invalid_argument("OptiEncode input shape mismatch");
    Datatype::Tensor<uint64_t> packed({TileCount(), input_blocks, degree});
    Utils::ParallelFor(TileCount() * input_blocks, threads, [&](size_t task, size_t) {
        const size_t tile = task / input_blocks, block = task % input_blocks;
        const int64_t top = (tile / tile_columns) * tile_height * stride;
        const int64_t left = (tile % tile_columns) * tile_width * stride;
        for (size_t i = 0; i < patch_height; ++i) for (size_t j = 0; j < patch_width; ++j) {
            const int64_t row = top + static_cast<int64_t>(i) - static_cast<int64_t>(padding);
            const int64_t col = left + static_cast<int64_t>(j) - static_cast<int64_t>(padding);
            if (row < 0 || col < 0 || row >= static_cast<int64_t>(height) || col >= static_cast<int64_t>(width)) continue;
            for (size_t c = 0; c < channels && block * channels + c < input_channels; ++c)
                packed(task * degree + channels * (i * patch_width + j) + c) =
                    input({block * channels + c, static_cast<size_t>(row), static_cast<size_t>(col)}) % modulus;
        }
    });
    return packed;
}
Datatype::Tensor<uint64_t> OptiEncodeLayout::DepackResult(const Datatype::Tensor<uint64_t> &poly) const {
    if (poly.shape() != std::vector<size_t>{TileCount(), output_blocks, degree})
        throw std::invalid_argument("OptiEncode output polynomial shape mismatch");
    Datatype::Tensor<uint64_t> output({output_channels, output_height, output_width});
    for (size_t c = 0; c < output_channels; ++c) for (size_t i = 0; i < output_height; ++i)
        for (size_t j = 0; j < output_width; ++j) {
            const size_t tile = (i / tile_height) * tile_columns + j / tile_width;
            output({c, i, j}) = poly({tile, c / channels, OutputIndex(i % tile_height, j % tile_width, c % channels)});
        }
    return output;
}
std::vector<uint64_t> OptiEncodeLayout::PackWeight(const Datatype::Tensor<uint64_t> &weights,
    size_t input_block, size_t output_block, size_t automorphism, uint64_t modulus,
    bool normalize) const {
    if (modulus < 3 || (normalize && !(modulus & 1)) || modulus > (uint64_t(1) << 61))
        throw std::invalid_argument("Plaintext normalization requires an odd modulus <= 2^61");
    if (weights.shape() != std::vector<size_t>{output_channels, input_channels, kernel, kernel})
        throw std::invalid_argument("OptiEncode weight shape mismatch");
    uint64_t inverse_c = 1;
    if (normalize)
        for (size_t c = channels; c > 1; c >>= 1) inverse_c = OptiEncodeMultiplyMod(inverse_c, (modulus + 1) / 2, modulus);
    const uint64_t period = 2 * degree;
    const uint64_t g = OptiEncodePowerMod(generator, automorphism, period);
    const size_t giant = automorphism - automorphism % baby_steps;
    const uint64_t undo = OptiEncodePowerMod(generator, channels - giant, period);
    std::vector<uint64_t> result(degree, 0);
    for (size_t o = 0; o < channels && output_block * channels + o < output_channels; ++o)
        for (size_t c = 0; c < channels && input_block * channels + c < input_channels; ++c)
            for (size_t i = 0; i < kernel; ++i) for (size_t j = 0; j < kernel; ++j) {
                const int64_t exponent = -static_cast<int64_t>(channels * (i * patch_width + j) + c);
                const uint64_t wrapped = (exponent % static_cast<int64_t>(period) + static_cast<int64_t>(period)) % period;
                const uint64_t transformed = ((wrapped * g + o) * undo) % period;
                const uint64_t value = OptiEncodeMultiplyMod(weights({output_block * channels + o,
                    input_block * channels + c, i, j}) % modulus, inverse_c, modulus);
                AddTerm(result, transformed, value, modulus);
            }
    return result;
}
std::vector<uint32_t> OptiEncodeLayout::GaloisElements() const {
    std::vector<uint32_t> result;
    if (baby_steps > 1) result.push_back(generator);
    if (giant_steps > 1) result.push_back(static_cast<uint32_t>(OptiEncodePowerMod(generator, baby_steps, 2 * degree)));
    return result;
}
} // namespace Operator
