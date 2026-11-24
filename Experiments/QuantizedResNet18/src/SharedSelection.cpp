#include <QuantExperiment/SharedSelection.h>
#include <algorithm>
#include <stdexcept>

namespace QuantExperiment {
using Datatype::Tensor;
Tensor<uint64_t> MaximumShared(const Tensor<uint64_t> &left, const Tensor<uint64_t> &right,
    NonlinearOperator::FixPoint<uint64_t> &fixpoint, int bitwidth) {
    if (left.shape() != right.shape() || bitwidth < 3 || bitwidth > 64)
        throw std::invalid_argument("Invalid shared maximum dimensions or bitwidth");
    auto difference = left - right;
    Tensor<uint8_t> choose_right(left.shape());
    fixpoint.less_than_zero(difference, choose_right, bitwidth);
    auto delta = right - left;
    Tensor<uint64_t> selected(left.shape());
    fixpoint.mux(choose_right, delta, selected, bitwidth, bitwidth);
    auto output = left + selected;
    output.bitwidth = bitwidth;
    output.scale = left.scale;
    return output;
}

Tensor<uint64_t> MaxPoolShared(const Tensor<uint64_t> &input, size_t kernel, size_t stride,
    size_t padding, NonlinearOperator::FixPoint<uint64_t> &fixpoint, int bitwidth) {
    const auto shape = input.shape();
    if (shape.size() != 3 || !kernel || !stride || !shape[0] || !shape[1] || !shape[2] ||
        bitwidth < 3 || bitwidth > 64 || padding >= kernel ||
        shape[1] + 2 * padding < kernel || shape[2] + 2 * padding < kernel)
        throw std::invalid_argument("Invalid shared max pooling parameters");
    const size_t oh = (shape[1] + 2 * padding - kernel) / stride + 1;
    const size_t ow = (shape[2] + 2 * padding - kernel) / stride + 1;
    const size_t count = shape[0] * oh * ow;
    size_t width = kernel * kernel;
    Tensor<uint64_t> candidates({count, width});
    const uint64_t sentinel = uint64_t(0) - (uint64_t(1) << (bitwidth - 2));
    for (size_t c = 0; c < shape[0]; ++c) for (size_t r = 0; r < oh; ++r)
        for (size_t col = 0; col < ow; ++col) for (size_t k = 0; k < width; ++k) {
            const int64_t row = static_cast<int64_t>(r * stride + k / kernel) - static_cast<int64_t>(padding);
            const int64_t column = static_cast<int64_t>(col * stride + k % kernel) - static_cast<int64_t>(padding);
            const bool valid = row >= 0 && column >= 0 && row < int64_t(shape[1]) && column < int64_t(shape[2]);
            candidates((c * oh + r) * ow * width + col * width + k) = valid
                ? input({c, size_t(row), size_t(column)}) : (fixpoint.party == ALICE ? sentinel : 0);
        }
    while (width > 1) {
        const size_t pairs = width / 2, next_width = (width + 1) / 2;
        Tensor<uint64_t> left({count * pairs}), right(left.shape());
        for (size_t i = 0; i < count; ++i) for (size_t p = 0; p < pairs; ++p) {
            left(i * pairs + p) = candidates(i * width + 2 * p);
            right(i * pairs + p) = candidates(i * width + 2 * p + 1);
        }
        const auto winners = MaximumShared(left, right, fixpoint, bitwidth);
        Tensor<uint64_t> next({count, next_width});
        for (size_t i = 0; i < count; ++i) {
            for (size_t p = 0; p < pairs; ++p) next(i * next_width + p) = winners(i * pairs + p);
            if (width & 1) next(i * next_width + pairs) = candidates(i * width + width - 1);
        }
        candidates = std::move(next);
        width = next_width;
    }
    candidates.reshape({shape[0], oh, ow});
    candidates.bitwidth = bitwidth;
    candidates.scale = input.scale;
    return candidates;
}

Tensor<uint64_t> ArgMaxShared(const Tensor<uint64_t> &input,
    NonlinearOperator::FixPoint<uint64_t> &fixpoint, int bitwidth) {
    if (input.shape().size() != 1 || input.size() == 0 || bitwidth < 3 || bitwidth > 64)
        throw std::invalid_argument("Argmax requires a nonempty vector and valid ring bitwidth");
    auto values = input;
    Tensor<uint64_t> indices(input.shape());
    if (fixpoint.party == ALICE) for (size_t i = 0; i < input.size(); ++i) indices(i) = i;
    while (values.size() > 1) {
        const size_t count = values.size(), pairs = count / 2;
        Tensor<uint64_t> difference({pairs});
        for (size_t i = 0; i < pairs; ++i) difference(i) = values(2*i) - values(2*i+1);
        Tensor<uint8_t> choose_right({pairs});
        fixpoint.less_than_zero(difference, choose_right, bitwidth);
        Tensor<uint8_t> choices({2*pairs});
        Tensor<uint64_t> delta({2*pairs}), selected(delta.shape());
        for (size_t i = 0; i < pairs; ++i) {
            choices(i) = choices(i+pairs) = choose_right(i);
            delta(i) = values(2*i+1) - values(2*i);
            delta(i+pairs) = indices(2*i+1) - indices(2*i);
        }
        fixpoint.mux(choices, delta, selected, bitwidth, bitwidth);
        Tensor<uint64_t> next_values({(count+1)/2}), next_indices(next_values.shape());
        for (size_t i = 0; i < pairs; ++i) {
            next_values(i) = values(2*i) + selected(i);
            next_indices(i) = indices(2*i) + selected(i+pairs);
        }
        if (count & 1) { next_values(pairs) = values(count-1); next_indices(pairs) = indices(count-1); }
        values = std::move(next_values);
        indices = std::move(next_indices);
    }
    indices.bitwidth = bitwidth;
    return indices;
}
} // namespace QuantExperiment
