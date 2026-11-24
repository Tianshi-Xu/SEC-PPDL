#pragma once
#include <Datatype/Tensor.h>
#include <NonlinearOperator/FixPoint.h>

namespace QuantExperiment {
// Signed ring shares; operands/differences must fit the signed bitwidth.
Datatype::Tensor<uint64_t> MaximumShared(const Datatype::Tensor<uint64_t> &left,
    const Datatype::Tensor<uint64_t> &right, NonlinearOperator::FixPoint<uint64_t> &fixpoint, int bitwidth);

// CHW max pooling with negative-infinity padding. Inputs must have magnitude
// less than 2^(bitwidth-2), so comparisons with the padding sentinel are valid.
Datatype::Tensor<uint64_t> MaxPoolShared(const Datatype::Tensor<uint64_t> &input,
    size_t kernel, size_t stride, size_t padding,
    NonlinearOperator::FixPoint<uint64_t> &fixpoint, int bitwidth);

// Returns an additive share of the index; ties choose the first index.
Datatype::Tensor<uint64_t> ArgMaxShared(const Datatype::Tensor<uint64_t> &input,
    NonlinearOperator::FixPoint<uint64_t> &fixpoint, int bitwidth);
} // namespace QuantExperiment
