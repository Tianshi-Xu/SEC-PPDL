#pragma once
#include <HE/HE.h>

namespace LinearOperator {
    Tensor<uint64_t> ElementWiseMul(const Tensor<uint64_t> &x, const Tensor<uint64_t> &y, HE::HEEvaluator* HE);
} // namespace LinearOperator