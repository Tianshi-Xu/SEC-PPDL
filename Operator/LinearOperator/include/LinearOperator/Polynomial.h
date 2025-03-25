#pragma once
#include <HE/HE.h>

namespace LinearOperator {
    Tensor<uint64_t> ElementWiseMul(Tensor<uint64_t> &x, Tensor<uint64_t> &y, HE::HEEvaluator* HE);
} // namespace LinearOperator