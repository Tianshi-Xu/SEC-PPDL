#pragma once

#include <seal/seal.h>
#include <Datatype/Tensor.h>
#include <HE/HE.h>
#include <HE/unified/UnifiedCiphertext.h>
#include <seal/util/polyarithsmallmod.h>

using namespace Datatype;
namespace Operator {
// let the last dimension of x be N, the polynomial degree
Tensor<HE::unified::UnifiedCiphertext> SSToHE(const Tensor<uint64_t> &x, HE::HEEvaluator* HE);

Tensor<uint64_t> HEToSS(Tensor<HE::unified::UnifiedCiphertext> out_ct, HE::HEEvaluator* HE);

Tensor<HE::unified::UnifiedCiphertext> SSToHE_coeff(const Tensor<uint64_t> &x, HE::HEEvaluator* HE);

Tensor<uint64_t> HEToSS_coeff(Tensor<HE::unified::UnifiedCiphertext> &out_ct, HE::HEEvaluator* HE);



} // namespace Operator
