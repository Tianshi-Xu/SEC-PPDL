#include <seal/seal.h>
#include <Datatype/Tensor.h>
#include <HE/HE.h>
#include "HE/unified/UnifiedPlaintext.h"
#include <HE/unified/UnifiedCiphertext.h>
#pragma once
using namespace HE::unified;
using namespace Datatype;
namespace Operator {
// let the last dimension of x be N, the polynomial degree
Tensor<UnifiedCiphertext> SSToHE(Tensor<uint64_t> x, HE::HEEvaluator* HE);


Tensor<uint64_t> HEToSS(Tensor<UnifiedCiphertext> out_ct, HE::HEEvaluator* HE);
}