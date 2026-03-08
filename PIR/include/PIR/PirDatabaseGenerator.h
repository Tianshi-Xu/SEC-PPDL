#pragma once

#include <cstdint>
#include <random>

#include <phantom/batchencoder.h>
#include <phantom/context.cuh>

#include "PIR/PirTypes.h"

namespace secppdl::pir
{

DbValueTensor generate_db_values(const PirShape &shape, uint64_t plain_mod, std::mt19937_64 &gen);
Tensor3D<PhantomPlaintext> encode_db_to_device(
    const DbValueTensor &db_values, const PirShape &shape, const PhantomContext &context,
    const PhantomBatchEncoder &batch_encoder);

} // namespace secppdl::pir
