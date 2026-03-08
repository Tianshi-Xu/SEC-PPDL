#pragma once

#include <cstddef>
#include <vector>

#include <phantom/batchencoder.h>
#include <phantom/context.cuh>
#include <phantom/secretkey.h>

#include "PIR/PirTypes.h"

namespace secppdl::pir
{

void print_overhead_estimates(
    std::size_t partitions, std::size_t chunks, std::size_t poly_degree, std::size_t num_primes);

void print_chunk_spot_check(
    const std::vector<PhantomCiphertext> &chunk_answers_before_rotation, const PirShape &shape,
    const DbValueTensor &db_values, const PhantomContext &context, PhantomSecretKey &secret_key,
    const PhantomBatchEncoder &batch_encoder);

bool verify_batch_results(
    const std::vector<PirGpuComputeResult> &results, const std::vector<QueryIndex> &query_indices,
    const DbValueTensor &db_values, const PirShape &shape, const PhantomContext &context, PhantomSecretKey &secret_key,
    const PhantomBatchEncoder &batch_encoder);

} // namespace secppdl::pir
