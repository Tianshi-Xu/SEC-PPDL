#pragma once

#include <cstdint>
#include <vector>

#include <phantom/batchencoder.h>
#include <phantom/context.cuh>
#include <phantom/secretkey.h>

#include "PIR/PirTypes.h"

namespace secppdl::pir
{

std::vector<uint32_t> required_galois_elts_for_pir(uint32_t poly_degree, uint32_t m, uint32_t chunks);
std::vector<PhantomCiphertext> expand_query_sealpir_device(
    const PhantomCiphertext &encrypted, uint32_t m, const PhantomContext &context, const PhantomGaloisKey &galois_keys);

PirQueryBundle generate_query_bundle(
    const PirShape &shape, uint64_t plain_mod, const PhantomContext &context, PhantomPublicKey &public_key,
    const PhantomBatchEncoder &batch_encoder);

BatchQueryBundle generate_batch_query_bundle(
    const PirShape &shape, const std::vector<QueryIndex> &query_indices, uint64_t plain_mod, const PhantomContext &context,
    PhantomPublicKey &public_key, const PhantomBatchEncoder &batch_encoder);

} // namespace secppdl::pir
