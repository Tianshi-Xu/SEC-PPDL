#include "PIR/PirAnswerGenerator.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace secppdl::pir
{

void print_overhead_estimates(
    std::size_t partitions, std::size_t chunks, std::size_t poly_degree, std::size_t num_primes)
{
    constexpr long double kPcieBandwidthBytesPerSec = 24.0L * 1000.0L * 1000.0L * 1000.0L;

    const long double num_plaintexts = static_cast<long double>(partitions) * static_cast<long double>(chunks);
    const long double bytes_per_plaintext =
        static_cast<long double>(poly_degree) * static_cast<long double>(num_primes) * 8.0L;
    const long double total_db_bytes = num_plaintexts * bytes_per_plaintext;
    const long double transfer_seconds = total_db_bytes / kPcieBandwidthBytesPerSec;
    const long double transfer_ms = transfer_seconds * 1000.0L;

    const long double total_mib = total_db_bytes / (1024.0L * 1024.0L);
    const long double total_gib = total_db_bytes / (1024.0L * 1024.0L * 1024.0L);

    std::cout << "\n===== Theoretical Overhead Estimates =====" << std::endl;
    std::cout << "Assumption: each DB plaintext polynomial footprint = poly_degree * num_primes * 8 bytes" << std::endl;
    std::cout << "  #DB plaintexts: " << static_cast<unsigned long long>(partitions * chunks) << std::endl;
    std::cout << "  bytes/plaintext: " << static_cast<unsigned long long>(bytes_per_plaintext) << " B" << std::endl;
    std::cout << "Storage Overhead (VRAM for loaded DB plaintexts): "
              << static_cast<unsigned long long>(total_db_bytes) << " B (" << std::fixed << std::setprecision(3)
              << total_mib << " MiB, " << total_gib << " GiB)" << std::endl;
    std::cout << "I/O Bandwidth Overhead (PCIe Gen4 x16 @ 24 GB/s): " << static_cast<double>(transfer_ms) << " ms ("
              << static_cast<double>(transfer_seconds) << " s)" << std::endl;
}

void print_chunk_spot_check(
    const std::vector<PhantomCiphertext> &chunk_answers_before_rotation, const PirShape &shape,
    const DbValueTensor &db_values, const PhantomContext &context, PhantomSecretKey &secret_key,
    const PhantomBatchEncoder &batch_encoder)
{
    if (chunk_answers_before_rotation.empty())
    {
        return;
    }

    std::cout << "\n===== Retrieval Spot Check (GPU pre-rotation chunks) =====" << std::endl;
    const std::size_t max_print_chunks = std::min<std::size_t>(shape.chunks, 8);

    for (std::size_t k = 0; k < max_print_chunks; ++k)
    {
        PhantomPlaintext chunk_pt;
        secret_key.decrypt(context, chunk_answers_before_rotation[k], chunk_pt);

        std::vector<uint64_t> decoded;
        batch_encoder.decode(context, chunk_pt, decoded);

        const uint64_t expected = db_values.at(shape.r_idx, shape.c_block, k, shape.oft);
        const uint64_t actual = shape.oft < decoded.size() ? decoded[shape.oft] : 0ULL;
        const bool ok = shape.oft < decoded.size() && (actual == expected);

        std::cout << "chunk[" << k << "] slot[" << shape.oft << "] expected=" << expected << ", eval=" << actual;
        if (ok)
        {
            std::cout << " (OK)";
        }
        else
        {
            std::cout << " (MISMATCH)";
        }
        std::cout << std::endl;
    }
}

bool verify_batch_results(
    const std::vector<PirGpuComputeResult> &results, const std::vector<QueryIndex> &query_indices,
    const DbValueTensor &db_values, const PirShape &shape, const PhantomContext &context, PhantomSecretKey &secret_key,
    const PhantomBatchEncoder &batch_encoder)
{
    if (results.size() != query_indices.size())
    {
        throw std::logic_error("results/query_indices size mismatch.");
    }

    bool all_ok = true;
    for (std::size_t b = 0; b < results.size(); ++b)
    {
        if (results[b].chunk_answers_before_rotation.size() != shape.chunks)
        {
            std::cerr << "[verify] query " << b << " missing chunk_answers_before_rotation." << std::endl;
            all_ok = false;
            continue;
        }

        const QueryIndex &q = query_indices[b];

        for (std::size_t k = 0; k < shape.chunks; ++k)
        {
            PhantomPlaintext chunk_pt;
            secret_key.decrypt(context, results[b].chunk_answers_before_rotation[k], chunk_pt);

            std::vector<uint64_t> decoded;
            batch_encoder.decode(context, chunk_pt, decoded);

            const uint64_t expected = db_values.at(q.r_idx, q.c_block, k, q.oft);
            const uint64_t actual = q.oft < decoded.size() ? decoded[q.oft] : 0ULL;
            const bool ok = q.oft < decoded.size() && (actual == expected);

            if (!ok)
            {
                all_ok = false;
                std::cerr << "[verify] mismatch: b=" << b << ", query_id=" << q.query_id << ", chunk=" << k
                          << ", slot=" << q.oft << ", expected=" << expected << ", actual=" << actual << std::endl;
            }
        }
    }

    return all_ok;
}

} // namespace secppdl::pir
