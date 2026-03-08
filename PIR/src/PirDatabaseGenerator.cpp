#include "PIR/PirDatabaseGenerator.h"

#include <random>
#include <vector>

namespace secppdl::pir
{

DbValueTensor generate_db_values(const PirShape &shape, uint64_t plain_mod, std::mt19937_64 &gen)
{
    std::uniform_int_distribution<uint64_t> dist;
    DbValueTensor db(shape.h, shape.blocks_per_row, shape.chunks, shape.slot_count);

    for (std::size_t r = 0; r < shape.h; ++r)
    {
        for (std::size_t c = 0; c < shape.blocks_per_row; ++c)
        {
            for (std::size_t k = 0; k < shape.chunks; ++k)
            {
                for (std::size_t s = 0; s < shape.slot_count; ++s)
                {
                    db.at(r, c, k, s) = dist(gen) % plain_mod;
                }
            }
        }
    }

    return db;
}

Tensor3D<PhantomPlaintext> encode_db_to_device(
    const DbValueTensor &db_values, const PirShape &shape, const PhantomContext &context,
    const PhantomBatchEncoder &batch_encoder)
{
    Tensor3D<PhantomPlaintext> db_plain(shape.h, shape.blocks_per_row, shape.chunks);
    std::vector<uint64_t> chunk_values(shape.slot_count, 0ULL);

    for (std::size_t r = 0; r < shape.h; ++r)
    {
        for (std::size_t c = 0; c < shape.blocks_per_row; ++c)
        {
            for (std::size_t k = 0; k < shape.chunks; ++k)
            {
                db_values.export_chunk(r, c, k, chunk_values);
                PhantomPlaintext plain;
                batch_encoder.encode(context, chunk_values, plain);
                db_plain.at(r, c, k) = std::move(plain);
            }
        }
    }

    return db_plain;
}

} // namespace secppdl::pir
