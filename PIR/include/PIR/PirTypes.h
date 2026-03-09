#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <phantom/ciphertext.h>
#include <phantom/plaintext.h>

namespace secppdl::pir
{

template <typename T>
class Tensor3D
{
public:
    Tensor3D() = default;

    Tensor3D(std::size_t dim0, std::size_t dim1, std::size_t dim2)
        : dim0_(dim0), dim1_(dim1), dim2_(dim2), data_(dim0 * dim1 * dim2)
    {}

    T &at(std::size_t i, std::size_t j, std::size_t k)
    {
        return data_.at((i * dim1_ + j) * dim2_ + k);
    }

    const T &at(std::size_t i, std::size_t j, std::size_t k) const
    {
        return data_.at((i * dim1_ + j) * dim2_ + k);
    }

private:
    std::size_t dim0_ = 0;
    std::size_t dim1_ = 0;
    std::size_t dim2_ = 0;
    std::vector<T> data_{};
};

class DbValueTensor
{
public:
    DbValueTensor() = default;

    DbValueTensor(std::size_t rows, std::size_t blocks, std::size_t chunks, std::size_t slots)
        : rows_(rows), blocks_(blocks), chunks_(chunks), slots_(slots), data_(rows * blocks * chunks * slots, 0ULL)
    {}

    uint64_t &at(std::size_t r, std::size_t c, std::size_t k, std::size_t s)
    {
        return data_.at((((r * blocks_) + c) * chunks_ + k) * slots_ + s);
    }

    const uint64_t &at(std::size_t r, std::size_t c, std::size_t k, std::size_t s) const
    {
        return data_.at((((r * blocks_) + c) * chunks_ + k) * slots_ + s);
    }

    void export_chunk(std::size_t r, std::size_t c, std::size_t k, std::vector<uint64_t> &out) const
    {
        out.resize(slots_);
        const std::size_t offset = (((r * blocks_) + c) * chunks_ + k) * slots_;
        std::copy_n(data_.data() + offset, slots_, out.data());
    }

private:
    std::size_t rows_ = 0;
    std::size_t blocks_ = 0;
    std::size_t chunks_ = 0;
    std::size_t slots_ = 0;
    std::vector<uint64_t> data_{};
};

struct PirUserInput
{
    std::size_t num = 0;
    std::size_t bit_width = 0;
    std::size_t query_id = 0;
};

struct PirShape
{
    std::size_t poly_modulus_degree = 0;
    std::size_t slot_count = 0;
    std::size_t h = 0;
    std::size_t w = 0;
    std::size_t blocks_per_row = 0;
    std::size_t chunks = 0;
    std::size_t r_idx = 0;
    std::size_t c_idx = 0;
    std::size_t c_block = 0;
    std::size_t oft = 0;
};

struct QueryIndex
{
    std::size_t query_id = 0;
    std::size_t r_idx = 0;
    std::size_t c_idx = 0;
    std::size_t c_block = 0;
    std::size_t oft = 0;
};

struct PirExecutionOptions
{
    std::size_t query_batch_size = 1;
    bool enable_ct_pt_fusion = false;
    bool enable_ct_pt_2d_fusion = false;
    std::size_t ct_pt_2d_tile_rows = 32;
    bool enable_ct_ct_fusion = false;
    bool capture_chunk_answers = true;
};

struct BatchPirExecutionOptions
{
    bool enable_ct_pt_fusion = false;
    bool enable_ct_ct_fusion = false;
    bool fuse_rotate_and_reduce = true;
    bool capture_chunk_answers = true;
    std::size_t expand_group_size = 4;
};

struct PirPhaseLatency
{
    double t_encode_ms = 0.0;
    double t_encrypt_ms = 0.0;
    double t_expand_ms = 0.0;
    double t_compute_muladd_ms = 0.0;
    double t_compute_rot_ms = 0.0;
    double t_decrypt_ms = 0.0;
    double t_decode_ms = 0.0;
};

struct PirQueryBundle
{
    PhantomCiphertext compressed_x{};
    std::vector<PhantomCiphertext> row_selectors{};
};

struct BatchQueryBundle
{
    std::vector<PhantomCiphertext> batch_compressed_x{};
    std::vector<std::vector<PhantomCiphertext>> batch_row_selectors{};
};

struct PirGpuComputeResult
{
    PhantomCiphertext answer{};
    std::vector<PhantomCiphertext> chunk_answers_before_rotation{};
    double t_compute_muladd_ms = 0.0;
    double t_compute_rot_ms = 0.0;
};

struct CliInput
{
    std::size_t num = 32768;
    std::size_t bit_width = 16;
    std::size_t batch_size = 2;
    uint64_t seed = 7;
    int gpu_id = 0;
};

} // namespace secppdl::pir
