#include <phantom/batchencoder.h>
#include <phantom/ciphertext.h>
#include <phantom/context.cuh>
#include <phantom/evaluate.cuh>
#include <phantom/phantom_memory_pool.cuh>
#include <phantom/plaintext.h>
#include <phantom/polymath.cuh>
#include <phantom/secretkey.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <future>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

using phantom::EncryptionParameters;
using phantom::arith::CoeffModulus;
using phantom::arith::PlainModulus;
using phantom::scheme_type;
using std::cerr;
using std::cout;
using std::endl;
using std::size_t;
using std::string;
using std::vector;

namespace
{

class Timer
{
public:
    void start()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double stop_ms() const
    {
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_{};
};

inline void gpu_sync_or_throw(const string &tag)
{
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            "cudaDeviceSynchronize failed at " + tag + ": " + string(cudaGetErrorString(err)));
    }
}

template <typename Fn>
double time_phase(Fn &&work, bool gpu_phase, const string &phase_name)
{
    if (gpu_phase)
    {
        gpu_sync_or_throw(phase_name + "/pre");
    }

    Timer timer;
    timer.start();
    work();

    if (gpu_phase)
    {
        gpu_sync_or_throw(phase_name + "/post");
    }
    return timer.stop_ms();
}

uint32_t ceil_log2_u64(uint64_t x)
{
    if (x <= 1)
    {
        return 0;
    }

    uint32_t lg = 0;
    --x;
    while (x)
    {
        x >>= 1;
        ++lg;
    }
    return lg;
}

uint64_t mod_pow_u64(uint64_t base, uint64_t exp, uint64_t mod)
{
    uint64_t result = 1 % mod;
    base %= mod;
    while (exp)
    {
        if (exp & 1ULL)
        {
            result = static_cast<uint64_t>((__uint128_t)result * base % mod);
        }
        base = static_cast<uint64_t>((__uint128_t)base * base % mod);
        exp >>= 1ULL;
    }
    return result;
}

uint32_t mod_pow_u32(uint32_t base, uint32_t exp, uint32_t mod)
{
    uint64_t result = 1ULL % mod;
    uint64_t cur = base % mod;
    while (exp)
    {
        if (exp & 1U)
        {
            result = (result * cur) % mod;
        }
        cur = (cur * cur) % mod;
        exp >>= 1U;
    }
    return static_cast<uint32_t>(result);
}

uint64_t mod_inverse_u64(uint64_t a, uint64_t mod)
{
    if (a == 0 || mod <= 2)
    {
        throw std::invalid_argument("Invalid modular inverse input.");
    }
    return mod_pow_u64(a, mod - 2, mod);
}

template <typename T>
class Tensor3D
{
public:
    Tensor3D() = default;

    Tensor3D(size_t dim0, size_t dim1, size_t dim2)
        : dim0_(dim0), dim1_(dim1), dim2_(dim2), data_(dim0 * dim1 * dim2)
    {}

    T &at(size_t i, size_t j, size_t k)
    {
        return data_.at((i * dim1_ + j) * dim2_ + k);
    }

    const T &at(size_t i, size_t j, size_t k) const
    {
        return data_.at((i * dim1_ + j) * dim2_ + k);
    }

private:
    size_t dim0_ = 0;
    size_t dim1_ = 0;
    size_t dim2_ = 0;
    vector<T> data_{};
};

class DbValueTensor
{
public:
    DbValueTensor(size_t rows, size_t blocks, size_t chunks, size_t slots)
        : rows_(rows), blocks_(blocks), chunks_(chunks), slots_(slots), data_(rows * blocks * chunks * slots, 0ULL)
    {}

    uint64_t &at(size_t r, size_t c, size_t k, size_t s)
    {
        return data_.at((((r * blocks_) + c) * chunks_ + k) * slots_ + s);
    }

    const uint64_t &at(size_t r, size_t c, size_t k, size_t s) const
    {
        return data_.at((((r * blocks_) + c) * chunks_ + k) * slots_ + s);
    }

    void export_chunk(size_t r, size_t c, size_t k, vector<uint64_t> &out) const
    {
        out.resize(slots_);
        const size_t offset = (((r * blocks_) + c) * chunks_ + k) * slots_;
        std::copy_n(data_.data() + offset, slots_, out.data());
    }

private:
    size_t rows_ = 0;
    size_t blocks_ = 0;
    size_t chunks_ = 0;
    size_t slots_ = 0;
    vector<uint64_t> data_{};
};

struct PirShape
{
    size_t poly_modulus_degree = 0;
    size_t slot_count = 0;
    size_t h = 0;
    size_t w = 0;
    size_t blocks_per_row = 0;
    size_t chunks = 0;
};

struct QueryIndex
{
    size_t query_id = 0;
    size_t r_idx = 0;
    size_t c_idx = 0;
    size_t c_block = 0;
    size_t oft = 0;
};

struct PirGpuComputeResult
{
    PhantomCiphertext answer{};
    vector<PhantomCiphertext> chunk_answers_before_rotation{};
    double t_compute_muladd_ms = 0.0;
    double t_compute_rot_ms = 0.0;
};

struct BatchPirExecutionOptions
{
    bool enable_ct_pt_fusion = false;
    bool enable_ct_ct_fusion = false;
    bool fuse_rotate_and_reduce = true;
    bool capture_chunk_answers = true;
    size_t expand_group_size = 4;
};

struct BatchQueryBundle
{
    vector<PhantomCiphertext> batch_compressed_x{};
    vector<vector<PhantomCiphertext>> batch_row_selectors{};
};

struct CliInput
{
    size_t num = 32768;
    size_t bit_width = 16;
    size_t batch_size = 2;
    uint64_t seed = 7;
    int gpu_id = 0;
};

void print_usage(const char *prog)
{
    cout << "Usage: " << prog << " [num] [bit_width] [batch_size] [seed] [gpu_id]" << endl;
    cout << "  num: total DB items, must be divisible by poly_modulus_degree(32768), default=32768" << endl;
    cout << "  bit_width: must be multiple of 16, default=16" << endl;
    cout << "  batch_size: number of concurrent queries, default=2" << endl;
    cout << "  seed: RNG seed, default=7" << endl;
    cout << "  gpu_id: CUDA device id, default=0" << endl;
}

CliInput parse_cli(int argc, char **argv)
{
    CliInput input;

    if (argc > 1 && string(argv[1]) == "--help")
    {
        print_usage(argv[0]);
        std::exit(0);
    }

    if (argc > 1)
    {
        input.num = static_cast<size_t>(std::stoull(argv[1]));
    }
    if (argc > 2)
    {
        input.bit_width = static_cast<size_t>(std::stoull(argv[2]));
    }
    if (argc > 3)
    {
        input.batch_size = static_cast<size_t>(std::stoull(argv[3]));
    }
    if (argc > 4)
    {
        input.seed = static_cast<uint64_t>(std::stoull(argv[4]));
    }
    if (argc > 5)
    {
        input.gpu_id = static_cast<int>(std::stoi(argv[5]));
    }

    if (input.batch_size == 0)
    {
        throw std::invalid_argument("batch_size must be positive.");
    }
    if (input.gpu_id < 0)
    {
        throw std::invalid_argument("gpu_id must be non-negative.");
    }

    return input;
}

void select_cuda_device_or_throw(int gpu_id)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaGetDeviceCount failed: " + string(cudaGetErrorString(err)));
    }
    if (device_count <= 0)
    {
        throw std::runtime_error("No CUDA device found.");
    }
    if (gpu_id >= device_count)
    {
        throw std::invalid_argument("gpu_id out of range, available device count: " + std::to_string(device_count));
    }

    err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaSetDevice failed: " + string(cudaGetErrorString(err)));
    }

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, gpu_id);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaGetDeviceProperties failed: " + string(cudaGetErrorString(err)));
    }
    cout << "INFO: Using GPU[" << gpu_id << "]: " << prop.name << endl;
}

PirShape build_shape(size_t num, size_t bit_width, size_t poly_modulus_degree, size_t slot_count)
{
    if (num == 0 || (num % poly_modulus_degree) != 0)
    {
        throw std::invalid_argument("num must be positive and divisible by poly_modulus_degree.");
    }
    if (bit_width == 0 || (bit_width % 16) != 0)
    {
        throw std::invalid_argument("bit width must be a positive multiple of 16.");
    }

    PirShape shape;
    shape.poly_modulus_degree = poly_modulus_degree;
    shape.slot_count = slot_count;

    const size_t dim1 = num / poly_modulus_degree;
    const size_t h = static_cast<size_t>(std::llround(std::sqrt(static_cast<long double>(dim1))));
    if (h * h != dim1)
    {
        throw std::invalid_argument("SmartPIR setup requires num/poly_modulus_degree to be a perfect square.");
    }

    shape.h = h;
    shape.w = h * poly_modulus_degree;
    shape.blocks_per_row = shape.w / poly_modulus_degree;
    shape.chunks = bit_width / 16;

    if (shape.blocks_per_row == 0 || shape.blocks_per_row > poly_modulus_degree)
    {
        throw std::invalid_argument("blocks_per_row must be in [1, poly_modulus_degree].");
    }

    return shape;
}

QueryIndex map_query(size_t query_id, const PirShape &shape)
{
    if (query_id >= shape.h * shape.w)
    {
        throw std::invalid_argument("query_id out of range.");
    }

    QueryIndex q;
    q.query_id = query_id;
    q.r_idx = query_id / shape.w;
    q.c_idx = query_id % shape.w;
    q.c_block = q.c_idx / shape.poly_modulus_degree;
    q.oft = query_id % shape.poly_modulus_degree;

    if (q.r_idx >= shape.h || q.c_block >= shape.blocks_per_row)
    {
        throw std::invalid_argument("Computed SmartPIR indices are out of range.");
    }

    return q;
}

vector<uint32_t> required_galois_elts_for_pir(uint32_t poly_degree, uint32_t m, uint32_t chunks)
{
    vector<uint32_t> elts;
    const uint32_t logm = ceil_log2_u64(m);
    const uint32_t two_n = poly_degree << 1;

    for (uint32_t i = 0; i < logm; ++i)
    {
        elts.push_back((poly_degree + (1U << i)) >> i);
    }

    for (uint32_t step = 1; step < chunks; ++step)
    {
        elts.push_back(mod_pow_u32(3U, step, two_n));
    }

    std::sort(elts.begin(), elts.end());
    elts.erase(std::unique(elts.begin(), elts.end()), elts.end());
    return elts;
}

DbValueTensor generate_db_values(const PirShape &shape, uint64_t plain_mod, std::mt19937_64 &gen)
{
    std::uniform_int_distribution<uint64_t> dist;
    DbValueTensor db(shape.h, shape.blocks_per_row, shape.chunks, shape.slot_count);

    for (size_t r = 0; r < shape.h; ++r)
    {
        for (size_t c = 0; c < shape.blocks_per_row; ++c)
        {
            for (size_t k = 0; k < shape.chunks; ++k)
            {
                for (size_t s = 0; s < shape.slot_count; ++s)
                {
                    db.at(r, c, k, s) = dist(gen) % plain_mod;
                }
            }
        }
    }

    return db;
}

void multiply_power_of_x_device(
    const PhantomCiphertext &encrypted, PhantomCiphertext &destination, uint32_t index, const PhantomContext &context)
{
    destination = encrypted;

    const auto coeff_count = static_cast<uint32_t>(encrypted.poly_modulus_degree());
    const auto coeff_mod_count = static_cast<uint32_t>(encrypted.coeff_modulus_size());
    const auto poly_count = static_cast<uint32_t>(encrypted.size());
    if (coeff_count == 0 || coeff_mod_count == 0 || poly_count == 0)
    {
        return;
    }

    const auto full_mod_count = static_cast<uint32_t>(context.gpu_rns_tables().size());
    if (coeff_mod_count > full_mod_count)
    {
        throw std::runtime_error("multiply_power_of_x_device: coeff_mod_count exceeds context modulus size.");
    }

    const uint64_t *moduli_device = reinterpret_cast<const uint64_t *>(context.gpu_rns_tables().modulus());
    if (!moduli_device)
    {
        throw std::runtime_error("multiply_power_of_x_device: null device modulus pointer.");
    }

    launch_negacyclic_shift_kernel(
        encrypted.data(), destination.data(), coeff_count, coeff_mod_count, poly_count, index, moduli_device,
        cudaStreamPerThread);

    const auto launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error(
            "multiply_power_of_x_device launch failed: " + string(cudaGetErrorString(launch_err)));
    }
}

inline void apply_galois_device(
    const PhantomContext &context, const PhantomCiphertext &src, uint32_t galois_elt, const PhantomGaloisKey &galois_keys,
    PhantomCiphertext &dst)
{
    dst = src;
    phantom::apply_galois_inplace(context, dst, static_cast<size_t>(galois_elt), galois_keys);
}

vector<PhantomCiphertext> expand_query_sealpir_device(
    const PhantomCiphertext &encrypted, uint32_t m, const PhantomContext &context, const PhantomGaloisKey &galois_keys)
{
    if (m == 0)
    {
        throw std::invalid_argument("expand_query_sealpir_device: m must be positive.");
    }

    const uint32_t n = static_cast<uint32_t>(context.key_context_data().parms().poly_modulus_degree());
    const uint32_t logn = ceil_log2_u64(n);
    const uint32_t logm = ceil_log2_u64(m);
    if (logm > logn)
    {
        throw std::logic_error("expand_query_sealpir_device: m > poly_modulus_degree is not allowed.");
    }

    if (m == 1)
    {
        return vector<PhantomCiphertext>{ encrypted };
    }

    vector<uint32_t> galois_elts(logn);
    for (uint32_t i = 0; i < logn; ++i)
    {
        galois_elts[i] = (n + (1U << i)) >> i;
    }

    vector<uint64_t> two_coeff(static_cast<size_t>(n), 0ULL);
    two_coeff[0] = 2ULL;
    PhantomPlaintext two;
    two.load(two_coeff.data(), context, 0, 1.0);

    vector<PhantomCiphertext> temp{ encrypted };
    PhantomCiphertext temp_rotated;
    PhantomCiphertext temp_shifted;
    PhantomCiphertext temp_rotated_shifted;

    for (uint32_t i = 0; i + 1 < logm; ++i)
    {
        vector<PhantomCiphertext> newtemp(temp.size() << 1);
        const uint32_t index_raw = (n << 1) - (1U << i);
        const uint32_t index = (index_raw * galois_elts[i]) % (n << 1);

        for (size_t a = 0; a < temp.size(); ++a)
        {
            apply_galois_device(context, temp[a], galois_elts[i], galois_keys, temp_rotated);
            newtemp[a] = temp[a];
            phantom::add_inplace(context, newtemp[a], temp_rotated);

            multiply_power_of_x_device(temp[a], temp_shifted, index_raw, context);
            multiply_power_of_x_device(temp_rotated, temp_rotated_shifted, index, context);
            newtemp[a + temp.size()] = temp_shifted;
            phantom::add_inplace(context, newtemp[a + temp.size()], temp_rotated_shifted);
        }

        temp.swap(newtemp);
    }

    vector<PhantomCiphertext> newtemp(temp.size() << 1);
    const uint32_t index_raw = (n << 1) - (1U << (logm - 1));
    const uint32_t index = (index_raw * galois_elts[logm - 1]) % (n << 1);

    for (size_t a = 0; a < temp.size(); ++a)
    {
        if (a >= (m - (1U << (logm - 1))))
        {
            newtemp[a] = temp[a];
            phantom::multiply_plain_inplace(context, newtemp[a], two);
        }
        else
        {
            apply_galois_device(context, temp[a], galois_elts[logm - 1], galois_keys, temp_rotated);
            newtemp[a] = temp[a];
            phantom::add_inplace(context, newtemp[a], temp_rotated);

            multiply_power_of_x_device(temp[a], temp_shifted, index_raw, context);
            multiply_power_of_x_device(temp_rotated, temp_rotated_shifted, index, context);
            newtemp[a + temp.size()] = temp_shifted;
            phantom::add_inplace(context, newtemp[a + temp.size()], temp_rotated_shifted);
        }
    }

    vector<PhantomCiphertext> expanded;
    expanded.reserve(m);
    for (uint32_t i = 0; i < m; ++i)
    {
        expanded.emplace_back(std::move(newtemp[i]));
    }
    return expanded;
}

Tensor3D<PhantomPlaintext> encode_db_to_device(
    const DbValueTensor &db_values, const PirShape &shape, const PhantomContext &context,
    const PhantomBatchEncoder &batch_encoder)
{
    Tensor3D<PhantomPlaintext> db_plain(shape.h, shape.blocks_per_row, shape.chunks);
    vector<uint64_t> chunk_values(shape.slot_count, 0ULL);

    for (size_t r = 0; r < shape.h; ++r)
    {
        for (size_t c = 0; c < shape.blocks_per_row; ++c)
        {
            for (size_t k = 0; k < shape.chunks; ++k)
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

void copy_ciphertext_device_fast(const PhantomCiphertext &src, PhantomCiphertext &dst)
{
    const bool same_layout = dst.size() == src.size() && dst.chain_index() == src.chain_index() &&
                             dst.coeff_modulus_size() == src.coeff_modulus_size() &&
                             dst.poly_modulus_degree() == src.poly_modulus_degree() && dst.data() != nullptr;
    if (!same_layout)
    {
        dst = src;
        return;
    }

    const size_t coeff_count = src.size() * src.coeff_modulus_size() * src.poly_modulus_degree();
    const size_t bytes = coeff_count * sizeof(uint64_t);
    const cudaError_t err = cudaMemcpyAsync(dst.data(), src.data(), bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("copy_ciphertext_device_fast cudaMemcpyAsync failed: " + string(cudaGetErrorString(err)));
    }
    dst.set_scale(src.scale());
    dst.set_ntt_form(src.is_ntt_form());
    dst.set_correction_factor(src.correction_factor());
    dst.SetNoiseScaleDeg(src.GetNoiseScaleDeg());
}

BatchQueryBundle generate_batch_query_bundle(
    const PirShape &shape, const vector<QueryIndex> &query_indices, uint64_t plain_mod, const PhantomContext &context,
    PhantomPublicKey &public_key, const PhantomBatchEncoder &batch_encoder)
{
    BatchQueryBundle bundle;

    const size_t batch_size = query_indices.size();
    bundle.batch_compressed_x.assign(batch_size, PhantomCiphertext{});
    bundle.batch_row_selectors.assign(batch_size, vector<PhantomCiphertext>(shape.h));

    vector<uint64_t> vec_ones(shape.slot_count, 1ULL);
    vector<uint64_t> vec_zeros(shape.slot_count, 0ULL);
    PhantomPlaintext query_y_one;
    PhantomPlaintext query_y_zero;
    batch_encoder.encode(context, vec_ones, query_y_one);
    batch_encoder.encode(context, vec_zeros, query_y_zero);

    const uint64_t blocks_mod = shape.blocks_per_row % plain_mod;
    if (blocks_mod == 0)
    {
        throw std::invalid_argument("blocks_per_row has no inverse modulo plain_modulus.");
    }
    const uint64_t inv_scale = mod_inverse_u64(blocks_mod, plain_mod);

    for (size_t b = 0; b < batch_size; ++b)
    {
        vector<uint64_t> query_x_coeff(shape.poly_modulus_degree, 0ULL);
        query_x_coeff[query_indices[b].c_block] = inv_scale;

        PhantomPlaintext query_x_plain;
        query_x_plain.load(query_x_coeff.data(), context, 0, 1.0);

        public_key.encrypt_asymmetric(context, query_x_plain, bundle.batch_compressed_x[b]);

        for (size_t r = 0; r < shape.h; ++r)
        {
            if (r == query_indices[b].r_idx)
            {
                public_key.encrypt_asymmetric(context, query_y_one, bundle.batch_row_selectors[b][r]);
            }
            else
            {
                public_key.encrypt_asymmetric(context, query_y_zero, bundle.batch_row_selectors[b][r]);
            }
        }
    }

    return bundle;
}

class BatchPirGpuExecutor
{
public:
    BatchPirGpuExecutor(
        const PhantomContext &context, const PhantomRelinKey &relin_keys, const PhantomGaloisKey &galois_keys,
        BatchPirExecutionOptions options)
        : context_(context), relin_keys_(relin_keys), galois_keys_(galois_keys), options_(std::move(options))
    {
        // Keep using Phantom's global CUB caching allocator path.
        (void)phantom_device_allocator();
    }

    vector<PirGpuComputeResult> run(
        const Tensor3D<PhantomPlaintext> &db_plain, size_t batch_size,
        const vector<PhantomCiphertext> &batch_compressed_x,
        const vector<vector<PhantomCiphertext>> &batch_row_selectors, size_t row_count, size_t block_count,
        size_t chunk_count) const
    {
        if (batch_size == 0)
        {
            return {};
        }
        if (batch_compressed_x.size() != batch_size)
        {
            throw std::invalid_argument("batch_compressed_x size mismatch.");
        }
        if (batch_row_selectors.size() != batch_size)
        {
            throw std::invalid_argument("batch_row_selectors size mismatch.");
        }
        if (row_count == 0 || block_count == 0 || chunk_count == 0)
        {
            throw std::invalid_argument("row_count, block_count, and chunk_count must be positive.");
        }

        for (size_t b = 0; b < batch_size; ++b)
        {
            if (batch_row_selectors[b].size() != row_count)
            {
                throw std::invalid_argument("batch_row_selectors[b] row_count mismatch.");
            }
        }

        vector<vector<PhantomCiphertext>> batch_expanded_x(batch_size);
        time_phase(
            [&]() {
                const size_t group_size = std::max<size_t>(1, options_.expand_group_size);
                for (size_t base = 0; base < batch_size; base += group_size)
                {
                    const size_t end = std::min(batch_size, base + group_size);
                    for (size_t b = base; b < end; ++b)
                    {
                        batch_expanded_x[b] = expand_query_sealpir_device(
                            batch_compressed_x[b], static_cast<uint32_t>(block_count), context_, galois_keys_);
                        if (batch_expanded_x[b].size() != block_count)
                        {
                            throw std::logic_error("expand_query_sealpir_device block_count mismatch.");
                        }
                    }
                }
            },
            true, "Batch_T_ExpandX");
        if (batch_expanded_x.front().empty())
        {
            throw std::invalid_argument("expanded query selectors are empty.");
        }
        const size_t query_chain_index = batch_expanded_x.front().front().chain_index();
        // Step 2.1: transform expanded selectors to NTT once.
        time_phase(
            [&]() {
                for (size_t b = 0; b < batch_size; ++b)
                {
                    for (size_t c = 0; c < block_count; ++c)
                    {
                        PhantomCiphertext &selector = batch_expanded_x[b][c];
                        if (!selector.is_ntt_form())
                        {
                            phantom::transform_to_ntt_inplace(context_, selector);
                        }
                        if (selector.chain_index() != query_chain_index)
                        {
                            throw std::logic_error("expanded_x chain index mismatch in lazy-INTT path.");
                        }
                    }
                }
            },
            true, "Batch_T_ExpandX_ToNTT");

        vector<PirGpuComputeResult> results(batch_size);
        vector<vector<PhantomCiphertext>> t_layer2(batch_size, vector<PhantomCiphertext>(chunk_count));

        const double t_compute_muladd_ms = time_phase(
            [&]() {
                vector<PhantomCiphertext> row_accum_ntt(batch_size);
                vector<PhantomCiphertext> scratch_prod_ntt(batch_size);
                vector<uint8_t> row_initialized(batch_size, 0U);
                vector<vector<PhantomCiphertext>> row_terms(batch_size, vector<PhantomCiphertext>(row_count));

                // DB-locality order: k -> r -> c outer; b is inner.
                for (size_t k = 0; k < chunk_count; ++k)
                {
                    for (size_t r = 0; r < row_count; ++r)
                    {
                        std::fill(row_initialized.begin(), row_initialized.end(), 0U);

                        for (size_t c = 0; c < block_count; ++c)
                        {
                            // Step 2.2: plain is transformed on-the-fly to NTT for this c.
                            PhantomPlaintext plain_ntt = db_plain.at(r, c, k);
                            if (!plain_ntt.is_ntt_form())
                            {
                                phantom::transform_to_ntt_inplace(context_, plain_ntt, query_chain_index);
                            }
                            else if (plain_ntt.chain_index() != query_chain_index)
                            {
                                throw std::logic_error("plain_ntt chain index mismatch in lazy-INTT path.");
                            }

                            for (size_t b = 0; b < batch_size; ++b)
                            {
                                if (!row_initialized[b])
                                {
                                    copy_ciphertext_device_fast(batch_expanded_x[b][c], row_accum_ntt[b]);
                                    phantom::multiply_plain_ntt_inplace(context_, row_accum_ntt[b], plain_ntt);
                                    row_initialized[b] = 1U;
                                }
                                else
                                {
                                    if (options_.enable_ct_pt_fusion)
                                    {
                                        // Step 2.3: NTT-domain fused multiply-add.
                                        phantom::multiply_plain_ntt_and_add_inplace(
                                            context_, batch_expanded_x[b][c], plain_ntt, row_accum_ntt[b]);
                                    }
                                    else
                                    {
                                        copy_ciphertext_device_fast(batch_expanded_x[b][c], scratch_prod_ntt[b]);
                                        phantom::multiply_plain_ntt_inplace(context_, scratch_prod_ntt[b], plain_ntt);
                                        phantom::add_inplace(context_, row_accum_ntt[b], scratch_prod_ntt[b]);
                                    }
                                }
                            }
                        }

                        for (size_t b = 0; b < batch_size; ++b)
                        {
                            if (!row_initialized[b])
                            {
                                throw std::logic_error("row_accum is uninitialized.");
                            }
                            // Step 2.4: lazy INTT once per (b,r,k), after finishing c-loop.
                            phantom::transform_from_ntt_inplace(context_, row_accum_ntt[b]);
                            if (options_.enable_ct_ct_fusion)
                            {
                                phantom::multiply_and_relin_inplace(
                                    context_, row_accum_ntt[b], batch_row_selectors[b][r], relin_keys_);
                            }
                            else
                            {
                                phantom::multiply_inplace(context_, row_accum_ntt[b], batch_row_selectors[b][r]);
                                phantom::relinearize_inplace(context_, row_accum_ntt[b], relin_keys_);
                            }
                            copy_ciphertext_device_fast(row_accum_ntt[b], row_terms[b][r]);
                        }
                    }
                    // Step 3.1: batch row accumulation with add_many to reduce add launches.
                    for (size_t b = 0; b < batch_size; ++b)
                    {
                        phantom::add_many(context_, row_terms[b], t_layer2[b][k]);
                    }
                }
            },
            true, "Batch_T_Compute_MulAddLazyINTT");

        if (options_.capture_chunk_answers)
        {
            for (size_t b = 0; b < batch_size; ++b)
            {
                results[b].chunk_answers_before_rotation = t_layer2[b];
            }
        }

        vector<PhantomCiphertext> answers(batch_size);
        const double t_compute_rot_ms = time_phase(
            [&]() {
                for (size_t b = 0; b < batch_size; ++b)
                {
                    answers[b] = t_layer2[b][0];
                }

                if (chunk_count <= 1)
                {
                    return;
                }

                // Step 3.2: async rotate dispatch by chunk; each worker uses a thread-local stream.
                int current_device = 0;
                cudaError_t device_err = cudaGetDevice(&current_device);
                if (device_err != cudaSuccess)
                {
                    throw std::runtime_error(
                        "cudaGetDevice failed before async rotate: " + string(cudaGetErrorString(device_err)));
                }
                vector<std::future<void>> rotate_jobs;
                rotate_jobs.reserve(chunk_count - 1);
                for (size_t k = 1; k < chunk_count; ++k)
                {
                    rotate_jobs.emplace_back(std::async(std::launch::async, [&, k, current_device]() {
                        cudaError_t set_err = cudaSetDevice(current_device);
                        if (set_err != cudaSuccess)
                        {
                            throw std::runtime_error(
                                "cudaSetDevice failed in async rotate worker: " + string(cudaGetErrorString(set_err)));
                        }
                        for (size_t b = 0; b < batch_size; ++b)
                        {
                            phantom::rotate_inplace(context_, t_layer2[b][k], static_cast<int>(k), galois_keys_);
                        }
                    }));
                }
                for (auto &job : rotate_jobs)
                {
                    job.get();
                }

                for (size_t b = 0; b < batch_size; ++b)
                {
                    for (size_t k = 1; k < chunk_count; ++k)
                    {
                        if (options_.fuse_rotate_and_reduce)
                        {
                            phantom::add_inplace(context_, answers[b], t_layer2[b][k]);
                        }
                    }
                }
            },
            true, options_.fuse_rotate_and_reduce ? "Batch_T_Compute_RotReduceAsync" : "Batch_T_Compute_RotAsync");

        for (size_t b = 0; b < batch_size; ++b)
        {
            if (options_.fuse_rotate_and_reduce)
            {
                results[b].answer = std::move(answers[b]);
            }
            else
            {
                results[b].answer = t_layer2[b][0];
                for (size_t k = 1; k < chunk_count; ++k)
                {
                    phantom::add_inplace(context_, results[b].answer, t_layer2[b][k]);
                }
            }
            results[b].t_compute_muladd_ms = t_compute_muladd_ms;
            results[b].t_compute_rot_ms = t_compute_rot_ms;
        }

        gpu_sync_or_throw("Batch_T_Compute_FinalAdd");
        return results;
    }

private:
    const PhantomContext &context_;
    const PhantomRelinKey &relin_keys_;
    const PhantomGaloisKey &galois_keys_;
    BatchPirExecutionOptions options_{};
};

bool verify_batch_results(
    const vector<PirGpuComputeResult> &results, const vector<QueryIndex> &query_indices, const DbValueTensor &db_values,
    const PirShape &shape, const PhantomContext &context, PhantomSecretKey &secret_key,
    const PhantomBatchEncoder &batch_encoder)
{
    if (results.size() != query_indices.size())
    {
        throw std::logic_error("results/query_indices size mismatch.");
    }

    bool all_ok = true;
    for (size_t b = 0; b < results.size(); ++b)
    {
        if (results[b].chunk_answers_before_rotation.size() != shape.chunks)
        {
            cerr << "[verify] query " << b << " missing chunk_answers_before_rotation." << endl;
            all_ok = false;
            continue;
        }

        const QueryIndex &q = query_indices[b];

        for (size_t k = 0; k < shape.chunks; ++k)
        {
            PhantomPlaintext chunk_pt;
            secret_key.decrypt(context, results[b].chunk_answers_before_rotation[k], chunk_pt);

            vector<uint64_t> decoded;
            batch_encoder.decode(context, chunk_pt, decoded);

            const uint64_t expected = db_values.at(q.r_idx, q.c_block, k, q.oft);
            const uint64_t actual = q.oft < decoded.size() ? decoded[q.oft] : 0ULL;
            const bool ok = q.oft < decoded.size() && (actual == expected);

            if (!ok)
            {
                all_ok = false;
                cerr << "[verify] mismatch: b=" << b << ", query_id=" << q.query_id << ", chunk=" << k
                     << ", slot=" << q.oft << ", expected=" << expected << ", actual=" << actual << endl;
            }
        }
    }

    return all_ok;
}

} // namespace

int main(int argc, char **argv)
{
    constexpr size_t kPolyModulusDegree = 32768;

    try
    {
        const CliInput input = parse_cli(argc, argv);
        select_cuda_device_or_throw(input.gpu_id);
        (void)phantom_device_allocator();

        PirShape shape = build_shape(input.num, input.bit_width, kPolyModulusDegree, kPolyModulusDegree);

        EncryptionParameters parms(scheme_type::bfv);
        parms.set_poly_modulus_degree(kPolyModulusDegree);
        parms.set_coeff_modulus(CoeffModulus::Create(kPolyModulusDegree, { 54, 54, 54, 54 }));
        parms.set_plain_modulus(PlainModulus::Batching(kPolyModulusDegree, 17));
        parms.set_galois_elts(required_galois_elts_for_pir(
            static_cast<uint32_t>(kPolyModulusDegree), static_cast<uint32_t>(shape.blocks_per_row),
            static_cast<uint32_t>(shape.chunks)));

        PhantomContext context(parms);
        PhantomBatchEncoder batch_encoder(context);
        shape.slot_count = batch_encoder.slot_count();

        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
        PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

        const uint64_t plain_mod = context.key_context_data().parms().plain_modulus().value();

        cout << "INFO: num=" << input.num << ", bit_width=" << input.bit_width << ", batch_size=" << input.batch_size
             << ", seed=" << input.seed << endl;
        cout << "INFO: SmartPIR shape: h=" << shape.h << ", w=" << shape.w << ", blocks_per_row="
             << shape.blocks_per_row << ", chunks=" << shape.chunks << endl;

        std::mt19937_64 gen(input.seed);
        DbValueTensor db_values = generate_db_values(shape, plain_mod, gen);

        Tensor3D<PhantomPlaintext> db_plain;
        const double t_encode_ms = time_phase(
            [&]() {
                db_plain = encode_db_to_device(db_values, shape, context, batch_encoder);
            },
            true, "T_Encode");

        vector<size_t> query_ids(input.batch_size, 0);
        std::uniform_int_distribution<size_t> qdist(0, input.num - 1);
        for (size_t b = 0; b < input.batch_size; ++b)
        {
            query_ids[b] = qdist(gen);
        }

        vector<QueryIndex> query_indices(input.batch_size);
        for (size_t b = 0; b < input.batch_size; ++b)
        {
            query_indices[b] = map_query(query_ids[b], shape);
            cout << "INFO: query[" << b << "] id=" << query_indices[b].query_id << " -> (r=" << query_indices[b].r_idx
                 << ", c_block=" << query_indices[b].c_block << ", oft=" << query_indices[b].oft << ")" << endl;
        }

        BatchQueryBundle query_bundle;
        const double t_encrypt_ms = time_phase(
            [&]() {
                query_bundle = generate_batch_query_bundle(
                    shape, query_indices, plain_mod, context, public_key, batch_encoder);
            },
            true, "T_Encrypt_Batch");

        BatchPirExecutionOptions options;
        options.enable_ct_pt_fusion = true;
        options.enable_ct_ct_fusion = true;
        options.fuse_rotate_and_reduce = true;
        options.capture_chunk_answers = true;
        options.expand_group_size = 4;

        BatchPirGpuExecutor executor(context, relin_keys, galois_keys, options);

        vector<PirGpuComputeResult> results;
        const double t_server_ms = time_phase(
            [&]() {
                results = executor.run(
                    db_plain, input.batch_size, query_bundle.batch_compressed_x, query_bundle.batch_row_selectors,
                    shape.h, shape.blocks_per_row, shape.chunks);
            },
            true, "T_Server_Batch");

        const bool ok = verify_batch_results(results, query_indices, db_values, shape, context, secret_key, batch_encoder);

        cout << "\n===== Batch GPU PIR Latency (ms) - Baseline =====" << endl;
        cout << std::fixed << std::setprecision(3);
        cout << "T_Encode:         " << t_encode_ms << endl;
        cout << "T_Encrypt_Batch:  " << t_encrypt_ms << endl;
        cout << "T_Server_Batch:   " << t_server_ms << endl;
        if (!results.empty())
        {
            cout << "T_Compute_MulAdd: " << results[0].t_compute_muladd_ms << endl;
            cout << "T_Compute_Rot:    " << results[0].t_compute_rot_ms << endl;
        }

        if (ok)
        {
            cout << "RESULT: PASS (all batched queries verified)." << endl;
            return 0;
        }

        cout << "RESULT: FAIL (verification mismatch)." << endl;
        return 1;
    }
    catch (const std::exception &e)
    {
        cerr << "ERROR: " << e.what() << endl;
        return 2;
    }
}
