#include <phantom/batchencoder.h>
#include <phantom/ciphertext.h>
#include <phantom/context.cuh>
#include <phantom/evaluate.cuh>
#include <phantom/plaintext.h>
#include <phantom/polymath.cuh>
#include <phantom/secretkey.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

using phantom::EncryptionParameters;
using phantom::scheme_type;
using phantom::arith::CoeffModulus;
using phantom::arith::PlainModulus;
using std::cin;
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

inline void print_line(int line_number)
{
    cout << "Line " << std::setw(3) << line_number << " --> ";
}

inline void print_parameters(const PhantomContext &context)
{
    const auto &context_data = context.key_context_data();
    const auto &parms = context_data.parms();

    string scheme_name;
    switch (parms.scheme())
    {
    case scheme_type::bfv:
        scheme_name = "BFV";
        break;
    case scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    case scheme_type::bgv:
        scheme_name = "BGV";
        break;
    default:
        throw std::invalid_argument("Unsupported scheme");
    }

    cout << "/" << endl;
    cout << "| Encryption parameters:" << endl;
    cout << "|   scheme: " << scheme_name << endl;
    cout << "|   poly_modulus_degree: " << parms.poly_modulus_degree() << endl;

    const auto &coeff_modulus = parms.coeff_modulus();
    size_t total_bits = 0;
    for (const auto &mod : coeff_modulus)
    {
        total_bits += static_cast<size_t>(mod.bit_count());
    }

    cout << "|   coeff_modulus size: " << total_bits << " (";
    for (size_t i = 0; i + 1 < coeff_modulus.size(); ++i)
    {
        cout << coeff_modulus[i].bit_count() << " + ";
    }
    cout << coeff_modulus.back().bit_count() << ") bits" << endl;

    if (parms.scheme() == scheme_type::bfv)
    {
        cout << "|   plain_modulus: " << parms.plain_modulus().value() << endl;
    }
    cout << "\\" << endl;
}

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
class Tensor2D
{
public:
    Tensor2D() = default;

    Tensor2D(size_t dim0, size_t dim1) : dim0_(dim0), dim1_(dim1), data_(dim0 * dim1)
    {}

    T &at(size_t i, size_t j)
    {
        return data_.at(i * dim1_ + j);
    }

    const T &at(size_t i, size_t j) const
    {
        return data_.at(i * dim1_ + j);
    }

private:
    size_t dim0_ = 0;
    size_t dim1_ = 0;
    vector<T> data_{};
};

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

struct PirUserInput
{
    size_t num = 0;
    size_t bit_width = 0;
    size_t query_id = 0;
};

struct PirShape
{
    size_t poly_modulus_degree = 0;
    size_t slot_count = 0;
    size_t h = 0;
    size_t w = 0;
    size_t blocks_per_row = 0;
    size_t chunks = 0;
    size_t r_idx = 0;
    size_t c_idx = 0;
    size_t c_block = 0;
    size_t oft = 0;
};

struct PirExecutionOptions
{
    size_t query_batch_size = 1;
    bool enable_ct_pt_fusion = false;
    bool enable_ct_ct_fusion = false;
    bool capture_chunk_answers = true;
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

PirUserInput read_user_input()
{
    PirUserInput input;
    cout << "INFO: Input the number of items (e.g., 2^15 (32768)):" << endl;
    cin >> input.num;
    cout << "INFO: Input the bit width (e.g., 64):" << endl;
    cin >> input.bit_width;
    cout << "INFO: Input the query id (e.g., 1, in [0,num-1]):" << endl;
    cin >> input.query_id;
    return input;
}

PirShape build_shape(const PirUserInput &input, size_t poly_modulus_degree, size_t slot_count)
{
    if (input.num == 0 || (input.num % poly_modulus_degree) != 0)
    {
        throw std::invalid_argument("num must be positive and divisible by poly_modulus_degree.");
    }
    if (input.bit_width == 0 || (input.bit_width % 16) != 0)
    {
        throw std::invalid_argument("bit width must be a positive multiple of 16.");
    }
    if (input.query_id >= input.num)
    {
        throw std::invalid_argument("query id must be in [0, num-1].");
    }

    PirShape shape;
    shape.poly_modulus_degree = poly_modulus_degree;
    shape.slot_count = slot_count;

    const size_t dim1 = input.num / poly_modulus_degree;
    const size_t h = static_cast<size_t>(std::llround(std::sqrt(static_cast<long double>(dim1))));
    if (h * h != dim1)
    {
        throw std::invalid_argument("SmartPIR setup requires num/poly_modulus_degree to be a perfect square.");
    }

    shape.h = h;
    shape.w = h * poly_modulus_degree;
    shape.blocks_per_row = shape.w / poly_modulus_degree;
    shape.chunks = input.bit_width / 16;

    if (shape.blocks_per_row == 0 || shape.blocks_per_row > poly_modulus_degree)
    {
        throw std::invalid_argument("blocks_per_row must be in [1, poly_modulus_degree].");
    }

    shape.r_idx = input.query_id / shape.w;
    shape.c_idx = input.query_id % shape.w;
    shape.c_block = shape.c_idx / poly_modulus_degree;
    shape.oft = input.query_id % poly_modulus_degree;

    if (shape.r_idx >= shape.h || shape.c_block >= shape.blocks_per_row)
    {
        throw std::invalid_argument("Computed SmartPIR indices are out of range.");
    }

    return shape;
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

void print_overhead_estimates(size_t partitions, size_t chunks, size_t poly_degree, size_t num_primes)
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

    cout << "\n===== Theoretical Overhead Estimates =====" << endl;
    cout << "Assumption: each DB plaintext polynomial footprint = poly_degree * num_primes * 8 bytes" << endl;
    cout << "  #DB plaintexts: " << static_cast<unsigned long long>(partitions * chunks) << endl;
    cout << "  bytes/plaintext: " << static_cast<unsigned long long>(bytes_per_plaintext) << " B" << endl;
    cout << "Storage Overhead (VRAM for loaded DB plaintexts): "
         << static_cast<unsigned long long>(total_db_bytes) << " B (" << std::fixed << std::setprecision(3)
         << total_mib << " MiB, " << total_gib << " GiB)" << endl;
    cout << "I/O Bandwidth Overhead (PCIe Gen4 x16 @ 24 GB/s): " << static_cast<double>(transfer_ms) << " ms ("
         << static_cast<double>(transfer_seconds) << " s)" << endl;
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

    uint64_t two_data = 2;
    PhantomPlaintext two;
    two.load(&two_data, context, 0, 1.0);

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

struct PirQueryBundle
{
    PhantomCiphertext compressed_x{};
    vector<PhantomCiphertext> row_selectors{};
};

PirQueryBundle generate_query_bundle(
    const PirShape &shape, uint64_t plain_mod, const PhantomContext &context, PhantomPublicKey &public_key,
    const PhantomBatchEncoder &batch_encoder)
{
    PirQueryBundle bundle;

    vector<uint64_t> query_x_coeff(shape.poly_modulus_degree, 0ULL);
    const uint64_t blocks_mod = shape.blocks_per_row % plain_mod;
    if (blocks_mod == 0)
    {
        throw std::invalid_argument("blocks_per_row has no inverse modulo plain_modulus.");
    }
    const uint64_t inv_scale = mod_inverse_u64(blocks_mod, plain_mod);
    query_x_coeff[shape.c_block] = inv_scale;

    PhantomPlaintext query_x_plain;
    query_x_plain.load(query_x_coeff.data(), context, 0, 1.0);

    vector<uint64_t> vec_ones(shape.slot_count, 1ULL);
    vector<uint64_t> vec_zeros(shape.slot_count, 0ULL);
    PhantomPlaintext query_y_one;
    PhantomPlaintext query_y_zero;
    batch_encoder.encode(context, vec_ones, query_y_one);
    batch_encoder.encode(context, vec_zeros, query_y_zero);

    public_key.encrypt_asymmetric(context, query_x_plain, bundle.compressed_x);

    bundle.row_selectors.assign(shape.h, PhantomCiphertext{});
    for (size_t r = 0; r < shape.h; ++r)
    {
        if (r == shape.r_idx)
        {
            public_key.encrypt_asymmetric(context, query_y_one, bundle.row_selectors[r]);
        }
        else
        {
            public_key.encrypt_asymmetric(context, query_y_zero, bundle.row_selectors[r]);
        }
    }

    return bundle;
}

struct PirGpuComputeResult
{
    PhantomCiphertext answer{};
    vector<PhantomCiphertext> chunk_answers_before_rotation{};
    double t_compute_muladd_ms = 0.0;
    double t_compute_rot_ms = 0.0;
};

class PirGpuExecutor
{
public:
    PirGpuExecutor(
        const PhantomContext &context, const PhantomRelinKey &relin_keys, const PhantomGaloisKey &galois_keys,
        PirExecutionOptions options)
        : context_(context), relin_keys_(relin_keys), galois_keys_(galois_keys), options_(options)
    {
        if (options_.query_batch_size != 1)
        {
            throw std::invalid_argument("query_batch_size > 1 is reserved for next batch-enabled version.");
        }
    }

    PirGpuComputeResult run(
        const Tensor3D<PhantomPlaintext> &db_plain, const vector<PhantomCiphertext> &expanded_x,
        const vector<PhantomCiphertext> &row_selectors, size_t row_count, size_t block_count, size_t chunk_count) const
    {
        if (row_selectors.size() != row_count)
        {
            throw std::invalid_argument("row_selectors size mismatch.");
        }
        if (expanded_x.size() != block_count)
        {
            throw std::invalid_argument("expanded_x size mismatch.");
        }

        PirGpuComputeResult result;

        Tensor2D<PhantomCiphertext> t_layer1(row_count, chunk_count);
        vector<PhantomCiphertext> t_layer2(chunk_count);

        result.t_compute_muladd_ms = time_phase(
            [&]() {
                // Extension point for ct-pt kernel fusion / row batching.
                for (size_t k = 0; k < chunk_count; ++k)
                {
                    for (size_t r = 0; r < row_count; ++r)
                    {
                        bool initialized = false;
                        PhantomCiphertext accum;

                        for (size_t c = 0; c < block_count; ++c)
                        {
                            PhantomCiphertext prod = expanded_x[c];
                            phantom::multiply_plain_inplace(context_, prod, db_plain.at(r, c, k));

                            if (!initialized)
                            {
                                accum = std::move(prod);
                                initialized = true;
                            }
                            else
                            {
                                phantom::add_inplace(context_, accum, prod);
                            }
                        }

                        t_layer1.at(r, k) = std::move(accum);
                    }
                }

                // Extension point for ct-ct + relinearize fusion.
                for (size_t k = 0; k < chunk_count; ++k)
                {
                    bool initialized = false;
                    PhantomCiphertext accum;

                    for (size_t r = 0; r < row_count; ++r)
                    {
                        PhantomCiphertext term = t_layer1.at(r, k);
                        if (options_.enable_ct_ct_fusion)
                        {
                            phantom::multiply_and_relin_inplace(context_, term, row_selectors[r], relin_keys_);
                        }
                        else
                        {
                            phantom::multiply_inplace(context_, term, row_selectors[r]);
                            phantom::relinearize_inplace(context_, term, relin_keys_);
                        }

                        if (!initialized)
                        {
                            accum = std::move(term);
                            initialized = true;
                        }
                        else
                        {
                            phantom::add_inplace(context_, accum, term);
                        }
                    }

                    t_layer2[k] = std::move(accum);
                }
            },
            true, "T_Compute_MulAdd");

        if (options_.capture_chunk_answers)
        {
            result.chunk_answers_before_rotation = t_layer2;
        }

        result.t_compute_rot_ms = time_phase(
            [&]() {
                for (size_t k = 1; k < chunk_count; ++k)
                {
                    phantom::rotate_inplace(context_, t_layer2[k], static_cast<int>(k), galois_keys_);
                }
            },
            true, "T_Compute_Rot");

        result.answer = t_layer2[0];
        for (size_t k = 1; k < chunk_count; ++k)
        {
            phantom::add_inplace(context_, result.answer, t_layer2[k]);
        }
        gpu_sync_or_throw("T_Compute_FinalAdd");

        return result;
    }

private:
    const PhantomContext &context_;
    const PhantomRelinKey &relin_keys_;
    const PhantomGaloisKey &galois_keys_;
    PirExecutionOptions options_{};
};

void print_chunk_spot_check(
    const vector<PhantomCiphertext> &chunk_answers_before_rotation, const PirShape &shape, const DbValueTensor &db_values,
    const PhantomContext &context, PhantomSecretKey &secret_key, const PhantomBatchEncoder &batch_encoder)
{
    if (chunk_answers_before_rotation.empty())
    {
        return;
    }

    cout << "\n===== Retrieval Spot Check (GPU pre-rotation chunks) =====" << endl;
    const size_t max_print_chunks = std::min<size_t>(shape.chunks, 8);

    for (size_t k = 0; k < max_print_chunks; ++k)
    {
        PhantomPlaintext chunk_pt;
        secret_key.decrypt(context, chunk_answers_before_rotation[k], chunk_pt);

        vector<uint64_t> decoded;
        batch_encoder.decode(context, chunk_pt, decoded);

        const uint64_t expected = db_values.at(shape.r_idx, shape.c_block, k, shape.oft);
        const uint64_t actual = shape.oft < decoded.size() ? decoded[shape.oft] : 0ULL;
        const bool ok = shape.oft < decoded.size() && (actual == expected);

        cout << "chunk[" << k << "] slot[" << shape.oft << "] expected=" << expected << ", eval=" << actual;
        if (ok)
        {
            cout << " (OK)";
        }
        else
        {
            cout << " (MISMATCH)";
        }
        cout << endl;
    }
}

} // namespace

int main()
{
    constexpr size_t kPolyModulusDegree = 32768;

    PirUserInput input = read_user_input();
    PirShape shape = build_shape(input, kPolyModulusDegree, kPolyModulusDegree);

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(kPolyModulusDegree);
    parms.set_coeff_modulus(CoeffModulus::Create(kPolyModulusDegree, { 54, 54, 54, 54 }));
    parms.set_plain_modulus(PlainModulus::Batching(kPolyModulusDegree, 17));
    parms.set_galois_elts(required_galois_elts_for_pir(
        static_cast<uint32_t>(kPolyModulusDegree), static_cast<uint32_t>(shape.blocks_per_row),
        static_cast<uint32_t>(shape.chunks)));

    PhantomContext context(parms);
    print_line(__LINE__);
    print_parameters(context);

    PhantomBatchEncoder batch_encoder(context);
    shape.slot_count = batch_encoder.slot_count();

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

    const uint64_t plain_mod = context.key_context_data().parms().plain_modulus().value();

    cout << "INFO: SmartPIR dimensions: h=" << shape.h << ", w=" << shape.w << ", w/N=" << shape.blocks_per_row
         << ", chunks=" << shape.chunks << endl;
    cout << "INFO: Query mapping: r_idx=" << shape.r_idx << ", c_idx=" << shape.c_idx
         << ", c_block=" << shape.c_block << ", oft=" << shape.oft << endl;
    cout << "INFO: Generating DB values on host and encoding DB plaintexts on GPU." << endl;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    DbValueTensor db_values = generate_db_values(shape, plain_mod, gen);

    PirPhaseLatency latency;

    Tensor3D<PhantomPlaintext> db_plain;
    latency.t_encode_ms = time_phase(
        [&]() {
            db_plain = encode_db_to_device(db_values, shape, context, batch_encoder);
        },
        true, "T_Encode");

    PirQueryBundle query_bundle;
    latency.t_encrypt_ms = time_phase(
        [&]() {
            query_bundle = generate_query_bundle(shape, plain_mod, context, public_key, batch_encoder);
        },
        true, "T_Encrypt");

    vector<PhantomCiphertext> expanded_x;
    latency.t_expand_ms = time_phase(
        [&]() {
            expanded_x = expand_query_sealpir_device(
                query_bundle.compressed_x, static_cast<uint32_t>(shape.blocks_per_row), context, galois_keys);
        },
        true, "T_ExpandX");

    PirExecutionOptions exec_options;
    exec_options.query_batch_size = 1;
    exec_options.enable_ct_pt_fusion = false;
    exec_options.enable_ct_ct_fusion = false;
    exec_options.capture_chunk_answers = true;

    PirGpuExecutor executor(context, relin_keys, galois_keys, exec_options);
    PirGpuComputeResult compute_result = executor.run(
        db_plain, expanded_x, query_bundle.row_selectors, shape.h, shape.blocks_per_row, shape.chunks);

    latency.t_compute_muladd_ms = compute_result.t_compute_muladd_ms;
    latency.t_compute_rot_ms = compute_result.t_compute_rot_ms;

    PhantomPlaintext answer_pt;
    latency.t_decrypt_ms = time_phase(
        [&]() {
            secret_key.decrypt(context, compute_result.answer, answer_pt);
        },
        true, "T_Decrypt");

    vector<uint64_t> answer_plain;
    latency.t_decode_ms = time_phase(
        [&]() {
            batch_encoder.decode(context, answer_pt, answer_plain);
        },
        true, "T_Decode");

    cout << "\n===== Phase Latency (ms) =====" << endl;
    cout << std::fixed << std::setprecision(3);
    cout << "T_Encode:         " << latency.t_encode_ms << endl;
    cout << "T_Encrypt:        " << latency.t_encrypt_ms << endl;
    cout << "T_ExpandX:        " << latency.t_expand_ms << endl;
    cout << "T_Compute_MulAdd: " << latency.t_compute_muladd_ms << endl;
    cout << "T_Compute_Rot:    " << latency.t_compute_rot_ms << endl;
    cout << "T_Decrypt:        " << latency.t_decrypt_ms << endl;
    cout << "T_Decode:         " << latency.t_decode_ms << endl;

    if (shape.oft < answer_plain.size())
    {
        cout << "INFO: Final answer slot[" << shape.oft << "] = " << answer_plain[shape.oft] << endl;
    }

    print_chunk_spot_check(
        compute_result.chunk_answers_before_rotation, shape, db_values, context, secret_key, batch_encoder);

    const size_t num_primes = context.key_context_data().parms().coeff_modulus().size();
    print_overhead_estimates(shape.h * shape.blocks_per_row, shape.chunks, shape.poly_modulus_degree, num_primes);

    return 0;
}
