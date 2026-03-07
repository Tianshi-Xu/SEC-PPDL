#include <HE/unified/UnifiedCiphertext.h>
#include <HE/unified/UnifiedContext.h>
#include <HE/unified/UnifiedEncoder.h>
#include <HE/unified/UnifiedEvaluator.h>
#include <HE/unified/UnifiedEvk.h>
#include <HE/unified/UnifiedPlaintext.h>
#include <seal/seal.h>
#include <seal/util/polyarithsmallmod.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_HE_GPU
#include <cuda_runtime.h>
#include <phantom/polymath.cuh>
#endif

using HE::unified::UnifiedBatchEncoder;
using HE::unified::UnifiedCiphertext;
using HE::unified::UnifiedContext;
using HE::unified::UnifiedEvaluator;
using HE::unified::UnifiedGaloisKeys;
using HE::unified::UnifiedPlaintext;
using HE::unified::UnifiedRelinKeys;
using seal::CoeffModulus;
using seal::Decryptor;
using seal::Encryptor;
using seal::EncryptionParameters;
using seal::KeyGenerator;
using seal::PlainModulus;
using seal::Plaintext;
using seal::PublicKey;
using seal::RelinKeys;
using seal::SEALContext;
using seal::SecretKey;
using seal::scheme_type;
using seal::util::negacyclic_shift_poly_coeffmod;
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

inline void print_parameters(const SEALContext &context)
{
    const auto &context_data = *context.key_context_data();

    string scheme_name;
    switch (context_data.parms().scheme())
    {
    case seal::scheme_type::bfv:
        scheme_name = "BFV";
        break;
    case seal::scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    case seal::scheme_type::bgv:
        scheme_name = "BGV";
        break;
    default:
        throw std::invalid_argument("Unsupported scheme");
    }

    cout << "/" << endl;
    cout << "| Encryption parameters:" << endl;
    cout << "|   scheme: " << scheme_name << endl;
    cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << endl;

    cout << "|   coeff_modulus size: " << context_data.total_coeff_modulus_bit_count() << " (";
    const auto coeff_modulus = context_data.parms().coeff_modulus();
    for (size_t i = 0; i + 1 < coeff_modulus.size(); ++i)
    {
        cout << coeff_modulus[i].bit_count() << " + ";
    }
    cout << coeff_modulus.back().bit_count() << ") bits" << endl;

    if (context_data.parms().scheme() == seal::scheme_type::bfv)
    {
        cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << endl;
    }
    cout << "\\" << endl;
}

inline void gpu_sync_or_throw(const string &tag)
{
#ifdef USE_HE_GPU
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaDeviceSynchronize failed at " + tag + ": " +
                                 string(cudaGetErrorString(err)));
    }
#else
    (void)tag;
#endif
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
        // Required for asynchronous CUDA kernels: ensure phase timing is accurate.
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
    // plain_modulus in this benchmark is prime (65537), so use Fermat inverse.
    if (a == 0 || mod <= 2)
    {
        throw std::invalid_argument("Invalid modular inverse input.");
    }
    return mod_pow_u64(a, mod - 2, mod);
}

void multiply_power_of_x_unified(
    const UnifiedCiphertext &encrypted, UnifiedCiphertext &destination, uint32_t index, const UnifiedContext &context)
{
    destination = encrypted;
    if (encrypted.on_host())
    {
        const auto &src = encrypted.hcipher();
        auto &dst = destination.hcipher();
        const auto ctx_data = context.hcontext().get_context_data(src.parms_id());
        if (!ctx_data)
        {
            throw std::runtime_error("multiply_power_of_x_unified: invalid parms_id.");
        }

        const size_t coeff_count = ctx_data->parms().poly_modulus_degree();
        const auto &coeff_modulus = ctx_data->parms().coeff_modulus();
        const size_t coeff_mod_count = coeff_modulus.size();
        const size_t encrypted_count = src.size();

        for (size_t poly_idx = 0; poly_idx < encrypted_count; ++poly_idx)
        {
            for (size_t mod_idx = 0; mod_idx < coeff_mod_count; ++mod_idx)
            {
                negacyclic_shift_poly_coeffmod(
                    src.data(poly_idx) + (mod_idx * coeff_count), coeff_count, index, coeff_modulus[mod_idx],
                    dst.data(poly_idx) + (mod_idx * coeff_count));
            }
        }
        return;
    }

#ifdef USE_HE_GPU
    if (encrypted.on_device())
    {
        const auto &src = encrypted.dcipher();
        auto &dst = destination.dcipher();
        const auto coeff_count = static_cast<uint32_t>(src.poly_modulus_degree());
        const auto coeff_mod_count = static_cast<uint32_t>(src.coeff_modulus_size());
        const auto poly_count = static_cast<uint32_t>(src.size());
        if (coeff_count == 0 || coeff_mod_count == 0 || poly_count == 0)
        {
            return;
        }

        const auto full_mod_count = static_cast<uint32_t>(context.dcontext().gpu_rns_tables().size());
        if (coeff_mod_count > full_mod_count)
        {
            throw std::runtime_error("multiply_power_of_x_unified: coeff_mod_count exceeds context modulus size.");
        }

        const uint64_t *moduli_device = reinterpret_cast<const uint64_t *>(context.dcontext().gpu_rns_tables().modulus());
        if (!moduli_device)
        {
            throw std::runtime_error("multiply_power_of_x_unified: null device modulus pointer.");
        }

        launch_negacyclic_shift_kernel(
            src.data(), dst.data(), coeff_count, coeff_mod_count, poly_count, index, moduli_device,
            cudaStreamPerThread);

        const auto launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            throw std::runtime_error(
                "multiply_power_of_x_unified GPU shift launch failed: " + string(cudaGetErrorString(launch_err)));
        }
        return;
    }
#endif

    throw std::invalid_argument("multiply_power_of_x_unified requires ciphertext on host or device.");
}

inline void apply_galois_unified(
    const UnifiedEvaluator &evaluator, const UnifiedCiphertext &src, uint32_t galois_elt,
    const UnifiedGaloisKeys &gal_keys, UnifiedCiphertext &dst)
{
    const bool host_path = src.on_host() && gal_keys.on_host();
    const bool device_path = src.on_device() && gal_keys.on_device();
    if (!host_path && !device_path)
    {
        throw std::invalid_argument("apply_galois_unified requires matching ciphertext/key location.");
    }
    dst = src;
    evaluator.apply_galois_inplace(dst, galois_elt, gal_keys);
}

vector<UnifiedCiphertext> expand_query_sealpir(
    const UnifiedCiphertext &encrypted, uint32_t m, const UnifiedContext &context, const UnifiedEvaluator &evaluator,
    const UnifiedGaloisKeys &gal_keys)
{
    const bool run_on_host = encrypted.on_host();
    const bool run_on_device = encrypted.on_device();
    if (!run_on_host && !run_on_device)
    {
        throw std::invalid_argument("expand_query_sealpir requires ciphertext on host or device.");
    }
    if ((run_on_host && !gal_keys.on_host()) || (run_on_device && !gal_keys.on_device()))
    {
        throw std::invalid_argument("expand_query_sealpir requires matching galois-key location.");
    }
    if (m == 0)
    {
        throw std::invalid_argument("expand_query_sealpir: m must be positive.");
    }

    const uint32_t n = static_cast<uint32_t>(context.hcontext().first_context_data()->parms().poly_modulus_degree());
    const uint32_t logn = ceil_log2_u64(n);
    const uint32_t logm = ceil_log2_u64(m);
    if (logm > logn)
    {
        throw std::logic_error("expand_query_sealpir: m > poly_modulus_degree is not allowed.");
    }

    if (m == 1)
    {
        return vector<UnifiedCiphertext>{ encrypted };
    }

    vector<uint32_t> galois_elts(logn);
    for (uint32_t i = 0; i < logn; ++i)
    {
        galois_elts[i] = (n + (1U << i)) >> i;
    }

    const auto ct_loc = run_on_device ? Datatype::DEVICE : Datatype::HOST;

    UnifiedPlaintext two(Datatype::HOST);
    two.hplain().resize(1);
    two.hplain().set_zero();
    two.hplain()[0] = 2;
    if (run_on_device)
    {
        two.to_device(context);
    }

    vector<UnifiedCiphertext> temp{ encrypted };
    UnifiedCiphertext temp_rotated(ct_loc);
    UnifiedCiphertext temp_shifted(ct_loc);
    UnifiedCiphertext temp_rotated_shifted(ct_loc);

    for (uint32_t i = 0; i + 1 < logm; ++i)
    {
        vector<UnifiedCiphertext> newtemp(temp.size() << 1, UnifiedCiphertext(ct_loc));
        const uint32_t index_raw = (n << 1) - (1U << i);
        const uint32_t index = (index_raw * galois_elts[i]) % (n << 1);
        for (size_t a = 0; a < temp.size(); ++a)
        {
            apply_galois_unified(evaluator, temp[a], galois_elts[i], gal_keys, temp_rotated);
            evaluator.add(temp[a], temp_rotated, newtemp[a]);
            multiply_power_of_x_unified(temp[a], temp_shifted, index_raw, context);
            multiply_power_of_x_unified(temp_rotated, temp_rotated_shifted, index, context);
            evaluator.add(temp_shifted, temp_rotated_shifted, newtemp[a + temp.size()]);
        }
        temp.swap(newtemp);
    }

    vector<UnifiedCiphertext> newtemp(temp.size() << 1, UnifiedCiphertext(ct_loc));
    const uint32_t index_raw = (n << 1) - (1U << (logm - 1));
    const uint32_t index = (index_raw * galois_elts[logm - 1]) % (n << 1);
    for (size_t a = 0; a < temp.size(); ++a)
    {
        if (a >= (m - (1U << (logm - 1))))
        {
            evaluator.multiply_plain(temp[a], two, newtemp[a]);
        }
        else
        {
            apply_galois_unified(evaluator, temp[a], galois_elts[logm - 1], gal_keys, temp_rotated);
            evaluator.add(temp[a], temp_rotated, newtemp[a]);
            multiply_power_of_x_unified(temp[a], temp_shifted, index_raw, context);
            multiply_power_of_x_unified(temp_rotated, temp_rotated_shifted, index, context);
            evaluator.add(temp_shifted, temp_rotated_shifted, newtemp[a + temp.size()]);
        }
    }

    vector<UnifiedCiphertext> expanded;
    expanded.reserve(m);
    for (uint32_t i = 0; i < m; ++i)
    {
        expanded.emplace_back(std::move(newtemp[i]));
    }
    return expanded;
}

vector<uint32_t> required_galois_elts_for_pir(uint32_t poly_degree, uint32_t m, uint32_t chunks)
{
    vector<uint32_t> elts;
    const uint32_t logm = ceil_log2_u64(m);
    const uint32_t two_n = poly_degree << 1;

    // SealPIR oblivious-expansion automorphisms.
    for (uint32_t i = 0; i < logm; ++i)
    {
        elts.push_back((poly_degree + (1U << i)) >> i);
    }

    // Row rotations used by chunk stitching (steps 1..chunks-1), represented by powers of 3 mod 2N.
    for (uint32_t step = 1; step < chunks; ++step)
    {
        elts.push_back(mod_pow_u32(3U, step, two_n));
    }

    std::sort(elts.begin(), elts.end());
    elts.erase(std::unique(elts.begin(), elts.end()), elts.end());
    return elts;
}

void print_overhead_estimates(
    size_t partitions, size_t chunks, size_t poly_degree, size_t num_primes)
{
    constexpr long double kPcieBandwidthBytesPerSec = 24.0L * 1000.0L * 1000.0L * 1000.0L; // 24 GB/s

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
    cout << "I/O Bandwidth Overhead (PCIe Gen4 x16 @ 24 GB/s): "
         << static_cast<double>(transfer_ms) << " ms (" << static_cast<double>(transfer_seconds) << " s)" << endl;
}
} // namespace

int main()
{
    constexpr size_t N = 32768;
    constexpr size_t poly_modulus_degree = 32768;

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 54, 54, 54, 54 }));
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 17)); // 65537

#ifdef USE_HE_GPU
    const Datatype::LOCATION backend = Datatype::DEVICE;
#else
    const Datatype::LOCATION backend = Datatype::HOST;
#endif

    UnifiedContext context(parms, backend);
    print_line(__LINE__);
    print_parameters(context.hcontext());

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    UnifiedRelinKeys relin_keys(Datatype::HOST);
    keygen.create_relin_keys(relin_keys);

    cout << "INFO: Input the number of items (e.g., 2^15 (32768)):" << endl;
    size_t num = 0;
    cin >> num;
    cout << "INFO: Input the bit width (e.g., 64):" << endl;
    size_t l = 0;
    cin >> l;
    cout << "INFO: Input the query id (e.g., 1, in [0,num-1]):" << endl;
    size_t id = 0;
    cin >> id;

    if (num == 0 || (num % N) != 0)
    {
        throw std::invalid_argument("num must be positive and divisible by N=32768.");
    }
    if (l == 0 || (l % 16) != 0)
    {
        throw std::invalid_argument("bit width l must be a positive multiple of 16.");
    }
    if (id >= num)
    {
        throw std::invalid_argument("query id must be in [0, num-1].");
    }

    const size_t dim1 = num / N;
    const size_t h = static_cast<size_t>(std::llround(std::sqrt(static_cast<long double>(dim1))));
    if (h * h != dim1)
    {
        throw std::invalid_argument("SmartPIR setup requires num/N to be a perfect square.");
    }
    const size_t w = h * N;
    const size_t blocks_per_row = w / N;
    const size_t chunks = l / 16;
    if (blocks_per_row == 0 || blocks_per_row > N)
    {
        throw std::invalid_argument("blocks_per_row must be in [1, N].");
    }

    // Corrected SmartPIR index mapping.
    const size_t r_idx = id / w;
    const size_t c_idx = id % w;
    const size_t c_block = c_idx / N;
    const size_t oft = id % N;
    if (r_idx >= h || c_block >= blocks_per_row)
    {
        throw std::invalid_argument("Computed SmartPIR indices are out of range.");
    }

    UnifiedGaloisKeys gal_keys(Datatype::HOST);
    const vector<uint32_t> galois_elts = required_galois_elts_for_pir(
        static_cast<uint32_t>(poly_modulus_degree), static_cast<uint32_t>(blocks_per_row),
        static_cast<uint32_t>(chunks));
    keygen.create_galois_keys(galois_elts, gal_keys.hgalois());

#ifdef USE_HE_GPU
    gal_keys.to_device(context);
    relin_keys.to_device(context);
#endif

    Encryptor encryptor(context, public_key);
    UnifiedEvaluator evaluator(context);
    UnifiedBatchEncoder encoder(context);
    Decryptor decryptor(context, secret_key);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    const uint64_t plain_mod = context.hcontext().first_context_data()->parms().plain_modulus().value();

    cout << "INFO: SmartPIR dimensions: h=" << h << ", w=" << w << ", w/N=" << blocks_per_row
         << ", chunks=" << chunks << endl;
    cout << "INFO: Query mapping: r_idx=" << r_idx << ", c_idx=" << c_idx << ", c_block=" << c_block
         << ", oft=" << oft << endl;
    cout << "INFO: Generating DB and encoding M[h][w/N][chunks] with UnifiedBatchEncoder." << endl;

    vector<vector<vector<vector<uint64_t>>>> db_values(
        h, vector<vector<vector<uint64_t>>>(blocks_per_row, vector<vector<uint64_t>>(chunks, vector<uint64_t>(N))));
    for (size_t r = 0; r < h; ++r)
    {
        for (size_t c = 0; c < blocks_per_row; ++c)
        {
            for (size_t k = 0; k < chunks; ++k)
            {
                for (size_t s = 0; s < N; ++s)
                {
                    db_values[r][c][k][s] = dist(gen) % plain_mod;
                }
            }
        }
    }

#ifdef USE_HE_GPU
    const bool gpu_compute_phase = true;
#else
    const bool gpu_compute_phase = false;
#endif

    cout << "INFO: Server encodes DB blocks into plaintext tensor M." << endl;
    vector<vector<vector<UnifiedPlaintext>>> M(
        h, vector<vector<UnifiedPlaintext>>(blocks_per_row, vector<UnifiedPlaintext>(chunks)));
    const double t_encode_ms = time_phase(
        [&]() {
            for (size_t r = 0; r < h; ++r)
            {
                for (size_t c = 0; c < blocks_per_row; ++c)
                {
                    for (size_t k = 0; k < chunks; ++k)
                    {
                        M[r][c][k] = UnifiedPlaintext(Datatype::HOST);
                        encoder.encode(db_values[r][c][k], M[r][c][k]);
                    }
                }
            }
        },
        false, "T_Encode");

    // QueryGen (X: coefficient-encoded compressed query).
    Plaintext pquery_x_comp(poly_modulus_degree);
    pquery_x_comp.set_zero();
    const uint64_t blocks_mod = blocks_per_row % plain_mod;
    if (blocks_mod == 0)
    {
        throw std::invalid_argument("blocks_per_row has no inverse modulo plain_modulus.");
    }
    const uint64_t inv_scale = mod_inverse_u64(blocks_mod, plain_mod);
    pquery_x_comp[c_block] = inv_scale;

    // QueryGen (Y: slot/batch-encoded uncompressed row selectors).
    vector<uint64_t> vec_ones(N, 1ULL);
    vector<uint64_t> vec_zeros(N, 0ULL);
    UnifiedPlaintext pquery_y_one(Datatype::HOST);
    UnifiedPlaintext pquery_y_zero(Datatype::HOST);
    encoder.encode(vec_ones, pquery_y_one);
    encoder.encode(vec_zeros, pquery_y_zero);

    UnifiedCiphertext compressed_x(Datatype::HOST);
    vector<UnifiedCiphertext> query_y(h, UnifiedCiphertext(Datatype::HOST));
    const double t_encrypt_ms = time_phase(
        [&]() {
            encryptor.encrypt(pquery_x_comp, compressed_x);
            for (size_t j = 0; j < h; ++j)
            {
                if (j == r_idx)
                {
                    encryptor.encrypt(pquery_y_one, query_y[j]);
                }
                else
                {
                    encryptor.encrypt(pquery_y_zero, query_y[j]);
                }
            }
        },
        false, "T_Encrypt");

#ifdef USE_HE_GPU
    compressed_x.to_device(context);
#endif

    cout << "INFO: Server expands compressed X into " << blocks_per_row << " ciphertext selectors." << endl;
    vector<UnifiedCiphertext> expanded_x =
        expand_query_sealpir(compressed_x, static_cast<uint32_t>(blocks_per_row), context, evaluator, gal_keys);

    vector<UnifiedCiphertext> expanded_x_host = expanded_x;
#ifdef USE_HE_GPU
    for (auto &ct : expanded_x_host)
    {
        if (ct.on_device())
        {
            ct.to_host(context);
        }
    }
#endif
    vector<UnifiedCiphertext> query_y_host = query_y;

#ifdef USE_HE_GPU
    for (size_t r = 0; r < h; ++r)
    {
        for (size_t c = 0; c < blocks_per_row; ++c)
        {
            for (size_t k = 0; k < chunks; ++k)
            {
                M[r][c][k].to_device(context);
            }
        }
    }
    for (size_t i = 0; i < blocks_per_row; ++i)
    {
        if (expanded_x[i].on_host())
        {
            expanded_x[i].to_device(context);
        }
    }
    for (size_t j = 0; j < h; ++j)
    {
        query_y[j].to_device(context);
    }
#endif

    cout << "INFO: Server runs SmartPIR answer generation." << endl;
    vector<vector<UnifiedCiphertext>> t_layer1(h, vector<UnifiedCiphertext>(chunks));
    vector<UnifiedCiphertext> t_layer2(chunks);
    const double t_compute_muladd_ms = time_phase(
        [&]() {
            // Step 3.2: first multiplicative layer (ct-pt).
            for (size_t k = 0; k < chunks; ++k)
            {
                for (size_t j = 0; j < h; ++j)
                {
                    bool inited = false;
                    UnifiedCiphertext accum;
                    for (size_t i = 0; i < blocks_per_row; ++i)
                    {
                        UnifiedCiphertext prod = expanded_x[i];
                        evaluator.multiply_plain_inplace(prod, M[j][i][k]);
                        if (!inited)
                        {
                            accum = std::move(prod);
                            inited = true;
                        }
                        else
                        {
                            evaluator.add_inplace(accum, prod);
                        }
                    }
                    t_layer1[j][k] = std::move(accum);
                }
            }

            // Step 3.3: second multiplicative layer (ct-ct + immediate relinearize).
            for (size_t k = 0; k < chunks; ++k)
            {
                bool inited = false;
                UnifiedCiphertext accum;
                for (size_t j = 0; j < h; ++j)
                {
                    UnifiedCiphertext term = t_layer1[j][k];
                    evaluator.multiply_inplace(term, query_y[j]);
                    evaluator.relinearize_inplace(term, relin_keys);
                    if (!inited)
                    {
                        accum = std::move(term);
                        inited = true;
                    }
                    else
                    {
                        evaluator.add_inplace(accum, term);
                    }
                }
                t_layer2[k] = std::move(accum);
            }
        },
        gpu_compute_phase, "T_Compute_MulAdd");

    vector<UnifiedCiphertext> t_layer2_pre_rot = t_layer2;
    const double t_compute_rot_ms = time_phase(
        [&]() {
            for (size_t k = 1; k < chunks; ++k)
            {
                evaluator.rotate_rows_inplace(t_layer2[k], static_cast<int>(k), gal_keys);
            }
        },
        gpu_compute_phase, "T_Compute_Rot");

    UnifiedCiphertext ans_ct = t_layer2[0];
    for (size_t k = 1; k < chunks; ++k)
    {
        evaluator.add_inplace(ans_ct, t_layer2[k]);
    }
    if (gpu_compute_phase)
    {
        gpu_sync_or_throw("FinalAnswerAdd");
    }

#ifdef USE_HE_GPU
    ans_ct.to_host(context);
#endif

    Plaintext ans_pt;
    const double t_decrypt_ms = time_phase(
        [&]() {
            decryptor.decrypt(ans_ct, ans_pt);
        },
        false, "T_Decrypt");

    vector<uint64_t> ans_plain;
    const double t_decode_ms = time_phase(
        [&]() {
            encoder.decode(ans_pt, ans_plain);
        },
        false, "T_Decode");

    cout << "\n===== Phase Latency (ms) =====" << endl;
    cout << std::fixed << std::setprecision(3);
    cout << "T_Encode:         " << t_encode_ms << endl;
    cout << "T_Encrypt:        " << t_encrypt_ms << endl;
    cout << "T_Compute_MulAdd: " << t_compute_muladd_ms << endl;
    cout << "T_Compute_Rot:    " << t_compute_rot_ms << endl;
    cout << "T_Decrypt:        " << t_decrypt_ms << endl;
    cout << "T_Decode:         " << t_decode_ms << endl;

    cout << "\n===== Retrieval Spot Check (SmartPIR 2D) =====" << endl;
    cout << "query id: " << id << " (r_idx=" << r_idx << ", c_block=" << c_block << ", oft=" << oft << ")" << endl;
    cout << "plain_modulus: " << plain_mod << endl;
    const size_t max_print_chunks = std::min<size_t>(chunks, 8);
    for (size_t k = 0; k < max_print_chunks; ++k)
    {
        // Selected-path reference for 2D logic.
        UnifiedPlaintext ref_db(Datatype::HOST);
        encoder.encode(db_values[r_idx][c_block][k], ref_db);
        UnifiedCiphertext ref_ct = expanded_x_host[c_block];
        evaluator.multiply_plain_inplace(ref_ct, ref_db);
        evaluator.multiply_inplace(ref_ct, query_y_host[r_idx]);
        evaluator.relinearize_inplace(ref_ct, relin_keys);

        UnifiedCiphertext check_ct = t_layer2_pre_rot[k];
#ifdef USE_HE_GPU
        if (ref_ct.on_device())
        {
            ref_ct.to_host(context);
        }
        if (check_ct.on_device())
        {
            check_ct.to_host(context);
        }
#endif
        Plaintext ref_pt;
        Plaintext check_pt;
        decryptor.decrypt(ref_ct, ref_pt);
        decryptor.decrypt(check_ct, check_pt);
        vector<uint64_t> ref_plain;
        vector<uint64_t> check_plain;
        encoder.decode(ref_pt, ref_plain);
        encoder.decode(check_pt, check_plain);

        bool equal = (ref_plain.size() == check_plain.size());
        if (equal)
        {
            for (size_t s = 0; s < ref_plain.size(); ++s)
            {
                if (ref_plain[s] != check_plain[s])
                {
                    equal = false;
                    break;
                }
            }
        }

        cout << "chunk[" << k << "] slot[" << oft << "] db=" << db_values[r_idx][c_block][k][oft]
             << ", ref=" << ref_plain[oft] << ", eval=" << check_plain[oft];
        if (equal)
        {
            cout << " (OK)";
        }
        else
        {
            cout << " (MISMATCH)";
        }
        cout << endl;
    }

    const size_t num_primes = context.hcontext().first_context_data()->parms().coeff_modulus().size();
    print_overhead_estimates(h * blocks_per_row, chunks, poly_modulus_degree, num_primes);

    return 0;
}
