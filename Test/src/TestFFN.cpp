#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iomanip>
#include <mutex>
#include <queue>
#include <random>
#include <seal/seal.h>
#include <thread>
#include <unordered_map>
#include "HE/unified/UnifiedEncoder.h"
#include "HE/unified/UnifiedEvaluator.h"
#include "HE/unified/UnifiedEvk.h"
#include "HE/unified/UnifiedPlaintext.h"
#ifdef USE_HE_GPU
#include <nvtx3/nvToolsExt.h>
#endif

using namespace std;
using namespace seal;
using namespace HE::unified;

std::unordered_map<std::string, long long> nvtxDurations;

void nvtxPush(const std::string &name, LOCATION backend)
{
    if (backend == DEVICE)
    {
#ifdef USE_HE_GPU
        nvtxRangePush(name.c_str());
#endif
    }
    else
    {
        auto now = std::chrono::high_resolution_clock::now();
        nvtxDurations[name] -= std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    }
}

void nvtxPop(const std::string &name, LOCATION backend)
{
    if (backend == DEVICE)
    {
#ifdef USE_HE_GPU
        nvtxRangePop();
#endif
    }
    else
    {
        auto now = std::chrono::high_resolution_clock::now();
        nvtxDurations[name] += std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    }
}

void printNVTXStats()
{
    if (!nvtxDurations.empty())
    {
        std::cout << "\nTime Statistics (ms):\n";
        for (const auto &[name, duration] : nvtxDurations)
        {
            std::cout << name << ": " << duration << "\n";
        }
    }
}

// auto backend = Datatype::DEVICE;
auto backend = Datatype::HOST;

// Thread Pool implementation
class ThreadPool
{
private:
    vector<thread> workers;
    queue<function<void()>> tasks;
    mutex queue_mutex;
    condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t threads) : stop(false)
    {
        for (size_t i = 0; i < threads; ++i)
            workers.emplace_back([this] {
                for (;;)
                {
                    function<void()> task;
                    {
                        unique_lock<mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
    }

    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args) -> future<typename result_of<F(Args...)>::type>
    {
        using return_type = typename result_of<F(Args...)>::type;
        auto task = make_shared<packaged_task<return_type()>>(bind(std::forward<F>(f), forward<Args>(args)...));
        future<return_type> res = task->get_future();
        {
            unique_lock<mutex> lock(queue_mutex);
            if (stop)
                throw runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool()
    {
        {
            unique_lock<mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (thread &worker : workers)
            worker.join();
    }
};

inline void print_line(int line_number)
{
    std::cout << "Line " << std::setw(3) << line_number << " --> ";
}

template <typename T>
inline void print_matrix(std::vector<T> matrix, std::size_t row_size)
{
    /*
    We're not going to print every column of the matrix (there are 2048). Instead
    print this many slots from beginning and end of the matrix.
    */
    std::size_t print_size = 5;

    std::cout << std::endl;
    std::cout << "    [";
    for (std::size_t i = 0; i < print_size; i++)
    {
        std::cout << std::setw(3) << std::right << matrix[i] << ",";
    }
    std::cout << std::setw(3) << " ...,";
    for (std::size_t i = row_size - print_size; i < row_size; i++)
    {
        std::cout << std::setw(3) << matrix[i] << ((i != row_size - 1) ? "," : " ]\n");
    }
    std::cout << "    [";
    for (std::size_t i = row_size; i < row_size + print_size; i++)
    {
        std::cout << std::setw(3) << matrix[i] << ",";
    }
    std::cout << std::setw(3) << " ...,";
    for (std::size_t i = 2 * row_size - print_size; i < 2 * row_size; i++)
    {
        std::cout << std::setw(3) << matrix[i] << ((i != 2 * row_size - 1) ? "," : " ]\n");
    }
    std::cout << std::endl;
}

void fill_random_vector(std::vector<uint64_t> &vec, uint64_t plainWidth)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(0, (1ULL << plainWidth) - 1);
    for (auto &v : vec)
    {
        v = dis(gen);
    }
}

size_t get_baby_step(size_t M)
{
    size_t minval = M, maxk = 0;
    for (size_t k = 1; k <= 3 * std::sqrt(M); k++)
    {
        auto currval = std::ceil((M + 0.0) / (k + 0.0)) + k - 1;
        if (currval <= minval)
        {
            minval = currval;
            maxk = std::max(maxk, k);
        }
    }
    return maxk;
}

int main()
{
    uint64_t polyModulusDegree = 4096;
    uint64_t plainWidth = 17;

    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_poly_modulus_degree(polyModulusDegree);
    parms.set_plain_modulus(seal::PlainModulus::Batching(polyModulusDegree, plainWidth));
    parms.set_coeff_modulus(seal::CoeffModulus::Create(polyModulusDegree, { 44, 44 }));
    UnifiedContext context(parms, backend);
    UnifiedBatchEncoder encoder(context);
    UnifiedEvaluator evaluator(context);

    SecretKey *secretKeys = new SecretKey();
    PublicKey *publicKeys = new PublicKey();
    // RelinKeys *relinKeys = new RelinKeys();
    UnifiedGaloisKeys *galoisKeys = new UnifiedGaloisKeys(HOST);

    KeyGenerator keygen(context);
    *secretKeys = keygen.secret_key();
    keygen.create_public_key(*publicKeys);
    // keygen.create_relin_keys(*relinKeys);
    keygen.create_galois_keys(*galoisKeys);
    if (backend == Datatype::DEVICE)
    {
        galoisKeys->to_device(context);
    }

    Encryptor encryptor(context, *publicKeys);
    Decryptor decryptor(context, *secretKeys);

    // Create thread pool for parallel processing
    size_t num_threads = thread::hardware_concurrency();
    ThreadPool pool(num_threads);
    vector<future<void>> futures;

    size_t slot_count = encoder.slot_count();
    size_t row_size = slot_count / 2;

    size_t seq_len = 128;

    // Generate activation matrix (128 x 4096), 4 bits width
    size_t activation_rows = seq_len; // 128
    size_t activation_cols = 4096;
    vector<uint64_t> activation_matrix(activation_rows * activation_cols, 0ULL);
    fill_random_vector(activation_matrix, 4);

    // Generate weight matrix (4096 x 12288), 4 bits width
    size_t weight_rows = 4096;
    size_t weight_cols = 12288; // 4096 * 3
    vector<uint64_t> weight_matrix(weight_rows * weight_cols, 0ULL);
    fill_random_vector(weight_matrix, 4);

    // Encrypt packed activation matrix
    auto start_time = chrono::high_resolution_clock::now();

    size_t num_activation_ctxt = activation_rows * activation_cols / slot_count;
    size_t num_col_per_act_ctxt = slot_count / activation_rows;
    vector<vector<uint64_t>> packed_activation(num_activation_ctxt, vector<uint64_t>(slot_count, 0ULL));
    cout << num_activation_ctxt << " * [" << activation_rows << ", " << num_col_per_act_ctxt << "]" << endl;
    vector<UnifiedCiphertext> encrypted_activation(num_activation_ctxt, HOST);
    // Column-wise packing
    for (size_t i = 0; i < num_activation_ctxt; i++)
    {
        vector<uint64_t> packed_activation(slot_count, 0ULL);
        for (size_t j = 0; j < num_col_per_act_ctxt; j++)
        {
            for (size_t k = 0; k < activation_rows; k++)
            {
                packed_activation[k] = activation_matrix[i * slot_count + k];
            }
        }
        UnifiedPlaintext plain_activation(HOST);
        encoder.encode(packed_activation, plain_activation);
        encryptor.encrypt(plain_activation, encrypted_activation[i]);
#ifndef USE_HE_GPU
        // 注意：SEAL库rotate_rows函数需要输入的密文在非NTT域上，否则会报错
        // evaluator.transform_to_ntt_inplace(encrypted_activation[i]);
#endif
        if (backend == Datatype::DEVICE)
        {
            encrypted_activation[i].to_device(context);
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Encrypted packed activation matrix - Ready. Time: " << duration.count() << " ms" << endl;

    // Encode packed weight matrix
    start_time = chrono::high_resolution_clock::now();

    size_t tile_size = num_col_per_act_ctxt;
    size_t num_tiled_weight_rows = weight_rows / tile_size;
    size_t num_tiled_weight_cols = weight_cols / tile_size;

    // Diagonal packing with tile_size-segmented (slot_count / tile_size) copies
    size_t copy_count = slot_count / tile_size;
    cout << num_tiled_weight_rows << " * " << num_tiled_weight_cols << " * " << tile_size << " * [" << copy_count
         << ", " << tile_size << "]" << "("
         << num_tiled_weight_cols * num_tiled_weight_rows * tile_size * slot_count * sizeof(uint64_t) /
                static_cast<double>(1024 * 1024 * 1024)
         << "GB)" << endl;

#ifdef ONLINE_ENCODING
    vector<vector<UnifiedPlaintext>> encoded_weight(
        num_tiled_weight_rows * num_tiled_weight_cols, vector<UnifiedPlaintext>(tile_size, backend));

    // Parallelize weight matrix encoding
    futures.clear();
    for (size_t i = 0; i < num_tiled_weight_rows; i++)
    {
        for (size_t j = 0; j < num_tiled_weight_cols; j++) // (i, j)-th tile
        {
            futures.push_back(pool.enqueue([&, i, j]() {
                size_t base_row_idx = i * tile_size;
                size_t base_col_idx = j * tile_size;

                for (size_t di = 0; di < tile_size; di++) // (i, j, di)-th diagonal
                {
                    for (size_t dj = 0; dj < tile_size; dj++) // (i, j, di, dj)-th element
                    {
                        vector<uint64_t> packed_weight(slot_count, 0ULL);
                        for (size_t copy_idx = 0; copy_idx < copy_count; copy_idx++)
                        {
                            size_t col_idx = base_col_idx + di + dj;
                            size_t row_idx = base_row_idx + dj;
                            packed_weight[dj * copy_count + copy_idx] = weight_matrix[row_idx * weight_cols + col_idx];
                            encoder.encode(packed_weight, encoded_weight[i * num_tiled_weight_cols + j][di]);
                        }
                    }
                }
            }));
        }
    }

    // Wait for all weight encoding tasks to complete
    for (auto &future : futures)
    {
        future.wait();
    }
#endif

    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Encoded packed weight matrix - Ready. Time: " << duration.count() << " ms" << endl;

    // Ciphertext activation-Plaintext weight matrix multiplication
    start_time = chrono::high_resolution_clock::now();

    // 1. Generate zeros ciphertext
    UnifiedPlaintext zeros_pt(HOST);
    UnifiedCiphertext zeros_ct(HOST);
    std::vector<uint64_t> zeros(polyModulusDegree, 0);
    encoder.encode(zeros, zeros_pt);
    encryptor.encrypt(zeros_pt, zeros_ct);
    if (backend == Datatype::DEVICE)
    {
        zeros_ct.to_device(context);
    }

    vector<uint64_t> rand_raw(slot_count, 0ULL);
    fill_random_vector(rand_raw, 2);
    UnifiedPlaintext random_pt(backend);
    encoder.encode(rand_raw, random_pt);
#ifndef USE_HE_GPU
    evaluator.transform_to_ntt_inplace(random_pt, encrypted_activation.front().hcipher().parms_id());
#endif

    // 2. Multiply activation matrix and weight matrix
    vector<UnifiedCiphertext> result_ctxts(num_tiled_weight_cols, zeros_ct);
#ifdef USE_HE_GPU
    evaluator.sync();
    for (size_t group_idx = 0; group_idx < num_tiled_weight_rows; group_idx++)
    {
        // BSGS matrix multiplication
        size_t baby_step = get_baby_step(num_col_per_act_ctxt);
        size_t giant_step = num_col_per_act_ctxt / baby_step;

        cout << group_idx << "-th group:" << baby_step - 1 << " baby-step pre-rotations, " << num_tiled_weight_cols
             << " * (" << num_col_per_act_ctxt << " multiplications, " << giant_step - 1
             << " giant-step post-rotations)" << endl;

        // 1. Baby-step Pre-Rotation
        const auto &baby_input_ctxt = encrypted_activation[group_idx];
        vector<UnifiedCiphertext> baby_ctxts(baby_step, backend);

        nvtxPush("BS-Rot", backend);
        for (size_t i = 0; i < baby_step; i++)
        {
            evaluator.rotate_rows(baby_input_ctxt, i, *galoisKeys, baby_ctxts[i]);
        }
        nvtxPop("BS-Rot", backend);

        auto &result_ctxt = result_ctxts[group_idx];
        for (size_t tiled_col_idx = 0; tiled_col_idx < num_tiled_weight_cols; tiled_col_idx++)
        {
#ifdef ONLINE_ENCODING
            auto &weight_ptxt =
                encoded_weight[group_idx * num_tiled_weight_cols * tile_size + tiled_col_idx * tile_size];
#else
            // vector<UnifiedPlaintext> weight_ptxt(baby_step, random_pt);
#endif

            for (size_t i = 0; i < giant_step; i++)
            {
                // 2. Baby-step Plaintext Weight Multiplication and Accumulation

                UnifiedCiphertext giant_ctxt(backend);

                nvtxPush("MAC", backend);
                // evaluator.multiply_plain(baby_input_ctxt, weight_ptxt[0], giant_ctxt);
                // 注意：这里模拟了Plaintext weight multiplication，实际中应该使用Encoded weight
                evaluator.multiply_plain(baby_input_ctxt, random_pt, giant_ctxt);

                for (size_t baby_idx = 1; baby_idx < baby_step; baby_idx++)
                {
                    UnifiedCiphertext temp_ctxt(backend);
                    evaluator.multiply_plain(baby_ctxts[baby_idx], random_pt, temp_ctxt);
                    evaluator.add_inplace(giant_ctxt, temp_ctxt);
                }
                nvtxPop("MAC", backend);

                nvtxPush("GS-Rot", backend);
                evaluator.rotate_rows_inplace(giant_ctxt, i * giant_step, *galoisKeys);
                nvtxPop("GS-Rot", backend);

                nvtxPush("GS-Add", backend);
                evaluator.add_inplace(result_ctxt, giant_ctxt);
                nvtxPop("GS-Add", backend);
            }
        }
    }
    evaluator.sync();
    printNVTXStats();
#else
    // Parallelize matrix multiplication by processing groups in parallel
    futures.clear();
    for (size_t group_idx = 0; group_idx < num_tiled_weight_rows; group_idx++)
    {
        futures.push_back(pool.enqueue([&, group_idx]() {
            // BSGS matrix multiplication
            size_t baby_step = get_baby_step(num_col_per_act_ctxt);
            size_t giant_step = num_col_per_act_ctxt / baby_step;

            cout << group_idx << "-th group:" << baby_step - 1 << " baby-step pre-rotations, " << num_tiled_weight_cols
                 << " * (" << num_col_per_act_ctxt << " multiplications, " << giant_step - 1
                 << " giant-step post-rotations)" << endl;

            // 1. Baby-step Pre-Rotation (Hoisting if supported)
            const auto &baby_input_ctxt = encrypted_activation[group_idx];
            vector<UnifiedCiphertext> baby_ctxts(baby_step, backend);
            for (size_t i = 0; i < baby_step; i++)
            {
                evaluator.rotate_rows(baby_input_ctxt, i, *galoisKeys, baby_ctxts[i]);
            }

            auto &result_ctxt = result_ctxts[group_idx];
            for (size_t tiled_col_idx = 0; tiled_col_idx < num_tiled_weight_cols; tiled_col_idx++)
            {
                auto &weight_ctxt =
                    encoded_weight[group_idx * num_tiled_weight_cols * tile_size + tiled_col_idx * tile_size];

                for (size_t i = 0; i < giant_step; i++)
                {
                    UnifiedCiphertext giant_ctxt(backend);
                    evaluator.multiply_plain(baby_input_ctxt, weight_ctxt[0], giant_ctxt);

                    // 2. Baby-step Plaintext Weight Multiplication and Accumulation
                    for (size_t baby_idx = 1; baby_idx < baby_step; baby_idx++)
                    {
                        UnifiedCiphertext temp_ctxt(backend);
                        evaluator.multiply_plain(baby_ctxts[baby_idx], weight_ctxt[baby_idx], temp_ctxt);
                        evaluator.add_inplace(giant_ctxt, temp_ctxt);
                    }

                    // 3. Giant-step Post-Rotation
                    evaluator.rotate_rows_inplace(giant_ctxt, i * giant_step, *galoisKeys);
                    evaluator.add_inplace(result_ctxt, giant_ctxt);
                }
            }
        }));
    }

    // Wait for all matrix multiplication tasks to complete
    for (auto &future : futures)
    {
        future.wait();
    }
#endif

    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Ciphertext activation-Plaintext weight matrix multiplication - Complete. Time: " << duration.count()
         << " ms" << endl;

    // Decrypt and decode
    if (backend == Datatype::DEVICE)
    {
        result_ctxts[0].to_host(context);
    }
    UnifiedPlaintext plain_result(HOST);
    decryptor.decrypt(result_ctxts[0], plain_result);
    vector<uint64_t> pod_result(slot_count, 0ULL);
    encoder.decode(plain_result, pod_result);
    print_matrix(pod_result, row_size);

    return 0;
}
