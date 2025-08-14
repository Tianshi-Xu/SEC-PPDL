#include <future>
#include <queue>
#include <seal/seal.h>
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

#define BFV_BATCH_NUM 2UL

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

// Global variable to control the number of threads
#ifdef USE_HE_GPU
size_t thread_num = 4;
#else
size_t thread_num = std::thread::hardware_concurrency();
#endif

auto backend = Datatype::DEVICE;
// auto backend = Datatype::HOST;

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
    std::mt19937 gen(0);
    std::uniform_int_distribution<uint64_t> dis(0, (1ULL << plainWidth) - 1);
    for (auto &v : vec)
    {
        v = dis(gen);
    }
}

void fill_random_weight(std::vector<uint64_t> &vec, size_t copy_count, uint64_t plainWidth)
{
    std::random_device rd;
    std::mt19937 gen(0);
    std::uniform_int_distribution<uint64_t> dis(0, (1ULL << plainWidth) - 1);

    auto len = vec.size() / copy_count;
    for (size_t i = 0; i < copy_count; i++)
    {
        auto random_val = dis(gen);
        for (size_t j = 0; j < len; j++)
        {
            vec[i * len + j] = random_val;
        }
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
    if (backend == HOST)
    {
        thread_num = 128;
    }

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
    UnifiedGaloisKeys *galoisKeys = new UnifiedGaloisKeys(HOST);

    KeyGenerator keygen(context);
    *secretKeys = keygen.secret_key();
    keygen.create_public_key(*publicKeys);
    keygen.create_galois_keys(*galoisKeys);
    if (backend == Datatype::DEVICE)
    {
        galoisKeys->to_device(context);
    }

    Encryptor encryptor(context, *publicKeys);
    Decryptor decryptor(context, *secretKeys);

    // Create thread pool for parallel processing with configurable thread number
    ThreadPool pool(thread_num);
    vector<future<void>> futures;

    size_t slot_count = encoder.slot_count();
    size_t row_size = slot_count / BFV_BATCH_NUM;

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

    size_t num_activation_ctxt = activation_rows * activation_cols / row_size;
    size_t num_col_per_act_ctxt = row_size / activation_rows;
    vector<vector<uint64_t>> packed_activation(num_activation_ctxt, vector<uint64_t>(slot_count, 0ULL));
    cout << num_activation_ctxt << " * [" << BFV_BATCH_NUM << ", " << activation_rows << ", " << num_col_per_act_ctxt
         << "]" << endl;
    vector<UnifiedCiphertext> encrypted_activation(num_activation_ctxt, HOST);
    // Column-wise packing
    for (size_t i = 0; i < num_activation_ctxt; i++)
    {
        vector<uint64_t> packed_activation(slot_count, 0ULL);
        for (size_t j = 0; j < num_col_per_act_ctxt; j++)
        {
            for (size_t k = 0; k < activation_rows; k++)
            {
                for (size_t bfv_batch_idx = 0; bfv_batch_idx < BFV_BATCH_NUM; bfv_batch_idx++)
                {
                    packed_activation[bfv_batch_idx * row_size + j * activation_rows + k] =
                        activation_matrix[k * activation_cols + i * num_col_per_act_ctxt + j];
                }
            }
        }
        UnifiedPlaintext plain_activation(HOST);
        encoder.encode(packed_activation, plain_activation);
        encryptor.encrypt(plain_activation, encrypted_activation[i]);
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
    size_t num_tiled_weight_cols = weight_cols / tile_size / BFV_BATCH_NUM;

    // Diagonal packing with tile_size-segmented 2 * (row_size / tile_size) copies
    size_t copy_count = row_size / tile_size;
    cout << num_tiled_weight_rows << " * " << num_tiled_weight_cols << " * " << tile_size << " * [" << BFV_BATCH_NUM
         << ", " << copy_count << ", " << tile_size << "]" << "("
         << num_tiled_weight_cols * num_tiled_weight_rows * tile_size * slot_count * sizeof(uint64_t) /
                static_cast<double>(1024 * 1024 * 1024)
         << "GB)" << endl;

// #define ONLINE_ENCODING
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
                size_t base_col_idx = j * tile_size * BFV_BATCH_NUM;

                for (size_t di = 0; di < tile_size; di++) // (i, j, di)-th diagonal
                {
                    vector<uint64_t> packed_weight(slot_count, 0ULL);
                    for (size_t batch_idx = 0; batch_idx < BFV_BATCH_NUM; batch_idx++)
                    {
                        size_t base_batch_col_idx = base_col_idx + batch_idx * tile_size;
                        for (size_t dj = 0; dj < tile_size; dj++) // (i, j, di, dj)-th element
                        {
                            for (size_t copy_idx = 0; copy_idx < copy_count; copy_idx++)
                            {
                                size_t col_idx = base_batch_col_idx + di + dj;
                                size_t row_idx = base_row_idx + dj;
                                packed_weight[batch_idx * row_size + dj * copy_count + copy_idx] =
                                    weight_matrix[row_idx * weight_cols + col_idx];
                            }
                        }
                    }
                    encoder.encode(packed_weight, encoded_weight[i * num_tiled_weight_cols + j][di]);
#ifdef USE_HE_GPU
                    if (backend == Datatype::DEVICE)
                    {
                        evaluator.transform_to_ntt_inplace(
                            encoded_weight[i * num_tiled_weight_cols + j][di],
                            encrypted_activation.front().dcipher().chain_index());
                    }
                    else
                    {
                        evaluator.transform_to_ntt_inplace(
                            encoded_weight[i * num_tiled_weight_cols + j][di],
                            encrypted_activation.front().hcipher().parms_id());
                    }
#else
                    evaluator.transform_to_ntt_inplace(
                        encoded_weight[i * num_tiled_weight_cols + j][di],
                        encrypted_activation.front().hcipher().parms_id());
#endif
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

#ifndef ONLINE_ENCODING
    vector<uint64_t> rand_raw(slot_count, 1ULL);
    fill_random_weight(rand_raw, tile_size * BFV_BATCH_NUM, 4);
    print_matrix(rand_raw, row_size);
    UnifiedPlaintext random_pt(backend);
    encoder.encode(rand_raw, random_pt);
#ifdef USE_HE_GPU
    if (backend == Datatype::DEVICE)
    {
        evaluator.transform_to_ntt_inplace(random_pt, encrypted_activation.front().dcipher().chain_index());
    }
    else
    {
        evaluator.transform_to_ntt_inplace(random_pt, encrypted_activation.front().hcipher().parms_id());
    }
#else
    evaluator.transform_to_ntt_inplace(random_pt, encrypted_activation.front().hcipher().parms_id());
#endif
#endif

    // == Multiply activation matrix and weight matrix
    // == BSGS block matrix multiplication
    vector<UnifiedCiphertext> result_ctxts(num_tiled_weight_cols, backend);

    start_time = chrono::high_resolution_clock::now();
#if 1
    evaluator.sync();
    for (size_t group_idx = 0; group_idx < num_tiled_weight_rows; group_idx++)
    // for (size_t group_idx = 0; group_idx < 4; group_idx++)
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
            evaluator.rotate_rows(baby_input_ctxt, -i * activation_rows, *galoisKeys, baby_ctxts[i]);
        }
        nvtxPop("BS-Rot", backend);

        for (size_t tiled_col_idx = 0; tiled_col_idx < num_tiled_weight_cols; tiled_col_idx++)
        {
            auto &result_ctxt = result_ctxts[tiled_col_idx];

#ifdef ONLINE_ENCODING
            auto &weight_ptxt = encoded_weight[group_idx * num_tiled_weight_cols + tiled_col_idx];
#endif

            for (size_t i = 0; i < giant_step; i++)
            {
                // 2. Baby-step Plaintext Weight Multiplication and Accumulation

                UnifiedCiphertext giant_ctxt(backend);

                nvtxPush("MAC", backend);
#ifdef ONLINE_ENCODING
                evaluator.multiply_plain_ntt(baby_ctxts[0], weight_ptxt[0], giant_ctxt);
#else
                evaluator.multiply_plain_ntt(baby_ctxts[0], random_pt, giant_ctxt);
#endif

                for (size_t baby_idx = 1; baby_idx < baby_step; baby_idx++)
                {
                    UnifiedCiphertext temp_ctxt(backend);
#ifdef ONLINE_ENCODING
                    evaluator.multiply_plain_ntt(
                        baby_ctxts[baby_idx], weight_ptxt[i * baby_step + baby_idx], temp_ctxt);
#else
                    evaluator.multiply_plain_ntt(baby_ctxts[baby_idx], random_pt, temp_ctxt);
#endif
                    evaluator.add_inplace(giant_ctxt, temp_ctxt);
                }
                nvtxPop("MAC", backend);

                if (group_idx == 0 && i == 0)
                {
                    result_ctxt = giant_ctxt;
                }
                else
                {
                    nvtxPush("GS-Rot", backend);
                    evaluator.rotate_rows_inplace(giant_ctxt, i * giant_step, *galoisKeys);
                    nvtxPop("GS-Rot", backend);
                    nvtxPush("GS-Add", backend);
                    evaluator.add_inplace(result_ctxt, giant_ctxt);
                    nvtxPop("GS-Add", backend);
                }
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
            // TODO:
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
    cout << "    + Noise budget after PCMM: " << decryptor.invariant_noise_budget(result_ctxts[0]) << " bits" << endl;
    UnifiedPlaintext plain_result(HOST);
    decryptor.decrypt(result_ctxts[0], plain_result);
    vector<uint64_t> pod_result(slot_count, 0ULL);
    encoder.decode(plain_result, pod_result);
    print_matrix(pod_result, row_size);

    return 0;
}
