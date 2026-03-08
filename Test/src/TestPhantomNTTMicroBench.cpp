#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <phantom/context.cuh>

using std::cerr;
using std::cout;
using std::endl;
using std::size_t;
using std::string;
using std::vector;

using phantom::EncryptionParameters;
using phantom::scheme_type;
using phantom::arith::CoeffModulus;
using phantom::arith::Modulus;
using phantom::arith::PlainModulus;

namespace {

struct Options {
    int device = 0;
    int warmup_iters = 3;
    int measure_iters = 20;
    int poly_degree = 32768;
    int coeff_mod_count = 4;
    int coeff_mod_bits = 54;
    int plain_modulus_bits = 17;
    vector<int> batch_sizes = {1, 64};
};

void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA error at " << msg << ": " << cudaGetErrorString(err) << endl;
        std::exit(1);
    }
}

vector<int> parse_batch_sizes(const string &csv) {
    vector<int> out;
    std::stringstream ss(csv);
    string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) {
            continue;
        }
        const int v = std::stoi(tok);
        if (v > 0) {
            out.push_back(v);
        }
    }
    if (out.empty()) {
        out.push_back(1);
    }
    return out;
}

Options parse_options(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const string arg(argv[i]);
        if (arg.rfind("--device=", 0) == 0) {
            opt.device = std::stoi(arg.substr(9));
        } else if (arg.rfind("--warmup=", 0) == 0) {
            opt.warmup_iters = std::max(0, std::stoi(arg.substr(9)));
        } else if (arg.rfind("--iters=", 0) == 0) {
            opt.measure_iters = std::max(1, std::stoi(arg.substr(8)));
        } else if (arg.rfind("--poly-degree=", 0) == 0) {
            opt.poly_degree = std::stoi(arg.substr(14));
        } else if (arg.rfind("--coeff-count=", 0) == 0) {
            opt.coeff_mod_count = std::stoi(arg.substr(14));
        } else if (arg.rfind("--coeff-bits=", 0) == 0) {
            opt.coeff_mod_bits = std::stoi(arg.substr(13));
        } else if (arg.rfind("--plain-bits=", 0) == 0) {
            opt.plain_modulus_bits = std::stoi(arg.substr(13));
        } else if (arg.rfind("--batches=", 0) == 0) {
            opt.batch_sizes = parse_batch_sizes(arg.substr(10));
        } else if (arg == "--help" || arg == "-h") {
            cout << "Usage: test_ntt_microbench_gpu [--device=N] [--warmup=W] [--iters=I] [--poly-degree=P] "
                    "[--coeff-count=C] [--coeff-bits=B] [--plain-bits=T] [--batches=1,64,...]"
                 << endl;
            std::exit(0);
        } else {
            cerr << "Unknown argument: " << arg << endl;
            std::exit(1);
        }
    }
    return opt;
}

void ntt_forward_batch(uint64_t *data, const DNTTTable &tables, size_t q_count, size_t batch_num, cudaStream_t stream) {
#ifdef RNS_POLY_BATCH
    nwt_2d_radix8_forward_inplace(data, tables, q_count, 0, batch_num, stream);
#else
    const size_t stride = tables.n() * q_count;
    for (size_t b = 0; b < batch_num; ++b) {
        nwt_2d_radix8_forward_inplace(data + b * stride, tables, q_count, 0, stream);
    }
#endif
}

void ntt_backward_batch(uint64_t *data, const DNTTTable &tables, size_t q_count, size_t batch_num, cudaStream_t stream) {
#ifdef RNS_POLY_BATCH
    nwt_2d_radix8_backward_inplace(data, tables, q_count, 0, batch_num, stream);
#else
    const size_t stride = tables.n() * q_count;
    for (size_t b = 0; b < batch_num; ++b) {
        nwt_2d_radix8_backward_inplace(data + b * stride, tables, q_count, 0, stream);
    }
#endif
}

float benchmark_forward(uint64_t *data, const DNTTTable &tables, size_t q_count, size_t batch_num, int warmup_iters,
                        int iters, cudaStream_t stream) {
    for (int i = 0; i < warmup_iters; ++i) {
        ntt_forward_batch(data, tables, q_count, batch_num, stream);
    }
    check_cuda(cudaStreamSynchronize(stream), "forward warmup sync");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create event start");
    check_cuda(cudaEventCreate(&stop), "create event stop");

    check_cuda(cudaEventRecord(start, stream), "record start");
    for (int i = 0; i < iters; ++i) {
        ntt_forward_batch(data, tables, q_count, batch_num, stream);
    }
    check_cuda(cudaEventRecord(stop, stream), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    check_cuda(cudaEventDestroy(start), "destroy event start");
    check_cuda(cudaEventDestroy(stop), "destroy event stop");
    return elapsed_ms / static_cast<float>(iters);
}

float benchmark_roundtrip(uint64_t *data, const DNTTTable &tables, size_t q_count, size_t batch_num, int warmup_iters,
                          int iters, cudaStream_t stream) {
    for (int i = 0; i < warmup_iters; ++i) {
        ntt_forward_batch(data, tables, q_count, batch_num, stream);
        ntt_backward_batch(data, tables, q_count, batch_num, stream);
    }
    check_cuda(cudaStreamSynchronize(stream), "roundtrip warmup sync");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create event start");
    check_cuda(cudaEventCreate(&stop), "create event stop");

    check_cuda(cudaEventRecord(start, stream), "record start");
    for (int i = 0; i < iters; ++i) {
        ntt_forward_batch(data, tables, q_count, batch_num, stream);
        ntt_backward_batch(data, tables, q_count, batch_num, stream);
    }
    check_cuda(cudaEventRecord(stop, stream), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    check_cuda(cudaEventDestroy(start), "destroy event start");
    check_cuda(cudaEventDestroy(stop), "destroy event stop");
    return elapsed_ms / static_cast<float>(iters);
}

} // namespace

int main(int argc, char **argv) {
    const Options opt = parse_options(argc, argv);

    check_cuda(cudaSetDevice(opt.device), "set device");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, opt.device), "get device properties");
    cout << "Device: " << prop.name << " (id=" << opt.device << ")" << endl;

    if (opt.poly_degree <= 0 || opt.coeff_mod_count <= 0 || opt.coeff_mod_bits <= 1 || opt.plain_modulus_bits <= 1) {
        cerr << "Invalid argument: poly-degree/coefficient/plain bits must be positive." << endl;
        return 1;
    }

    vector<int> coeff_bits(static_cast<size_t>(opt.coeff_mod_count), opt.coeff_mod_bits);
    vector<Modulus> coeff_modulus = CoeffModulus::Create(static_cast<size_t>(opt.poly_degree), coeff_bits);

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(static_cast<size_t>(opt.poly_degree));
    parms.set_coeff_modulus(coeff_modulus);
    parms.set_plain_modulus(PlainModulus::Batching(static_cast<size_t>(opt.poly_degree), opt.plain_modulus_bits));

    PhantomContext context(parms);
    const size_t q_count = context.key_context_data().parms().coeff_modulus().size();
    const cudaStream_t stream = cudaStreamPerThread;

    cout << "Config: N=" << opt.poly_degree << ", q_count=" << q_count << ", coeff_bits=" << opt.coeff_mod_bits
         << ", plain_bits=" << opt.plain_modulus_bits << ", warmup=" << opt.warmup_iters
         << ", iters=" << opt.measure_iters << endl;
    cout << "Batch sizes:";
    for (const int b : opt.batch_sizes) {
        cout << " " << b;
    }
    cout << endl;

    cout << std::fixed << std::setprecision(3);
    for (const int batch_num_i : opt.batch_sizes) {
        const size_t batch_num = static_cast<size_t>(batch_num_i);
        const size_t elem_count = static_cast<size_t>(opt.poly_degree) * q_count * batch_num;
        const size_t bytes = elem_count * sizeof(uint64_t);

        size_t free_mem = 0;
        size_t total_mem = 0;
        check_cuda(cudaMemGetInfo(&free_mem, &total_mem), "mem info");

        if (bytes > static_cast<size_t>(free_mem * 0.75)) {
            cout << "[SKIP] batch=" << batch_num << " requires " << (bytes / (1024.0 * 1024.0 * 1024.0))
                 << " GiB, free " << (free_mem / (1024.0 * 1024.0 * 1024.0)) << " GiB" << endl;
            continue;
        }

        uint64_t *d_data = nullptr;
        check_cuda(cudaMalloc(&d_data, bytes), "malloc data");

        check_cuda(cudaMemsetAsync(d_data, 0, bytes, stream), "memset data");
        check_cuda(cudaStreamSynchronize(stream), "sync memset");

        const float forward_ms = benchmark_forward(
            d_data, context.gpu_rns_tables(), q_count, batch_num, opt.warmup_iters, opt.measure_iters, stream);

        check_cuda(cudaMemsetAsync(d_data, 0, bytes, stream), "memset data 2");
        check_cuda(cudaStreamSynchronize(stream), "sync memset 2");

        const float roundtrip_ms = benchmark_roundtrip(
            d_data, context.gpu_rns_tables(), q_count, batch_num, opt.warmup_iters, opt.measure_iters, stream);

        const double coeffs = static_cast<double>(elem_count);
        const double coeffs_per_ms = coeffs / std::max(1e-9, static_cast<double>(forward_ms));
        const double coeffs_per_s = coeffs_per_ms * 1000.0;

        cout << "[batch=" << batch_num << "]"
             << " data=" << (bytes / (1024.0 * 1024.0)) << " MiB"
             << ", forward_ntt=" << forward_ms << " ms/op"
             << ", roundtrip(fwd+inv)=" << roundtrip_ms << " ms/op"
             << ", fwd_throughput=" << (coeffs_per_s / 1e6) << " Mcoeff/s" << endl;

        check_cuda(cudaFree(d_data), "free data");
    }

    return 0;
}
