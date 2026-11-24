#include <QuantExperiment/Context.h>
#include <LinearLayer/ConvOptiEncode.h>
#include <exception>
#include <mutex>
#include <thread>
#include <unistd.h>

namespace QuantExperiment {
using Datatype::Tensor;
void Config::Validate() const {
    if (activation_bits < 2 || activation_bits > 16 || activation_bits + 2 > activation_ring_bits)
        throw std::invalid_argument("Activation quantizer must fit the comparison ring");
    if (approximate_relu && !power_of_two)
        throw std::invalid_argument("Approximate ReLU requires the power-of-two work ring");
    if ((party != 1 && party != 2) || threads < 1 || threads > 128 || degree != 8192 ||
        activation_ring_bits < 10 || activation_ring_bits > 60 || accumulator_bits < 20 ||
        accumulator_bits > 60 || plaintext_bits < 20 || plaintext_bits > 60 || port < 1024 ||
        port + threads > 65535)
        throw std::invalid_argument("Invalid isolated quantization experiment configuration");
    if (power_of_two &&
        (legacy_conversion || !ciphertext_inverse || activation_ring_bits != accumulator_bits ||
         accumulator_bits > 32 || linear == LinearBackend::Cheetah))
        throw std::invalid_argument(
            "PO2 mode requires a common 20..32-bit ring and ciphertext normalization");
}

Context::Context(Config cfg) : config(std::move(cfg)) {
    config.Validate();
    std::string pattern =
        (std::filesystem::temp_directory_path() / "sec-ppdl-quant-XXXXXX").string();
    char *directory = mkdtemp(pattern.data());
    if (!directory)
        throw std::runtime_error("Cannot create a private OT session directory");
    session_directory_ = directory;
    channels_.reserve(config.threads);
    packs_.resize(config.threads);
    for (int i = 0; i < config.threads; ++i)
        channels_.emplace_back(std::make_unique<Utils::NetIO>(
            config.party == 1 ? nullptr : config.address.c_str(), config.port + i));
    Parallel([&](size_t i) {
        packs_[i] = std::make_unique<PreparedOTPack>(channels_[i].get(), config.party, config.ot,
                                                     session_directory_ / std::to_string(i));
    });
    for (auto &pack : packs_)
        pack_views_.push_back(pack.get());
    for (int i = 0; i < config.threads; ++i) {
        truncation_.emplace_back(
            std::make_unique<OTProtocol::TruncationProtocol>(config.party, pack_views_[i]));
        auxiliary_.emplace_back(std::make_unique<OTProtocol::AuxProtocols>(
            config.party, channels_[i].get(), pack_views_[i]));
        truncation_views_.push_back(truncation_.back().get());
        auxiliary_views_.push_back(auxiliary_.back().get());
    }
    fix = std::make_unique<NonlinearOperator::FixPoint<uint64_t>>(
        truncation_views_.data(), auxiliary_views_.data(), config.threads);
    for (int i = 0; i < config.threads; ++i)
        lane_fix_.emplace_back(std::make_unique<NonlinearOperator::FixPoint<uint64_t>>(
            &truncation_views_[i], &auxiliary_views_[i], 1));
    he = std::make_unique<HE::HEEvaluator>(
        channels_[0].get(), config.party, config.degree, config.plaintext_bits,
        config.linear == LinearBackend::GPU ? Datatype::DEVICE : Datatype::HOST,
        config.ciphertext_prime_bits,
        config.power_of_two ? uint64_t(1) << config.accumulator_bits : 0);
    he->GenerateNewKey(config.linear == LinearBackend::Cheetah
                           ? std::vector<uint32_t>{}
                           : LinearLayer::Conv2DOptiEncode::GaloisElements(config.degree));
}
Context::~Context() {
    if (he) {
        he->FreeKey();
        delete he->param;
        he->param = nullptr;
    }
    lane_fix_.clear();
    fix.reset();
    auxiliary_.clear();
    truncation_.clear();
    packs_.clear();
    std::filesystem::remove_all(session_directory_);
}
void Context::Parallel(const std::function<void(size_t)> &function) {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::exception_ptr error;
    for (int i = 0; i < config.threads; ++i)
        threads.emplace_back([&, i] {
            try {
                function(i);
            } catch (...) {
                std::lock_guard<std::mutex> guard(mutex);
                if (!error)
                    error = std::current_exception();
            }
        });
    for (auto &thread : threads)
        thread.join();
    if (error)
        std::rethrow_exception(error);
}
uint64_t Context::Bytes() const {
    uint64_t total = 0;
    for (const auto &io : channels_)
        total += io->counter;
    return total;
}
void Context::Sync() {
    channels_[0]->sync();
}
void Context::BeginPlanning() {
    for (auto &pack : packs_)
        pack->BeginPlanning();
}
void Context::EndPlanning() {
    for (auto &pack : packs_)
        pack->EndPlanning();
}
void Context::Preprocess() {
    if (config.preprocess)
        Parallel([&](size_t i) { packs_[i]->Preprocess(); });
    else
        for (auto &pack : packs_)
            pack->UseLive();
}
void Context::CheckConsumed() const {
    for (const auto &pack : packs_)
        pack->CheckConsumed();
}
uint64_t Context::PlannedRows() const {
    uint64_t count = 0;
    for (const auto &pack : packs_)
        count += pack->PlannedRows();
    return count;
}
void Context::ConvertChunks(Tensor<uint64_t> &input,
                            const std::function<void(Tensor<uint64_t> &, size_t)> &function,
                            bool parallel) {
    if (!parallel || config.threads == 1) {
        function(input, 0);
        return;
    }
    Parallel([&](size_t lane) {
        const size_t begin = input.size() * lane / config.threads,
                     end = input.size() * (lane + 1) / config.threads;
        if (begin == end)
            return;
        Tensor<uint64_t> chunk({end - begin});
        std::copy(input.data().begin() + begin, input.data().begin() + end, chunk.data().begin());
        function(chunk, lane);
        std::copy(chunk.data().begin(), chunk.data().end(), input.data().begin() + begin);
    });
}
void Context::RingToField(Tensor<uint64_t> &input) {
    if (config.power_of_two) {
        for (auto &value : input.data())
            value &= he->plain_mod - 1;
        return;
    }
    ConvertChunks(
        input,
        [&](Tensor<uint64_t> &chunk, size_t lane) {
            if (config.legacy_conversion)
                lane_fix_[lane]->SignedRing2Field(chunk, he->plain_mod,
                                                  config.activation_ring_bits);
            else
                lane_fix_[lane]->BoundedRing2Field(chunk, he->plain_mod,
                                                   config.activation_ring_bits);
        },
        config.parallel_conversions);
}
void Context::FieldToRing(Tensor<uint64_t> &input) {
    if (config.power_of_two) {
        for (auto &value : input.data())
            value &= he->plain_mod - 1;
        return;
    }
    if (config.legacy_conversion) {
        if (config.party == 1)
            for (auto &v : input.data())
                v = (v + he->plain_mod / 2) % he->plain_mod;
        fix->Field2Ring(input, he->plain_mod, config.accumulator_bits);
        if (config.party == 1)
            for (auto &v : input.data())
                v -= he->plain_mod / 2;
    } else
        ConvertChunks(
            input,
            [&](Tensor<uint64_t> &chunk, size_t lane) {
                lane_fix_[lane]->BoundedField2Ring(chunk, he->plain_mod, config.accumulator_bits,
                                                   0);
            },
            config.parallel_conversions);
}
void Context::ExtendUnsigned(Tensor<uint64_t> &input, int from, int to) {
    if (from == to)
        return;
    // The core extend loop drops non-divisible tails. Per-lane single-worker
    // calls cover each chunk completely without modifying the core framework.
    ConvertChunks(
        input,
        [&](Tensor<uint64_t> &chunk, size_t lane) {
            lane_fix_[lane]->extend(chunk, from, to, false);
        },
        true);
}
void Context::ExtendSigned(Tensor<uint64_t> &input, int from, int to) {
    if (from == to)
        return;
    ConvertChunks(
        input,
        [&](Tensor<uint64_t> &chunk, size_t lane) {
            Tensor<uint64_t> widened(chunk.shape());
            auxiliary_[lane]->s_extend(chunk.size(), chunk.data().data(), widened.data().data(),
                                       from, to, nullptr);
            chunk = std::move(widened);
        },
        true);
}
void Context::ApproximateShiftNonnegative(Tensor<uint64_t> &input, int shift) {
    const int bits = config.accumulator_bits;
    if (shift <= 0 || shift >= bits)
        throw std::invalid_argument("Invalid approximate shift");
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    const uint64_t divisor = uint64_t(1) << shift;
    Parallel([&](size_t lane) {
        const size_t begin = input.size() * lane / config.threads;
        const size_t count = input.size() * (lane + 1) / config.threads - begin;
        if (!count)
            return;
        auto *ot = packs_[lane]->iknp_straight;
        std::vector<uint64_t> wrap(count);
        // MSB(x)=0 implies wrap(a+b)=MSB(a) OR MSB(b).
        // One arithmetic COT shares this OR modulo 2^shift.
        if (config.party == 1) {
            std::vector<uint64_t> delta(count);
            for (size_t i = 0; i < count; ++i)
                delta[i] = 1 - ((input(begin + i) & mask) >> (bits - 1));
            ot->send_cot(wrap.data(), delta.data(), count, shift);
            for (size_t i = 0; i < count; ++i)
                wrap[i] = (1 - delta[i]) - wrap[i];
        } else {
            std::unique_ptr<bool[]> choices(new bool[count]);
            for (size_t i = 0; i < count; ++i)
                choices[i] = (input(begin + i) & mask) >> (bits - 1);
            ot->recv_cot(wrap.data(), choices.get(), count, shift);
        }
        for (size_t i = 0; i < count; ++i) {
            const uint64_t share = input(begin + i) & mask;
            // Floor on Alice and ceil on Bob keep zero and all multiples exact.
            const auto high = (share + (config.party == 2 ? divisor - 1 : 0)) >> shift;
            input(begin + i) = (high - (wrap[i] << (bits - shift))) & mask;
        }
    });
    input.bitwidth = bits;
}
} // namespace QuantExperiment
