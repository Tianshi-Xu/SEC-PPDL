#include <QuantExperiment/PreparedOT.h>
#include <OTPrimitive/cf2_ot_pack.h>
#include <OTPrimitive/cheetah_ot_pack.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace QuantExperiment {
namespace {
using IO = Utils::NetIO;
using BaseOT = OTPrimitive::OT<IO>;
using Block = Utils::block128;
struct Budget {
    uint64_t rows = 0, used = 0, raw = 0;
    int payload_bits = 0;
};
struct PadPool {
    std::vector<uint64_t> words;
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> choice;
    size_t cursor = 0;
    void Resize(size_t count, int bits) {
        // Comparisons consume only a few low bits from each pad in a large 1-out-of-N table.
        bytes.resize(bits <= 8 ? count : 0);
        words.resize(bits <= 8 ? 0 : count);
    }
    void Set(size_t index, uint64_t value) {
        if (bytes.empty())
            words[index] = value;
        else
            bytes[index] = static_cast<uint8_t>(value);
    }
    uint64_t Get(size_t index) const {
        return bytes.empty() ? words[index] : bytes[index];
    }
};
uint64_t Mask(int bits) {
    if (bits < 1 || bits > 64)
        throw std::invalid_argument("OT payload width must be 1..64");
    return bits == 64 ? UINT64_MAX : (uint64_t(1) << bits) - 1;
}
int Log2(int n) {
    if (n < 2 || n > 256 || (n & (n - 1)))
        throw std::invalid_argument("OT arity must be a power of two in 2..256");
    int result = 0;
    for (; n > 1; n >>= 1)
        ++result;
    return result;
}

class CachedFerret : public emp::FerretCOT<IO> {
  public:
    CachedFerret(int role, IO **ios, const std::string &path, OTPhase &phase)
        : emp::FerretCOT<IO>(role, 1, ios, false, true, emp::ferret_b13, path), phase_(phase) {}
    ~CachedFerret() {
        // Keep bootstrap state in memory only; the pinned EMP destructor would
        // otherwise persist it to disk. Discard it before destroying the base.
        std::vector<uint8_t> discarded(static_cast<size_t>(this->state_size()));
        this->assemble_state(discarded.data(), discarded.size());
        std::fill(discarded.begin(), discarded.end(), 0);
    }
    void Prepare(uint64_t count) {
        cache_.resize(count);
        cursor_ = 0;
        if (count)
            this->rcot(cache_.data(), static_cast<int64_t>(count));
    }
    void CheckConsumed() const {
        if (cursor_ != cache_.size())
            throw std::runtime_error("Unused Ferret preprocessing correlations");
    }
    void send_cot(Block *data, int64_t length) override {
        if (phase_ != OTPhase::Prepared) {
            emp::FerretCOT<IO>::send_cot(data, length);
            return;
        }
        Take(data, length);
        std::unique_ptr<bool[]> correction(new bool[length]);
        this->io->recv_bool(correction.get(), length);
        for (int64_t i = 0; i < length; ++i)
            if (correction[i])
                data[i] ^= this->Delta;
    }
    void recv_cot(Block *data, const bool *choices, int64_t length) override {
        if (phase_ != OTPhase::Prepared) {
            emp::FerretCOT<IO>::recv_cot(data, choices, length);
            return;
        }
        Take(data, length);
        std::unique_ptr<bool[]> correction(new bool[length]);
        for (int64_t i = 0; i < length; ++i)
            correction[i] = choices[i] ^ Utils::getLSB(data[i]);
        this->io->send_bool(correction.get(), length);
        this->io->flush();
    }

  private:
    OTPhase &phase_;
    std::vector<Block> cache_;
    size_t cursor_ = 0;
    void Take(Block *out, int64_t length) {
        if (length < 0 || static_cast<uint64_t>(length) > cache_.size() - cursor_)
            throw std::runtime_error("Ferret preprocessing exhausted: online refill is disabled");
        std::copy_n(cache_.data() + cursor_, length, out);
        cursor_ += length;
    }
};

class PreparedSilent : public OTPrimitive::SilentOT<IO> {
  public:
    using OTPrimitive::SilentOT<IO>::send;
    using OTPrimitive::SilentOT<IO>::recv;
    PreparedSilent(int role, IO **ios, const std::string &path, OTPhase &phase)
        : OTPrimitive::SilentOT<IO>(role, 1, ios, false, false, path, false) {
        delete this->ferret;
        this->ferret = nullptr;
        cached_ = new CachedFerret(role, ios, path, phase);
        this->ferret = cached_;
    }
    ~PreparedSilent() override {
        delete cached_;
        this->ferret = nullptr;
    }
    CachedFerret &Cache() {
        return *cached_;
    }
    void send(uint64_t **data, int length, int bits) override {
        Mask(bits);
        std::vector<Block> raw(length);
        this->send_ot_rcm_cc(raw.data(), length);
        Block seed;
        this->ferret->prg.random_block(&seed, 1);
        this->ferret->io->send_block(&seed, 1);
        this->ferret->mitccrh.setS(seed);
        this->ferret->io->flush();
        std::array<Block, 16> pad{};
        std::vector<uint64_t> packed((16 * bits + 63) / 64);
        for (int i = 0; i < length; i += 8) {
            const int count = std::min(8, length - i), words = (2 * count * bits + 63) / 64;
            for (int j = 0; j < count; ++j) {
                pad[2 * j] = raw[i + j];
                pad[2 * j + 1] = raw[i + j] ^ this->ferret->Delta;
            }
            this->ferret->mitccrh.template hash<8, 2>(pad.data());
            OTPrimitive::pack_ot_messages<uint64_t>(packed.data(), data + i, pad.data(), words,
                                                    count, bits, 2);
            this->ferret->io->send_data(packed.data(), words * 8);
        }
    }
    void recv(uint64_t *data, uint8_t *choices, int length, int bits) override {
        Mask(bits);
        std::vector<Block> raw(length);
        this->recv_ot_rcm_cc(raw.data(), reinterpret_cast<const bool *>(choices), length);
        Block seed;
        this->ferret->io->recv_block(&seed, 1);
        this->ferret->mitccrh.setS(seed);
        std::array<Block, 8> pad{};
        std::vector<uint64_t> packed((16 * bits + 63) / 64);
        for (int i = 0; i < length; i += 8) {
            const int count = std::min(8, length - i), words = (2 * count * bits + 63) / 64;
            std::copy_n(raw.data() + i, count, pad.data());
            this->ferret->mitccrh.template hash<8, 1>(pad.data());
            this->ferret->io->recv_data(packed.data(), words * 8);
            OTPrimitive::unpack_ot_messages<uint64_t>(data + i, choices + i, packed.data(),
                                                      pad.data(), count, bits, 2);
        }
    }

  private:
    CachedFerret *cached_ = nullptr;
};

// Expand random-choice binary ROTs into a complete 1-out-of-N random table.
// This follows the tree-key derivation used by the repository's SilentOTN,
// but performs every input-independent hash before online execution.
void GeneratePadBatch(PreparedSilent &source, bool sender, int n, size_t offset, size_t count,
                      PadPool &pool) {
    const int levels = Log2(n);
    const size_t binary = count * levels;
    std::vector<Block> first(binary), second(sender ? binary : 0);
    std::unique_ptr<bool[]> choices(sender ? nullptr : new bool[binary]);
    if (sender)
        source.send_ot_rm_rc(first.data(), second.data(), binary);
    else
        source.recv_ot_rm_rc(first.data(), choices.get(), binary);
    if (n == 2) {
        for (size_t i = 0; i < count; ++i) {
            if (sender) {
                pool.Set(2 * (offset + i), _mm_extract_epi64(first[i], 0));
                pool.Set(2 * (offset + i) + 1, _mm_extract_epi64(second[i], 0));
            } else {
                pool.Set(offset + i, _mm_extract_epi64(first[i], 0));
                pool.choice[offset + i] = choices[i];
            }
        }
        return;
    }
    Utils::MITCCRH<8> tree;
    std::vector<Block> in0(n - 1), in1(n - 1), hashes(2 * n - 2), receiver_in(levels),
        receiver_out(levels);
    int index = 0;
    for (int level = 0; level < levels; ++level)
        for (int prefix = 0; prefix < (1 << level); ++prefix) {
            in0[index] = emp::makeBlock(prefix, 0);
            in1[index] = emp::makeBlock((1 << level) + prefix, 0);
            ++index;
        }
    for (size_t i = 0; i < count; ++i) {
        if (sender) {
            tree.renew_ks(first.data() + i * levels, levels);
            tree.hash_exp(hashes.data(), in0.data(), levels);
            tree.renew_ks(second.data() + i * levels, levels);
            tree.hash_exp(hashes.data() + n - 1, in1.data(), levels);
            for (int value = 0; value < n; ++value) {
                uint64_t pad = 0;
                int base = 0;
                for (int level = 0; level < levels; ++level) {
                    const int prefix = value & ((1 << level) - 1);
                    pad ^= _mm_extract_epi64(
                        hashes[base + prefix + ((value & (1 << level)) ? n - 1 : 0)], 0);
                    base += 1 << level;
                }
                pool.Set((offset + i) * n + value, pad);
            }
        } else {
            uint8_t value = 0;
            for (int level = 0; level < levels; ++level)
                value |= uint8_t(choices[i * levels + level]) << level;
            for (int level = 0; level < levels; ++level)
                receiver_in[level] = emp::makeBlock(value & ((1 << (level + 1)) - 1), 0);
            tree.renew_ks(first.data() + i * levels, levels);
            tree.hash_single(receiver_out.data(), receiver_in.data(), levels);
            uint64_t pad = 0;
            for (const auto &block : receiver_out)
                pad ^= _mm_extract_epi64(block, 0);
            pool.Set(offset + i, pad);
            pool.choice[offset + i] = value;
        }
    }
}

void GeneratePads(PreparedSilent &source, bool sender, int n, const Budget &budget, PadPool &pool) {
    const size_t count = budget.rows;
    pool.cursor = 0;
    pool.Resize(sender ? count * n : count, budget.payload_bits);
    pool.choice.resize(sender ? 0 : count);
    // Keep temporary ROT blocks bounded per worker even when preprocessing an entire VGG image.
    const size_t batch_rows = (1u << 18) / Log2(n);
    for (size_t offset = 0; offset < count; offset += batch_rows)
        GeneratePadBatch(source, sender, n, offset, std::min(batch_rows, count - offset), pool);
}

// The stock SilentOTN only implements byte tables. Wider tables are handled
// using independent byte-slice OTs with identical choices (fresh correlations).
class FerretTable : public BaseOT {
  public:
    FerretTable(PreparedSilent &silent, int arity) : silent_(silent), arity_(arity) {}
    void send(uint8_t **data, int length, int bits) override {
        silent_.send(data, length, arity_, bits);
    }
    void recv(uint8_t *data, uint8_t *choices, int length, int bits) override {
        silent_.recv(data, choices, length, arity_, bits);
    }
    void send(uint64_t **data, int length, int bits) override {
        if (arity_ == 2) {
            silent_.send(data, length, bits);
            return;
        }
        std::vector<uint8_t> storage(static_cast<size_t>(length) * arity_);
        std::vector<uint8_t *> rows(length);
        for (int i = 0; i < length; ++i)
            rows[i] = storage.data() + static_cast<size_t>(i) * arity_;
        for (int offset = 0; offset < bits; offset += 8) {
            const int width = std::min(8, bits - offset);
            for (int i = 0; i < length; ++i)
                for (int j = 0; j < arity_; ++j)
                    rows[i][j] = (data[i][j] >> offset) & Mask(width);
            send(rows.data(), length, width);
        }
    }
    void recv(uint64_t *data, uint8_t *choices, int length, int bits) override {
        if (arity_ == 2) {
            silent_.recv(data, choices, length, bits);
            return;
        }
        std::vector<uint8_t> slice(length);
        std::fill_n(data, length, 0);
        for (int offset = 0; offset < bits; offset += 8) {
            recv(slice.data(), choices, length, std::min(8, bits - offset));
            for (int i = 0; i < length; ++i)
                data[i] |= uint64_t(slice[i]) << offset;
        }
    }

  private:
    PreparedSilent &silent_;
    int arity_;
};

class TrackedOT : public BaseOT {
  public:
    TrackedOT(BaseOT &target, IO &io, int arity, OTBackend backend, OTPhase &phase, Budget &budget,
              Budget &raw, PadPool &pool)
        : target_(target), io_(io), arity_(arity), backend_(backend), phase_(phase),
          budget_(budget), raw_(raw), pool_(pool) {}
    void send(const Block *a, const Block *b, int n) override {
        if (Record(n, 0))
            return;
        target_.send(a, b, n);
    }
    void recv(Block *out, const bool *choices, int n) override {
        if (Record(n, 0)) {
            std::fill_n(out, n, Utils::zero_block());
            return;
        }
        target_.recv(out, choices, n);
    }
    void send(uint8_t **data, int n, int bits) override {
        if (!Record(n, bits)) {
            if (PreparedFerret())
                SendValues(data, n, bits);
            else
                target_.send(data, n, bits);
        }
    }
    void recv(uint8_t *out, uint8_t *choices, int n, int bits) override {
        if (Record(n, bits))
            std::fill_n(out, n, 0);
        else {
            if (PreparedFerret())
                RecvValues(out, choices, n, bits);
            else
                target_.recv(out, choices, n, bits);
        }
    }
    void recv(uint8_t *out, const uint8_t *choices, int n, int bits) override {
        std::vector<uint8_t> copy(choices, choices + n);
        recv(out, copy.data(), n, bits);
    }
    void send(uint64_t **data, int n, int bits) override {
        Mask(bits);
        if (!Record(n, bits)) {
            if (PreparedFerret())
                SendValues(data, n, bits);
            else
                target_.send(data, n, bits);
        }
    }
    void recv(uint64_t *out, uint8_t *choices, int n, int bits) override {
        Mask(bits);
        if (Record(n, bits))
            std::fill_n(out, n, 0);
        else {
            if (PreparedFerret())
                RecvValues(out, choices, n, bits);
            else
                target_.recv(out, choices, n, bits);
        }
    }
    void recv(uint64_t *out, const uint8_t *choices, int n, int bits) override {
        std::vector<uint8_t> copy(choices, choices + n);
        recv(out, copy.data(), n, bits);
    }
    void send(uint8_t **data, int n, int arity, int bits) override {
        if (arity != arity_)
            throw std::invalid_argument("OT table arity mismatch");
        send(data, n, bits);
    }
    void recv(uint8_t *out, const uint8_t *choices, int n, int arity, int bits) override {
        if (arity != arity_)
            throw std::invalid_argument("OT table arity mismatch");
        recv(out, choices, n, bits);
    }
    void recv(uint8_t *out, uint8_t *choices, int n, int arity, int bits) override {
        recv(out, static_cast<const uint8_t *>(choices), n, arity, bits);
    }
    void send_cot(uint64_t *out, uint64_t *corr, int n, int bits) override {
        const uint64_t mask = Mask(bits);
        if (Record(n, bits)) {
            std::fill_n(out, n, 0);
            return;
        }
        if (phase_ != OTPhase::Prepared) {
            target_.send_cot(out, corr, n, bits);
            return;
        }
        const size_t cursor = Cursor();
        std::unique_ptr<bool[]> correction(new bool[n]);
        io_.recv_bool(correction.get(), n);
        std::vector<uint64_t> payload(n), packed((uint64_t(n) * bits + 63) / 64);
        for (int i = 0; i < n; ++i) {
            const auto index = cursor + i;
            out[i] = SenderPad(index, correction[i]) & mask;
            payload[i] = (out[i] + corr[i] - SenderPad(index, 1 ^ correction[i])) & mask;
        }
        OTPrimitive::pack_cot_messages(packed.data(), payload.data(), packed.size(), n, bits);
        io_.send_data(packed.data(), packed.size() * 8);
        io_.flush();
        Advance(n);
    }
    void recv_cot(uint64_t *out, bool *choices, int n, int bits) override {
        const uint64_t mask = Mask(bits);
        if (Record(n, bits)) {
            std::fill_n(out, n, 0);
            return;
        }
        if (phase_ != OTPhase::Prepared) {
            target_.recv_cot(out, choices, n, bits);
            return;
        }
        const size_t cursor = Cursor();
        std::unique_ptr<bool[]> correction(new bool[n]);
        for (int i = 0; i < n; ++i)
            correction[i] = choices[i] ^ ReceiverChoice(cursor + i);
        io_.send_bool(correction.get(), n);
        io_.flush();
        std::vector<uint64_t> payload(n), packed((uint64_t(n) * bits + 63) / 64);
        io_.recv_data(packed.data(), packed.size() * 8);
        OTPrimitive::unpack_cot_messages(payload.data(), packed.data(), n, bits);
        for (int i = 0; i < n; ++i)
            out[i] = (ReceiverPad(cursor + i) + (choices[i] ? payload[i] : 0)) & mask;
        Advance(n);
    }

  private:
    BaseOT &target_;
    IO &io_;
    int arity_;
    OTBackend backend_;
    OTPhase &phase_;
    Budget &budget_, &raw_;
    PadPool &pool_;
    bool PreparedFerret() const {
        return backend_ == OTBackend::Ferret && phase_ == OTPhase::Prepared;
    }
    size_t Cursor() const {
        if (backend_ == OTBackend::Ferret)
            return pool_.cursor;
        if (auto *source = dynamic_cast<OTPrimitive::SplitIKNP<IO> *>(&target_))
            return source->counter;
        return dynamic_cast<OTPrimitive::SplitKKOT<IO> &>(target_).counter;
    }
    uint64_t SenderPad(size_t row, int choice) const {
        if (backend_ == OTBackend::Ferret)
            return pool_.Get(row * arity_ + choice);
        return dynamic_cast<OTPrimitive::SplitIKNP<IO> &>(target_).h64[choice][row];
    }
    uint64_t ReceiverPad(size_t row) const {
        if (backend_ == OTBackend::Ferret)
            return pool_.Get(row);
        return dynamic_cast<OTPrimitive::SplitIKNP<IO> &>(target_).h64[0][row];
    }
    uint8_t ReceiverChoice(size_t row) const {
        if (backend_ == OTBackend::Ferret)
            return pool_.choice[row];
        return dynamic_cast<OTPrimitive::SplitIKNP<IO> &>(target_).r_off[row];
    }
    void Advance(int n) {
        if (backend_ == OTBackend::Ferret)
            pool_.cursor += n;
        else
            dynamic_cast<OTPrimitive::SplitIKNP<IO> &>(target_).counter += n;
    }
    template <typename T> void SendValues(T **data, int n, int bits) {
        Mask(bits);
        std::vector<uint64_t> corrections(n), wire((uint64_t(n) * Log2(arity_) + 63) / 64);
        io_.recv_data(wire.data(), wire.size() * 8);
        OTPrimitive::unpack_cot_messages(corrections.data(), wire.data(), n, Log2(arity_));
        // The core unpack helper leaves high carrier bits on values spanning
        // a word boundary. Choices must remain in [0,N) before indexing pads.
        for (auto &value : corrections)
            value &= uint64_t(arity_ - 1);
        constexpr int batch = 64;
        std::vector<Block> pad(batch * arity_);
        std::vector<T> packed((batch * arity_ * bits + sizeof(T) * 8 - 1) / (sizeof(T) * 8));
        for (int i = 0; i < n; i += batch) {
            const int count = std::min(batch, n - i),
                      words = (count * arity_ * bits + sizeof(T) * 8 - 1) / (sizeof(T) * 8);
            for (int j = 0; j < count; ++j)
                for (int k = 0; k < arity_; ++k)
                    pad[j * arity_ + k] = Utils::makeBlock128(
                        0,
                        pool_.Get((pool_.cursor + i + j) * arity_ + (k ^ corrections[i + j])));
            OTPrimitive::pack_ot_messages<T>(packed.data(), data + i, pad.data(), words, count,
                                             bits, arity_);
            io_.send_data(packed.data(), words * sizeof(T));
        }
        pool_.cursor += n;
        io_.flush();
    }
    template <typename T> void RecvValues(T *out, uint8_t *choices, int n, int bits) {
        Mask(bits);
        std::vector<uint64_t> corrections(n), wire((uint64_t(n) * Log2(arity_) + 63) / 64);
        for (int i = 0; i < n; ++i)
            corrections[i] = choices[i] ^ pool_.choice[pool_.cursor + i];
        OTPrimitive::pack_cot_messages(wire.data(), corrections.data(), wire.size(), n,
                                       Log2(arity_));
        io_.send_data(wire.data(), wire.size() * 8);
        io_.flush();
        constexpr int batch = 64;
        std::vector<Block> pad(batch);
        std::vector<T> packed((batch * arity_ * bits + sizeof(T) * 8 - 1) / (sizeof(T) * 8));
        for (int i = 0; i < n; i += batch) {
            const int count = std::min(batch, n - i),
                      words = (count * arity_ * bits + sizeof(T) * 8 - 1) / (sizeof(T) * 8);
            for (int j = 0; j < count; ++j)
                pad[j] = Utils::makeBlock128(0, pool_.Get(pool_.cursor + i + j));
            io_.recv_data(packed.data(), words * sizeof(T));
            OTPrimitive::unpack_ot_messages<T>(out + i, choices + i, packed.data(), pad.data(),
                                               count, bits, arity_);
        }
        pool_.cursor += n;
    }
    bool Record(int n, int bits) {
        if (n < 0)
            throw std::invalid_argument("Negative OT length");
        if (phase_ == OTPhase::Planning) {
            if (bits == 0)
                throw std::invalid_argument(
                    "Preprocessing covers the scalar OT flavors used by this experiment");
            budget_.rows += n;
            budget_.payload_bits = std::max(budget_.payload_bits, bits);
            if (backend_ == OTBackend::Ferret)
                raw_.raw += uint64_t(n) * Log2(arity_);
            return true;
        }
        if (phase_ == OTPhase::Prepared) {
            if (bits == 0 || bits > budget_.payload_bits)
                throw std::runtime_error("OT payload width differs from the public plan");
            if (uint64_t(n) > budget_.rows - budget_.used)
                throw std::runtime_error("OT preprocessing schedule exhausted");
            if (Cursor() != budget_.used)
                throw std::runtime_error("OT pool cursor differs from the input-independent plan");
            budget_.used += n;
        }
        return false;
    }
};
} // namespace

class PreparedOTPack::Impl {
  public:
    OTPhase phase = OTPhase::Live;
    OTBackend backend;
    IO *ios[1];
    Budget budgets[KKOT_TYPES + 2], forward_raw, reverse_raw;
    PadPool pads[KKOT_TYPES + 2];
    std::unique_ptr<OTPrimitive::IKNPOTPack<IO>> native;
    std::unique_ptr<PreparedSilent> forward, reverse;
    std::vector<std::unique_ptr<FerretTable>> tables;
    std::vector<std::unique_ptr<TrackedOT>> tracked;
    Impl(IO *io, int party, OTBackend selected, const std::filesystem::path &directory)
        : backend(selected) {
        ios[0] = io;
        if (backend == OTBackend::IKNP)
            native = std::make_unique<OTPrimitive::IKNPOTPack<IO>>(io, party);
        else {
            std::filesystem::create_directories(directory);
            for (const auto &name : {"forward.state", "reverse.state"})
                if (std::filesystem::exists(directory / name))
                    throw std::invalid_argument("Ferret session state must be fresh");
            forward = std::make_unique<PreparedSilent>(
                party, ios, (directory / "forward.state").string(), phase);
            reverse = std::make_unique<PreparedSilent>(
                3 - party, ios, (directory / "reverse.state").string(), phase);
            for (int i = 0; i < KKOT_TYPES; ++i)
                tables.emplace_back(std::make_unique<FerretTable>(*forward, 1 << (i + 1)));
        }
    }
    BaseOT &Target(size_t i) {
        if (native)
            return i < KKOT_TYPES
                       ? *native->kkot[i]
                       : (i == KKOT_TYPES ? *native->iknp_straight : *native->iknp_reversed);
        return i < KKOT_TYPES ? static_cast<BaseOT &>(*tables[i])
                              : (i == KKOT_TYPES ? static_cast<BaseOT &>(*forward)
                                                 : static_cast<BaseOT &>(*reverse));
    }
};

PreparedOTPack::PreparedOTPack(IO *channel, int role, OTBackend backend,
                               const std::filesystem::path &directory)
    : OTPrimitive::OTPack<IO>(channel, role),
      impl_(std::make_unique<Impl>(channel, role, backend, directory)) {
    this->io = channel;
    this->party = role;
    this->do_setup = true;
    for (size_t i = 0; i < KKOT_TYPES + 2; ++i) {
        auto &raw = i == KKOT_TYPES + 1 ? impl_->reverse_raw : impl_->forward_raw;
        impl_->tracked.emplace_back(std::make_unique<TrackedOT>(
            impl_->Target(i), *channel, i < KKOT_TYPES ? 1 << (i + 1) : 2, backend, impl_->phase,
            impl_->budgets[i], raw, impl_->pads[i]));
        if (i < KKOT_TYPES)
            this->kkot[i] = impl_->tracked.back().get();
        else if (i == KKOT_TYPES)
            this->iknp_straight = impl_->tracked.back().get();
        else
            this->iknp_reversed = impl_->tracked.back().get();
    }
}
PreparedOTPack::~PreparedOTPack() = default;
void PreparedOTPack::BeginPlanning() {
    impl_->phase = OTPhase::Planning;
    for (auto &budget : impl_->budgets)
        budget = {};
    impl_->forward_raw = {};
    impl_->reverse_raw = {};
}
void PreparedOTPack::EndPlanning() {
    impl_->phase = OTPhase::Live;
}
void PreparedOTPack::UseLive() {
    impl_->phase = OTPhase::Live;
}
void PreparedOTPack::Preprocess() {
    impl_->phase = OTPhase::Live;
    if (impl_->backend == OTBackend::IKNP) {
        for (size_t i = 0; i < KKOT_TYPES + 2; ++i) {
            const uint64_t count = impl_->budgets[i].rows;
            if (count > INT32_MAX)
                throw std::overflow_error("OT preprocessing pool exceeds the native index range");
            if (!count)
                continue;
            if (i < KKOT_TYPES) {
                auto &source = dynamic_cast<OTPrimitive::SplitKKOT<IO> &>(impl_->Target(i));
                source.set_precomp_batch_size(count);
                source.preprocess();
            } else {
                auto &source = dynamic_cast<OTPrimitive::SplitIKNP<IO> &>(impl_->Target(i));
                source.set_precomp_batch_size(count);
                source.preprocess();
            }
        }
    } else {
        // Generate both correlations and their complete random table pads
        // offline. Online only applies choice corrections and masks payloads.
        impl_->forward->Cache().Prepare(0);
        impl_->reverse->Cache().Prepare(0);
        for (size_t i = 0; i < KKOT_TYPES + 2; ++i)
            if (impl_->budgets[i].rows) {
                const bool reversed = i == KKOT_TYPES + 1;
                GeneratePads(reversed ? *impl_->reverse : *impl_->forward,
                             reversed ? this->party == 2 : this->party == 1,
                             i < KKOT_TYPES ? 1 << (i + 1) : 2, impl_->budgets[i],
                             impl_->pads[i]);
            }
    }
    for (auto &budget : impl_->budgets)
        budget.used = 0;
    impl_->phase = OTPhase::Prepared;
}
void PreparedOTPack::CheckConsumed() const {
    if (impl_->phase != OTPhase::Prepared)
        return;
    for (const auto &budget : impl_->budgets)
        if (budget.used != budget.rows)
            throw std::runtime_error("OT consumption differs from the public preprocessing plan");
    if (impl_->backend == OTBackend::Ferret) {
        for (size_t i = 0; i < KKOT_TYPES + 2; ++i)
            if (impl_->pads[i].cursor != impl_->budgets[i].rows)
                throw std::runtime_error(
                    "Ferret random pad consumption differs from the public plan");
        impl_->forward->Cache().CheckConsumed();
        impl_->reverse->Cache().CheckConsumed();
    }
}
uint64_t PreparedOTPack::PlannedRows() const {
    uint64_t count = 0;
    for (const auto &budget : impl_->budgets)
        count += budget.rows;
    return count;
}
uint64_t PreparedOTPack::PlannedRawCOTs() const {
    return impl_->forward_raw.raw + impl_->reverse_raw.raw;
}
} // namespace QuantExperiment
