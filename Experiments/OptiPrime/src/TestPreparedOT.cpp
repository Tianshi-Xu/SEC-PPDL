#include <QuantExperiment/PreparedOT.h>
#include <Utils/ArgMapping/ArgMapping.h>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <unistd.h>

namespace {
using IO = Utils::NetIO;
using Pack = QuantExperiment::PreparedOTPack;
uint64_t Mask(int b) {
    return b == 64 ? UINT64_MAX : (uint64_t(1) << b) - 1;
}
void Suite(Pack &pack, IO &io, int party, bool check) {
    constexpr int count = 137;
    for (int n : {2, 4, 8, 16, 32, 64, 128, 256})
        for (int width : {1, 2, 8}) {
            std::vector<std::vector<uint8_t>> messages(count, std::vector<uint8_t>(n));
            std::vector<uint8_t *> rows(count);
            std::vector<uint8_t> choices(count), out(count);
            for (int i = 0; i < count; ++i) {
                rows[i] = messages[i].data();
                choices[i] = (i * 7 + 3) % n;
                for (int j = 0; j < n; ++j)
                    messages[i][j] = (i * 13 + j * 17 + 5) & Mask(width);
            }
            auto *ot = pack.kkot[__builtin_ctz(n) - 1];
            if (party == 1)
                ot->send(rows.data(), count, width);
            else {
                ot->recv(out.data(), choices.data(), count, width);
                if (check)
                    for (int i = 0; i < count; ++i)
                        if (out[i] != messages[i][choices[i]])
                            throw std::runtime_error("Byte table OT mismatch");
            }
        }
    for (int n : {2, 4, 16, 256})
        for (int width : {9, 26, 64}) {
            std::vector<std::vector<uint64_t>> messages(count, std::vector<uint64_t>(n));
            std::vector<uint64_t *> rows(count);
            std::vector<uint8_t> choices(count);
            std::vector<uint64_t> out(count);
            for (int i = 0; i < count; ++i) {
                rows[i] = messages[i].data();
                choices[i] = (i * 7 + 3) % n;
                for (int j = 0; j < n; ++j)
                    messages[i][j] =
                        (uint64_t(i + 1) * 0x9e3779b97f4a7c15ULL + uint64_t(j) * 37) & Mask(width);
            }
            auto *ot = pack.kkot[__builtin_ctz(n) - 1];
            if (party == 1)
                ot->send(rows.data(), count, width);
            else {
                ot->recv(out.data(), choices.data(), count, width);
                if (check)
                    for (int i = 0; i < count; ++i)
                        if (out[i] != messages[i][choices[i]])
                            throw std::runtime_error("Wide table OT mismatch");
            }
        }
    for (int reverse : {0, 1})
        for (int width : {1, 10, 26, 32, 64}) {
            auto *ot = reverse ? pack.iknp_reversed : pack.iknp_straight;
            const bool sender = party == (reverse ? 2 : 1);
            std::vector<uint64_t> output(count), corr(count), peer(count);
            std::unique_ptr<bool[]> choices(new bool[count]);
            for (int i = 0; i < count; ++i) {
                corr[i] = (uint64_t(i + 1) * 0x9e3779b97f4a7c15ULL) & Mask(width);
                choices[i] = i % 3 != 0;
            }
            if (sender)
                ot->send_cot(output.data(), corr.data(), count, width);
            else
                ot->recv_cot(output.data(), choices.get(), count, width);
            if (check) {
                if (sender) {
                    io.send_data(output.data(), output.size() * 8);
                    io.flush();
                } else {
                    io.recv_data(peer.data(), peer.size() * 8);
                    for (int i = 0; i < count; ++i)
                        if (output[i] != ((peer[i] + (choices[i] ? corr[i] : 0)) & Mask(width)))
                            throw std::runtime_error("Additive correlated OT mismatch");
                }
            }
        }
}
} // namespace
int main(int argc, char **argv) {
    try {
        int party = 1, port = 22000, backend = 0, prep = 1;
        ArgMapping args;
        args.arg("r", party);
        args.arg("p", port);
        args.arg("ot", backend);
        args.arg("prep", prep);
        args.parse(argc, argv);
        IO io(party == 1 ? nullptr : "127.0.0.1", port);
        const auto directory = std::filesystem::temp_directory_path() /
                               ("sec-ppdl-ot-test-" + std::to_string(getpid()));
        {
            Pack pack(&io, party,
                      backend ? QuantExperiment::OTBackend::Ferret
                              : QuantExperiment::OTBackend::IKNP,
                      directory);
            pack.BeginPlanning();
            Suite(pack, io, party, false);
            pack.EndPlanning();
            io.sync();
            for (int round = 0; round < 2; ++round) {
                const auto before = io.counter;
                if (prep)
                    pack.Preprocess();
                else
                    pack.UseLive();
                io.sync();
                const auto offline = io.counter - before;
                const auto begin = io.counter;
                Suite(pack, io, party, true);
                pack.CheckConsumed();
                io.sync();
                std::cout << "PASS prepared-ot backend=" << backend << " prep=" << prep
                          << " round=" << round << " planned_rows=" << pack.PlannedRows()
                          << " raw_cot=" << pack.PlannedRawCOTs() << " offline_bytes=" << offline
                          << " online_bytes=" << io.counter - begin << std::endl;
            }
            // Consumption is one-time; exhaustion must not silently refill.
            if (prep) {
                bool exhausted = false;
                uint64_t out = 0, corr = 1;
                bool choice = true;
                try {
                    if (party == 1)
                        pack.iknp_straight->send_cot(&out, &corr, 1, 10);
                    else
                        pack.iknp_straight->recv_cot(&out, &choice, 1, 10);
                } catch (const std::runtime_error &) {
                    exhausted = true;
                }
                if (!exhausted)
                    throw std::runtime_error("OT pool allowed correlation reuse/refill");
            }
        }
        std::filesystem::remove_all(directory);
        std::cout << "PROTOCOL CHECK COMPLETE" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
