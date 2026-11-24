#pragma once
#include <OTPrimitive/ot_pack.h>
#include <Utils/net_io_channel.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace QuantExperiment {
enum class OTBackend { IKNP, Ferret };
enum class OTPhase { Live, Planning, Prepared };

// Planning is restricted to public dummy tensors by the experiment's planner.
// Prepared correlations are input-independent, consumed once, and never refilled
// silently during online execution. Protocol layout uses IKNP-style leaf/triple
// algorithms independently of the OT transport selected here.
class PreparedOTPack : public OTPrimitive::OTPack<Utils::NetIO> {
  public:
    PreparedOTPack(Utils::NetIO *io, int party, OTBackend backend,
                   const std::filesystem::path &session_directory);
    ~PreparedOTPack();
    void BeginPlanning();
    void EndPlanning();
    void Preprocess();
    void UseLive();
    void CheckConsumed() const;
    uint64_t PlannedRows() const;
    uint64_t PlannedRawCOTs() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace QuantExperiment
