#pragma once
#include <QuantExperiment/Config.h>
#include <HE/HE.h>
#include <NonlinearOperator/FixPoint.h>
#include <functional>

namespace QuantExperiment {
class Context {
  public:
    explicit Context(Config config);
    ~Context();
    Config config;
    std::unique_ptr<HE::HEEvaluator> he;
    std::unique_ptr<NonlinearOperator::FixPoint<uint64_t>> fix;
    uint64_t Bytes() const;
    void Sync();
    void BeginPlanning();
    void EndPlanning();
    void Preprocess();
    void CheckConsumed() const;
    uint64_t PlannedRows() const;
    void RingToField(Datatype::Tensor<uint64_t> &input);
    void FieldToRing(Datatype::Tensor<uint64_t> &input);
    void ExtendUnsigned(Datatype::Tensor<uint64_t> &input, int from, int to);
    void ExtendSigned(Datatype::Tensor<uint64_t> &input, int from, int to);
    void ApproximateShiftNonnegative(Datatype::Tensor<uint64_t> &input, int shift);
    void Parallel(const std::function<void(size_t)> &function);

  private:
    std::filesystem::path session_directory_;
    std::vector<std::unique_ptr<Utils::NetIO>> channels_;
    std::vector<std::unique_ptr<PreparedOTPack>> packs_;
    std::vector<OTPrimitive::OTPack<Utils::NetIO> *> pack_views_;
    std::vector<std::unique_ptr<OTProtocol::TruncationProtocol>> truncation_;
    std::vector<std::unique_ptr<OTProtocol::AuxProtocols>> auxiliary_;
    std::vector<OTProtocol::TruncationProtocol *> truncation_views_;
    std::vector<OTProtocol::AuxProtocols *> auxiliary_views_;
    std::vector<std::unique_ptr<NonlinearOperator::FixPoint<uint64_t>>> lane_fix_;
    void ConvertChunks(Datatype::Tensor<uint64_t> &input,
                       const std::function<void(Datatype::Tensor<uint64_t> &, size_t)> &function,
                       bool parallel);
};
} // namespace QuantExperiment
