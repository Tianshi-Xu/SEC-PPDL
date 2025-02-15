#pragma once

#include "HE/unified/UnifiedContext.h"
#include "HE/unified/UnifiedPlaintext.h"
#include <seal/batchencoder.h>

#ifdef USE_HE_GPU
#include <phantom/batchencoder.h>
#endif

namespace HE {
namespace unified {

class UnifiedBatchEncoder : public seal::BatchEncoder {
public:
  UnifiedBatchEncoder(const UnifiedContext &context)
      : seal::BatchEncoder::BatchEncoder(context), context_(context) {
#ifdef USE_HE_GPU
    if (context.backend() == DEVICE) {
      device_encoder_ = std::make_unique<PhantomBatchEncoder>(context);
    }
#endif
  }

  void encode(const std::vector<std::uint64_t> &values,
              UnifiedPlaintext &destination) const {
    if (destination.on_host()) {
      seal::BatchEncoder::encode(values, destination);
    }
#ifdef USE_HE_GPU
    if (destination.on_device()) {
      device_encoder_->encode(context_, values, destination);
    }
#endif
  }

private:
  const UnifiedContext &context_;
#ifdef USE_HE_GPU
  std::unique_ptr<PhantomBatchEncoder> device_encoder_;
#endif
};

} // namespace unified
} // namespace HE