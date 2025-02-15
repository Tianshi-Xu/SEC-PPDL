#pragma once

#include "HE/unified/Define.h"
#include <seal/plaintext.h>
#ifdef USE_HE_GPU
#include <phantom/context.cuh>
#include <phantom/plaintext.h>
#endif

namespace HE {
namespace unified {

class UnifiedPlaintext {
public:
  UnifiedPlaintext(LOCATION loc = UNDEF) : loc_(loc) {}

  UnifiedPlaintext(const seal::Plaintext &hplain);

  UnifiedPlaintext(seal::Plaintext &&hplain);

#ifdef USE_HE_GPU
  UnifiedPlaintext(const PhantomPlaintext &dplain);

  UnifiedPlaintext(PhantomPlaintext &&dplain);
#endif

  ~UnifiedPlaintext() = default;

  UnifiedPlaintext &operator=(seal::Plaintext &&);

  inline bool on_host() const {
    return loc_ == HOST || loc_ == HOST_AND_DEVICE;
  }

  inline bool on_device() const {
    return loc_ == DEVICE || loc_ == HOST_AND_DEVICE;
  }

  const seal::Plaintext &hplain() const {
    if (on_host()) {
      return host_plain_;
    }
    throw std::runtime_error("UnifiedPlaintext: NOT in HOST");
  }

  seal::Plaintext &hplain() {
    if (on_host()) {
      return host_plain_;
    }
    throw std::runtime_error("UnifiedPlaintext: NOT in HOST");
  }

  operator const seal::Plaintext &() const { return hplain(); }

  operator seal::Plaintext &() { return hplain(); }

#ifdef USE_HE_GPU
  const PhantomPlaintext &dplain() const {
    if (on_device()) {
      return device_plain_;
    }
    throw std::runtime_error("UnifiedPlaintext: NOT in DEVICE");
  }

  PhantomPlaintext &dplain() {
    if (on_device()) {
      return device_plain_;
    }
    throw std::runtime_error("UnifiedPlaintext: NOT in DEVICE");
  }

  operator const PhantomPlaintext &() const { return dplain(); }

  operator PhantomPlaintext &() { return dplain(); }

  static void to_device(const seal::SEALContext &hcontext,
                        const seal::Plaintext &hplain,
                        const PhantomContext &dcontext,
                        PhantomPlaintext &dplain);

  void to_device(const seal::SEALContext &hcontext,
                 const PhantomContext &dcontext);
#endif

  const double &scale() const;

  double &scale();

private:
  LOCATION loc_ = UNDEF;

  seal::Plaintext host_plain_;
#ifdef USE_HE_GPU
  PhantomPlaintext device_plain_;
#endif
};

} // namespace unified
} // namespace HE