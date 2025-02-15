#pragma once

#include "HE/unified/UnifiedContext.h"
#include <seal/ciphertext.h>

#ifdef USE_HE_GPU
#include <phantom/ciphertext.h>
#include <phantom/context.cuh>
#include <phantom/secretkey.h>
#endif

namespace HE {
namespace unified {

class UnifiedCiphertext {
public:
  UnifiedCiphertext(LOCATION loc = UNDEF);

  UnifiedCiphertext(const seal::Ciphertext &cipher);

  UnifiedCiphertext(seal::Ciphertext &&cipher);

#ifdef USE_HE_GPU
  UnifiedCiphertext(const PhantomCiphertext &cipher);

  UnifiedCiphertext(PhantomCiphertext &&cipher);
#endif

  ~UnifiedCiphertext() = default;

  LOCATION location() const { return loc_; }

  bool is_host() const { return loc_ == HOST; }

  bool is_device() const { return loc_ == DEVICE; }

  const seal::Ciphertext &hcipher() const {
    if (loc_ != HOST) {
      throw std::runtime_error("UnifiedCiphertext: NOT in HOST");
    }
    return host_cipher_;
  }

  seal::Ciphertext &hcipher() {
    if (loc_ != HOST) {
      throw std::runtime_error("UnifiedCiphertext: NOT in HOST");
    }
    return host_cipher_;
  }

  operator const seal::Ciphertext &() const { return hcipher(); }

  operator seal::Ciphertext &() { return hcipher(); }

  void to_host(const UnifiedContext &context);

  void to_device(const UnifiedContext &context);

#ifdef USE_HE_GPU
  static void to_host(const PhantomContext &dcontext,
                      const PhantomCiphertext &dcipher,
                      const seal::SEALContext &hcontext,
                      seal::Ciphertext &hcipher);

  const PhantomCiphertext &dcipher() const {
    if (loc_ != DEVICE) {
      throw std::runtime_error("UnifiedCiphertext: NOT in DEVICE");
    }
    return device_cipher_;
  }

  PhantomCiphertext &dcipher() {
    if (loc_ != DEVICE) {
      throw std::runtime_error("UnifiedCiphertext: NOT in DEVICE");
    }
    return device_cipher_;
  }

  static void to_device(const seal::SEALContext &hcontext,
                        const seal::Ciphertext &hcipher,
                        const PhantomContext &dcontext,
                        PhantomCiphertext &dcipher);

  operator const PhantomCiphertext &() const { return dcipher(); }

  operator PhantomCiphertext &() { return dcipher(); }
#endif

  // Unified API for SEAL & Phantom
  void save(std::ostream &stream) const;

  void load(const UnifiedContext &context, std::istream &stream);

  std::size_t coeff_modulus_size() const;

  const double &scale() const;

  double &scale();

private:
  LOCATION loc_ = UNDEF;

  seal::Ciphertext host_cipher_;
#ifdef USE_HE_GPU
  PhantomCiphertext device_cipher_;
#endif
};

} // namespace unified
} // namespace HE