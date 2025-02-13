#pragma once

#include "HE/unified/Define.h"
#include <phantom/ciphertext.h>
#include <phantom/context.cuh>
#include <phantom/secretkey.h>
#include <seal/ciphertext.h>

namespace HE {
namespace unified {

class UnifiedCiphertext {
public:
  UnifiedCiphertext(LOCATION loc = UNDEF);

  UnifiedCiphertext(const seal::Ciphertext &cipher);

  UnifiedCiphertext(seal::Ciphertext &&cipher);

  UnifiedCiphertext(const PhantomCiphertext &cipher);

  UnifiedCiphertext(PhantomCiphertext &&cipher);

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
  void to_device(const seal::SEALContext &hcontext,
                 const PhantomContext &dcontext);

  operator const seal::Ciphertext &() const { return hcipher(); }
  operator seal::Ciphertext &() { return hcipher(); }

  static void to_host(const PhantomContext &dcontext,
                      const PhantomCiphertext &dcipher,
                      const seal::SEALContext &hcontext,
                      seal::Ciphertext &hcipher);
  void to_host(const seal::SEALContext &hcontext,
               const PhantomContext &dcontext);

  operator const PhantomCiphertext &() const { return dcipher(); }
  operator PhantomCiphertext &() { return dcipher(); }

  // Unified API for SEAL & Phantom
  void save(std::ostream &stream) const;
  void load(const seal::SEALContext &hcontext, const PhantomContext &dcontext,
            std::istream &stream);

  std::size_t coeff_modulus_size() const;
  const double &scale() const;
  double &scale();

private:
  LOCATION loc_ = UNDEF;

  seal::Ciphertext host_cipher_;
  PhantomCiphertext device_cipher_;
};

void kswitchkey_to_device(const seal::SEALContext &hcontext,
                          const std::vector<seal::PublicKey> &hksk,
                          const PhantomContext &dcontext,
                          std::vector<PhantomPublicKey> &dksk);

void galoiskeys_to_device(const seal::SEALContext &hcontext,
                          const seal::GaloisKeys &hks,
                          const PhantomContext &dcontext,
                          PhantomGaloisKey &dks);

} // namespace unified
} // namespace HE