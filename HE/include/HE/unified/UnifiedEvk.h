#pragma once

#include "HE/unified/UnifiedContext.h"
#include <seal/galoiskeys.h>
#include <seal/relinkeys.h>

#ifdef USE_HE_GPU
#include <phantom/context.cuh>
#include <phantom/secretkey.h>
#endif

namespace HE {
namespace unified {

#ifdef USE_HE_GPU
void kswitchkey_to_device(const seal::SEALContext &hcontext,
                          const std::vector<seal::PublicKey> &hksk,
                          const PhantomContext &dcontext,
                          std::vector<PhantomPublicKey> &dksk);

inline void kswitchkey_to_device(const UnifiedContext &context,
                                 const std::vector<seal::PublicKey> &hksk,
                                 std::vector<PhantomPublicKey> &dksk) {
  kswitchkey_to_device(context, hksk, context, dksk);
}

void galoiskeys_to_device(const seal::SEALContext &hcontext,
                          const seal::GaloisKeys &hks,
                          const PhantomContext &dcontext,
                          PhantomGaloisKey &dks);

inline void galoiskeys_to_device(const UnifiedContext &context,
                                 const seal::GaloisKeys &hks,
                                 PhantomGaloisKey &dks) {
  galoiskeys_to_device(context, hks, context, dks);
}
#endif

// ******************** UnifiedRelinKeys ********************

class UnifiedRelinKeys {
public:
  UnifiedRelinKeys() = default;

  UnifiedRelinKeys(LOCATION loc = UNDEF) : loc_(loc) {}

  UnifiedRelinKeys(const seal::RelinKeys &key);

  UnifiedRelinKeys(seal::RelinKeys &&key);

#ifdef USE_HE_GPU
  // FIXME: needed?
  // UnifiedRelinKeys(const PhantomRelinKey &key);

  UnifiedRelinKeys(PhantomRelinKey &&key);
#endif

  ~UnifiedRelinKeys() = default;

  LOCATION location() const { return loc_; }

  bool is_host() const { return loc_ == HOST; }

  bool is_device() const { return loc_ == DEVICE; }

  const seal::RelinKeys &hrelin() const {
    if (loc_ != HOST && loc_ != HOST_AND_DEVICE) {
      throw std::runtime_error("UnifiedRelinKeys: NOT in HOST");
    }
    return host_relinkey_;
  }

  seal::RelinKeys &hrelin() {
    if (loc_ != HOST && loc_ != HOST_AND_DEVICE) {
      throw std::runtime_error("UnifiedRelinKeys: NOT in HOST");
    }
    return host_relinkey_;
  }

  operator const seal::RelinKeys &() const { return hrelin(); }

  operator seal::RelinKeys &() { return hrelin(); }

  void to_device(const UnifiedContext &context);

#ifdef USE_HE_GPU
  const PhantomRelinKey &drelin() const {
    if (loc_ != DEVICE && loc_ != HOST_AND_DEVICE) {
      throw std::runtime_error("UnifiedRelinKeys: NOT in DEVICE");
    }
    return device_relinkey_;
  }

  PhantomRelinKey &drelin() {
    if (loc_ != DEVICE && loc_ != HOST_AND_DEVICE) {
      throw std::runtime_error("UnifiedRelinKeys: NOT in DEVICE");
    }
    return device_relinkey_;
  }

  static void to_device(const seal::SEALContext &hcontext,
                        const seal::RelinKeys &hrelin,
                        const PhantomContext &dcontext,
                        PhantomRelinKey &drelin);

  operator const PhantomRelinKey &() const { return drelin(); }

  operator PhantomRelinKey &() { return drelin(); }
#endif

  // Unified API for SEAL & Phantom
  void save(std::ostream &stream) const;

  void load(const UnifiedContext &context, std::istream &stream);

private:
  LOCATION loc_ = UNDEF;

  seal::RelinKeys host_relinkey_;
#ifdef USE_HE_GPU
  PhantomRelinKey device_relinkey_;
#endif
};

// ******************** UnifiedGaloisKeys ********************

class UnifiedGaloisKeys {
public:
  UnifiedGaloisKeys() = default;

  UnifiedGaloisKeys(LOCATION loc = UNDEF) : loc_(loc) {}

  UnifiedGaloisKeys(const seal::GaloisKeys &key);

  UnifiedGaloisKeys(seal::GaloisKeys &&key);

#ifdef USE_HE_GPU
  UnifiedGaloisKeys(PhantomGaloisKey &&key);
#endif

  ~UnifiedGaloisKeys() = default;

  LOCATION location() const { return loc_; }

  bool is_host() const { return loc_ == HOST; }

  bool is_device() const { return loc_ == DEVICE; }

  const seal::GaloisKeys &hgalois() const {
    if (loc_ != HOST) {
      throw std::runtime_error("UnifiedGaloisKeys: NOT in HOST");
    }
    return host_galoiskey_;
  }

  seal::GaloisKeys &hgalois() {
    if (loc_ != HOST) {
      throw std::runtime_error("UnifiedGaloisKeys: NOT in HOST");
    }
    return host_galoiskey_;
  }

  operator const seal::GaloisKeys &() const { return hgalois(); }

  operator seal::GaloisKeys &() { return hgalois(); }

  void to_device(const UnifiedContext &context);

#ifdef USE_HE_GPU
  const PhantomGaloisKey &dgalois() const {
    if (loc_ != DEVICE) {
      throw std::runtime_error("UnifiedGaloisKeys: NOT in DEVICE");
    }
    return device_galoiskey_;
  }

  PhantomGaloisKey &dgalois() {
    if (loc_ != DEVICE) {
      throw std::runtime_error("UnifiedGaloisKeys: NOT in DEVICE");
    }
    return device_galoiskey_;
  }

  static void to_device(const seal::SEALContext &hcontext,
                        const seal::GaloisKeys &hrelin,
                        const PhantomContext &dcontext,
                        PhantomGaloisKey &drelin);

  operator const PhantomGaloisKey &() const { return dgalois(); }

  operator PhantomGaloisKey &() { return dgalois(); }
#endif

  // Unified API for SEAL & Phantom
  void save(std::ostream &stream) const;

  void load(const UnifiedContext &context, std::istream &stream);

private:
  LOCATION loc_ = UNDEF;

  seal::GaloisKeys host_galoiskey_;
#ifdef USE_HE_GPU
  PhantomGaloisKey device_galoiskey_;
#endif
};

} // namespace unified
} // namespace HE