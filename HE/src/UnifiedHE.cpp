#include "HE/unified/UnifiedCiphertext.h"
#include "HE/unified/UnifiedEvk.h"
#include "HE/unified/UnifiedPlaintext.h"
#include <seal/galoiskeys.h>

using namespace HE::unified;

// ******************** UnifiedPlaintext ********************

UnifiedPlaintext::UnifiedPlaintext(const seal::Plaintext &hplain)
    : UnifiedBase(HOST), host_plain_(hplain) {}

UnifiedPlaintext::UnifiedPlaintext(seal::Plaintext &&hplain)
    : UnifiedBase(HOST), host_plain_(std::move(hplain)) {}

void UnifiedPlaintext::to_device(const UnifiedContext &context) {
  if (loc_ != HOST) {
    throw std::runtime_error("UnifiedPlaintext: NOT in HOST");
  }
#ifdef USE_HE_GPU
  to_device(context, host_plain_, context, device_plain_);
  host_plain_.release();
  loc_ = DEVICE;
#endif
}

#ifdef USE_HE_GPU
UnifiedPlaintext::UnifiedPlaintext(const PhantomPlaintext &dplain)
    : UnifiedBase(DEVICE), device_plain_(dplain) {}

UnifiedPlaintext::UnifiedPlaintext(PhantomPlaintext &&dplain)
    : UnifiedBase(DEVICE), device_plain_(std::move(dplain)) {}

void UnifiedPlaintext::to_device(const seal::SEALContext &hcontext,
                                 const seal::Plaintext &hplain,
                                 const PhantomContext &dcontext,
                                 PhantomPlaintext &dplain) {
  const auto &first_parms = hcontext.first_context_data()->parms();
  const auto full_data_modulus_size = first_parms.coeff_modulus().size();

  auto data_ptr = hcontext.get_context_data(hplain.parms_id());
  if (!data_ptr) {
    data_ptr = hcontext.first_context_data();
  }
  const auto &curr_parms = data_ptr->parms();
  const auto curr_data_modulus_size = curr_parms.coeff_modulus().size();

  // * [Important] phantom.chain_index + seal.chain_index = size_Q
  auto phantom_chain_idx = full_data_modulus_size - curr_data_modulus_size + 1;

  dplain.load(hplain.data(), dcontext, phantom_chain_idx, hplain.scale());
}
#endif

const double &UnifiedPlaintext::scale() const {
  switch (loc_) {
  case HOST_AND_DEVICE:
  case HOST:
    return host_plain_.scale();
#ifdef USE_HE_GPU
  case DEVICE:
    return device_plain_.scale();
    break;
#endif
  default:
    throw std::invalid_argument("Invalid UnifiedPlaintext");
  }
}

double &UnifiedPlaintext::scale() {
  switch (loc_) {
  case HOST_AND_DEVICE:
  case HOST:
    return host_plain_.scale();
#ifdef USE_HE_GPU
  case DEVICE:
    return device_plain_.scale();
    break;
#endif
  default:
    throw std::invalid_argument("Invalid UnifiedPlaintext");
  }
}

// ******************** UnifiedCiphertext ********************

UnifiedCiphertext::UnifiedCiphertext(const seal::Ciphertext &cipher)
    : UnifiedBase(HOST), host_cipher_(cipher) {}

UnifiedCiphertext::UnifiedCiphertext(seal::Ciphertext &&cipher)
    : UnifiedBase(HOST), host_cipher_(std::move(cipher)) {}

#ifdef USE_HE_GPU
UnifiedCiphertext::UnifiedCiphertext(const PhantomCiphertext &cipher)
    : UnifiedBase(DEVICE), device_cipher_(cipher) {}

UnifiedCiphertext::UnifiedCiphertext(PhantomCiphertext &&cipher)
    : UnifiedBase(DEVICE), device_cipher_(std::move(cipher)) {}

void UnifiedCiphertext::to_device(const seal::SEALContext &hcontext,
                                  const seal::Ciphertext &hcipher,
                                  const PhantomContext &dcontext,
                                  PhantomCiphertext &dcipher) {
  const auto &first_parms = hcontext.first_context_data()->parms();
  const auto full_data_modulus_size = first_parms.coeff_modulus().size();
  const auto curr_data_modulus_size = hcipher.coeff_modulus_size();

  // * [Important] phantom.chain_index + seal.chain_index = size_Q
  auto phantom_chain_idx =
      (full_data_modulus_size + 1) - curr_data_modulus_size;

  // `noiseScaleDeg` in Phantom is always 1
  // 2PC : `is_asymmetric` is always `true`
  dcipher.load(hcipher.data(), dcontext, phantom_chain_idx, hcipher.size(),
               hcipher.scale(), hcipher.correction_factor(), 1,
               hcipher.is_ntt_form(), true);
}

void UnifiedCiphertext::to_host(const PhantomContext &dcontext,
                                const PhantomCiphertext &dcipher,
                                const seal::SEALContext &hcontext,
                                seal::Ciphertext &hcipher) {
  const auto chain_idx = dcipher.chain_index();
  const auto curr_data_modulus_size = dcipher.coeff_modulus_size();

  // PhantomCipher only holds the `chain_index`.
  // But SEAL indexes context_data with `param_id`.
  // * [Important] phantom.chain_index + seal.chain_index = size_Q
  // * Match context through `data_modulus_size`
  auto target_context = hcontext.first_context_data();
  while (target_context->parms().coeff_modulus().size() !=
         curr_data_modulus_size) {
    target_context = target_context->next_context_data();
  }
  const auto &parms = dcontext.get_context_data(chain_idx).parms();
  const auto coeff_modulus_size = parms.coeff_modulus().size();
  const auto poly_modulus_degree = parms.poly_modulus_degree();
  const auto size = dcipher.size() * coeff_modulus_size * poly_modulus_degree;

  hcipher.resize(hcontext, target_context->parms_id(), dcipher.size());
  hcipher.scale() = dcipher.scale();
  hcipher.correction_factor() = dcipher.correction_factor();
  hcipher.is_ntt_form() = dcipher.is_ntt_form();
  cudaMemcpy(hcipher.data(), dcipher.data(), size * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
}
#endif

void UnifiedCiphertext::to_device(const UnifiedContext &context) {
  if (loc_ != HOST) {
    throw std::runtime_error("UnifiedCiphertext: NOT in HOST");
  }
#ifdef USE_HE_GPU
  to_device(context, host_cipher_, context, device_cipher_);
  host_cipher_.release();
  loc_ = DEVICE;
#endif
}

void UnifiedCiphertext::to_host(const UnifiedContext &context) {
  if (loc_ != DEVICE) {
    throw std::runtime_error("UnifiedCiphertext: NOT in DEVICE");
  }
#ifdef USE_HE_GPU
  to_host(context, device_cipher_, context, host_cipher_);
  // device_cipher_.resize(0, 0, 0, cudaStreamPerThread);
  loc_ = HOST;
#endif
}

void UnifiedCiphertext::save(std::ostream &stream) const {
  if (loc_ == UNDEF) {
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
  std::size_t loc = loc_;
  stream.write(reinterpret_cast<const char *>(&loc), sizeof(std::size_t));
  switch (loc_) {
  case HOST:
    host_cipher_.save(stream);
    break;
#ifdef USE_HE_GPU
  case DEVICE:
    device_cipher_.save(stream);
    break;
#endif
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

void UnifiedCiphertext::load(const UnifiedContext &context,
                             std::istream &stream) {
  std::size_t loc;
  stream.read(reinterpret_cast<char *>(&loc), sizeof(std::size_t));
  loc_ = static_cast<LOCATION>(loc);
  switch (loc_) {
  case HOST:
    host_cipher_.load(context, stream);
    break;
#ifdef USE_HE_GPU
  case DEVICE:
    device_cipher_.load(stream);
    break;
#endif
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

std::size_t UnifiedCiphertext::coeff_modulus_size() const {
  switch (loc_) {
  case HOST:
    return host_cipher_.coeff_modulus_size();
#ifdef USE_HE_GPU
  case DEVICE:
    return device_cipher_.coeff_modulus_size();
    break;
#endif
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

const double &UnifiedCiphertext::scale() const {
  switch (loc_) {
  case HOST:
    return host_cipher_.scale();
#ifdef USE_HE_GPU
  case DEVICE:
    return device_cipher_.scale();
    break;
#endif
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

double &UnifiedCiphertext::scale() {
  switch (loc_) {
  case HOST:
    return host_cipher_.scale();
#ifdef USE_HE_GPU
  case DEVICE:
    return device_cipher_.scale();
    break;
#endif
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

#ifdef USE_HE_GPU
void HE::unified::kswitchkey_to_device(const seal::SEALContext &hcontext,
                                       const std::vector<seal::PublicKey> &hksk,
                                       const PhantomContext &dcontext,
                                       std::vector<PhantomPublicKey> &dksk) {
  auto dnum = hksk.size();
  dksk.resize(dnum);
  for (size_t i = 0; i < hksk.size(); i++) {
    PhantomCiphertext tmp;
    UnifiedCiphertext::to_device(hcontext, hksk[i].data(), dcontext, tmp);
    dksk[i].load(std::move(tmp));
  }
}

void HE::unified::galoiskeys_to_device(const seal::SEALContext &hcontext,
                                       const seal::GaloisKeys &hks,
                                       const PhantomContext &dcontext,
                                       PhantomGaloisKey &dks) {
  std::vector<uint32_t> galois_elts;
  std::vector<std::vector<PhantomPublicKey>> dgks;

  const auto &sgks = hks.data();
  for (size_t galois_elt_index = 0; galois_elt_index < sgks.size();
       galois_elt_index++) {
    std::vector<PhantomPublicKey> dgk;
    kswitchkey_to_device(hcontext, sgks[galois_elt_index], dcontext, dgk);
    if (!dgk.empty()) {
      // * [Important] phantom uses `galois_elts` to index `GaloisKeys`
      galois_elts.push_back((galois_elt_index << 1) | 1);
      // * PhantomGaloisKey only has a move constructor (using std::move)
      dgks.emplace_back(std::move(dgk));
    }
  }

  dks.load(dgks);
  const_cast<PhantomContext *>(&dcontext)->key_galois_tool_->galois_elts(
      galois_elts);
}
#endif

// ******************** UnifiedRelinKeys ********************

UnifiedRelinKeys::UnifiedRelinKeys(const seal::RelinKeys &key)
    : loc_(HOST), host_relinkey_(key) {}

UnifiedRelinKeys::UnifiedRelinKeys(seal::RelinKeys &&key)
    : loc_(HOST), host_relinkey_(std::move(key)) {}

void UnifiedRelinKeys::save(std::ostream &stream) const {
  if (loc_ != HOST) {
    throw std::runtime_error(
        "UnifiedRelinKeys: Only supports HOST-side serialization");
  }
  host_relinkey_.save(stream);
}

void UnifiedRelinKeys::load(const UnifiedContext &context,
                            std::istream &stream) {
  if (loc_ != HOST) {
    throw std::runtime_error(
        "UnifiedRelinKeys: Only supports HOST-side serialization");
  }
  host_relinkey_.load(context, stream);
}

void UnifiedRelinKeys::to_device(const UnifiedContext &context) {
  if (loc_ != HOST) {
    throw std::runtime_error("UnifiedRelinKeys: NOT in HOST");
  }
#ifdef USE_HE_GPU
  to_device(context, host_relinkey_, context, device_relinkey_);
  loc_ = HOST_AND_DEVICE;
#endif
}

#ifdef USE_HE_GPU
UnifiedRelinKeys::UnifiedRelinKeys(PhantomRelinKey &&key)
    : loc_(DEVICE), device_relinkey_(std::move(key)) {}

void UnifiedRelinKeys::to_device(const seal::SEALContext &hcontext,
                                 const seal::RelinKeys &hrelin,
                                 const PhantomContext &dcontext,
                                 PhantomRelinKey &drelin) {
  std::vector<PhantomPublicKey> drlk;
  kswitchkey_to_device(hcontext, *hrelin.data().data(), dcontext, drlk);
  drelin.load(std::move(drlk));
}
#endif

// ******************** UnifiedGaloisKeys ********************

UnifiedGaloisKeys::UnifiedGaloisKeys(const seal::GaloisKeys &key)
    : UnifiedBase(HOST), host_galoiskey_(key) {}

UnifiedGaloisKeys::UnifiedGaloisKeys(seal::GaloisKeys &&key)
    : UnifiedBase(HOST), host_galoiskey_(std::move(key)) {}

void UnifiedGaloisKeys::save(std::ostream &stream) const {
  if (loc_ != HOST) {
    throw std::runtime_error(
        "UnifiedRelinKeys: Only supports HOST-side serialization");
  }
  host_galoiskey_.save(stream);
}

void UnifiedGaloisKeys::load(const UnifiedContext &context,
                             std::istream &stream) {
  if (loc_ != HOST) {
    throw std::runtime_error(
        "UnifiedRelinKeys: Only supports HOST-side serialization");
  }
  host_galoiskey_.load(context, stream);
}

void UnifiedGaloisKeys::to_device(const UnifiedContext &context) {
  if (loc_ != HOST) {
    throw std::runtime_error("UnifiedGaloisKeys: NOT in HOST");
  }
#ifdef USE_HE_GPU
  to_device(context, host_galoiskey_, context, device_galoiskey_);
  loc_ = HOST_AND_DEVICE;
#endif
}

#ifdef USE_HE_GPU
UnifiedGaloisKeys::UnifiedGaloisKeys(PhantomGaloisKey &&key)
    : UnifiedBase(DEVICE), device_galoiskey_(std::move(key)) {}

void UnifiedGaloisKeys::to_device(const seal::SEALContext &hcontext,
                                  const seal::GaloisKeys &hgalois,
                                  const PhantomContext &dcontext,
                                  PhantomGaloisKey &dgalois) {
  galoiskeys_to_device(hcontext, hgalois, dcontext, dgalois);
}
#endif