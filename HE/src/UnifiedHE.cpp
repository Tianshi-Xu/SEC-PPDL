#include "HE/unified/UnifiedCiphertext.h"
#include "HE/unified/UnifiedPlaintext.h"
#include <seal/galoiskeys.h>

using namespace HE::unified;

// ******************** UnifiedPlaintext ********************

UnifiedPlaintext::UnifiedPlaintext(const seal::Plaintext &hplain)
    : loc_(HOST), host_plain_(hplain) {}

UnifiedPlaintext::UnifiedPlaintext(seal::Plaintext &&hplain)
    : loc_(HOST), host_plain_(std::move(hplain)) {}

UnifiedPlaintext::UnifiedPlaintext(const PhantomPlaintext &dplain)
    : loc_(DEVICE), device_plain_(dplain) {}

UnifiedPlaintext::UnifiedPlaintext(PhantomPlaintext &&dplain)
    : loc_(DEVICE), device_plain_(std::move(dplain)) {}

UnifiedPlaintext &UnifiedPlaintext::operator=(seal::Plaintext &&other) {
  loc_ = HOST;
  host_plain_ = std::move(other);
  return *this;
}

void UnifiedPlaintext::to_device(const seal::SEALContext &hcontext,
                                 const seal::Plaintext &hplain,
                                 const PhantomContext &dcontext,
                                 PhantomPlaintext &dplain) {
  const auto &first_parms = hcontext.first_context_data()->parms();
  const auto full_data_modulus_size = first_parms.coeff_modulus().size();
  const auto &curr_parms =
      hcontext.get_context_data(hplain.parms_id())->parms();
  const auto curr_data_modulus_size = curr_parms.coeff_modulus().size();

  // * [Important] phantom.chain_index + seal.chain_index = size_Q
  auto phantom_chain_idx = full_data_modulus_size - curr_data_modulus_size + 1;

  dplain.load(hplain.data(), dcontext, phantom_chain_idx, hplain.scale());
}

void UnifiedPlaintext::to_device(const seal::SEALContext &hcontext,
                                 const PhantomContext &dcontext) {
  if (loc_ != HOST) {
    throw std::runtime_error("UnifiedPlaintext: NOT in HOST");
  }
  to_device(hcontext, host_plain_, dcontext, device_plain_);
  host_plain_.release();
  loc_ = DEVICE;
}

const double &UnifiedPlaintext::scale() const {
  switch (loc_) {
  case HOST_AND_DEVICE:
  case HOST:
    return host_plain_.scale();
  case DEVICE:
    return device_plain_.scale();
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedPlaintext");
  }
}

double &UnifiedPlaintext::scale() {
  switch (loc_) {
  case HOST_AND_DEVICE:
  case HOST:
    return host_plain_.scale();
  case DEVICE:
    return device_plain_.scale();
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedPlaintext");
  }
}

// ******************** UnifiedCiphertext ********************

UnifiedCiphertext::UnifiedCiphertext(LOCATION loc) : loc_(loc) {}

UnifiedCiphertext::UnifiedCiphertext(const seal::Ciphertext &cipher)
    : loc_(HOST), host_cipher_(cipher) {}

UnifiedCiphertext::UnifiedCiphertext(seal::Ciphertext &&cipher)
    : loc_(HOST), host_cipher_(std::move(cipher)) {}

UnifiedCiphertext::UnifiedCiphertext(const PhantomCiphertext &cipher)
    : loc_(DEVICE), device_cipher_(cipher) {}

UnifiedCiphertext::UnifiedCiphertext(PhantomCiphertext &&cipher)
    : loc_(DEVICE), device_cipher_(std::move(cipher)) {}

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

void UnifiedCiphertext::to_device(const seal::SEALContext &hcontext,
                                  const PhantomContext &dcontext) {
  if (loc_ != HOST) {
    throw std::runtime_error("UnifiedCiphertext: NOT in HOST");
  }
  to_device(hcontext, host_cipher_, dcontext, device_cipher_);
  host_cipher_.release();
  loc_ = DEVICE;
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
  cudaMemcpy(dcipher.data(), hcipher.data(),
             size * coeff_modulus_size * poly_modulus_degree * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
}

void UnifiedCiphertext::to_host(const seal::SEALContext &hcontext,
                                const PhantomContext &dcontext) {
  if (loc_ != DEVICE) {
    throw std::runtime_error("UnifiedCiphertext: NOT in DEVICE");
  }
  const auto chain_idx = device_cipher_.chain_index();

  to_host(dcontext, device_cipher_, hcontext, host_cipher_);
  // device_cipher_.resize(0, 0, 0, cudaStreamPerThread);
  loc_ = HOST;
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
  case DEVICE:
    device_cipher_.save(stream);
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

void UnifiedCiphertext::load(const seal::SEALContext &hcontext,
                             const PhantomContext &dcontext,
                             std::istream &stream) {
  stream.read(reinterpret_cast<char *>(&loc_), sizeof(std::size_t));
  switch (loc_) {
  case HOST:
    // FIXME: why using unsafe_load
    host_cipher_.unsafe_load(hcontext, stream);
    break;
  case DEVICE:
    device_cipher_.load(stream);
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

std::size_t UnifiedCiphertext::coeff_modulus_size() const {
  switch (loc_) {
  case HOST:
    return host_cipher_.coeff_modulus_size();
  case DEVICE:
    return device_cipher_.coeff_modulus_size();
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

const double &UnifiedCiphertext::scale() const {
  switch (loc_) {
  case HOST:
    return host_cipher_.scale();
  case DEVICE:
    return device_cipher_.scale();
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

double &UnifiedCiphertext::scale() {
  switch (loc_) {
  case HOST:
    return host_cipher_.scale();
  case DEVICE:
    return device_cipher_.scale();
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

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