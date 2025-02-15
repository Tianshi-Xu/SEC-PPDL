#ifdef USE_HE_GPU

#include "HE/unified/UnifiedEvaluator.h"

using namespace HE::unified;

void UnifiedEvaluator::negate_inplace(UnifiedCiphertext &encrypted) const {
  if (backend_ == LOCATION::HOST) {
    seal_eval_->negate_inplace(encrypted);
  } else {
    phantom_eval_->negate_inplace(encrypted);
  }
}

void UnifiedEvaluator::add_inplace(UnifiedCiphertext &encrypted1,
                                   const UnifiedCiphertext &encrypted2) const {
  if (backend_ == LOCATION::HOST) {
    seal_eval_->add_inplace(encrypted1, encrypted2);
  } else {
    phantom_eval_->add_inplace(encrypted1, encrypted2);
  }
}

void UnifiedEvaluator::sub_inplace(UnifiedCiphertext &encrypted1,
                                   const UnifiedCiphertext &encrypted2) const {
  if (backend_ == LOCATION::HOST) {
    seal_eval_->sub_inplace(encrypted1, encrypted2);
  } else {
    phantom_eval_->sub_inplace(encrypted1, encrypted2);
  }
}

void UnifiedEvaluator::multiply_inplace(
    UnifiedCiphertext &encrypted1, const UnifiedCiphertext &encrypted2) const {
  if (backend_ == LOCATION::HOST) {
    seal_eval_->multiply_inplace(encrypted1, encrypted2);
  } else {
    phantom_eval_->multiply_inplace(encrypted1, encrypted2);
  }
}

template <typename RelinKey_t>
void UnifiedEvaluator::relinearize_inplace(UnifiedCiphertext &encrypted,
                                           const RelinKey_t &relin_keys) const {
  if constexpr (std::is_same_v<RelinKey_t, seal::RelinKeys>) {
    seal_eval_->relinearize_inplace(encrypted, relin_keys);
  } else if constexpr (std::is_same_v<RelinKey_t, PhantomRelinKey>) {
    phantom_eval_->relinearize_inplace(encrypted, relin_keys);
  }
}

void UnifiedEvaluator::mod_switch_to_next(
    const UnifiedCiphertext &encrypted, UnifiedCiphertext &destination) const {
  if (backend_ == LOCATION::HOST) {
    seal_eval_->mod_switch_to_next(encrypted, destination);
  } else {
    phantom_eval_->mod_switch_to_next(encrypted, destination);
  }
}

void UnifiedEvaluator::rescale_to_next(const UnifiedCiphertext &encrypted,
                                       UnifiedCiphertext &destination) const {
  if (backend_ == LOCATION::HOST) {
    seal_eval_->rescale_to_next(encrypted, destination);
  } else {
    phantom_eval_->rescale_to_next(encrypted, destination);
  }
}

void UnifiedEvaluator::add_plain_inplace(UnifiedCiphertext &encrypted,
                                         const UnifiedPlaintext &plain) const {
  if (backend_ == LOCATION::HOST) {
    seal_eval_->add_plain_inplace(encrypted, plain);
  } else {
    phantom_eval_->add_plain_inplace(encrypted, plain);
  }
}

void UnifiedEvaluator::sub_plain_inplace(UnifiedCiphertext &encrypted,
                                         const UnifiedPlaintext &plain) const {
  if (backend_ == LOCATION::HOST) {
    seal_eval_->sub_plain_inplace(encrypted, plain);
  } else {
    phantom_eval_->sub_plain_inplace(encrypted, plain);
  }
}

void UnifiedEvaluator::multiply_plain_inplace(
    UnifiedCiphertext &encrypted, const UnifiedPlaintext &plain) const {
  if (backend_ == LOCATION::HOST) {
    seal_eval_->multiply_plain_inplace(encrypted, plain);
  } else {
    phantom_eval_->multiply_plain_inplace(encrypted, plain);
  }
}

void UnifiedEvaluator::rotate_vector_inplace(
    UnifiedCiphertext &encrypted, int step,
    const UnifiedGaloisKeys &galois_key) const {
  if (backend_ == LOCATION::HOST) {
    seal_eval_->rotate_vector_inplace(encrypted, step, galois_key);
  } else {
    phantom_eval_->rotate_vector_inplace(encrypted, step, galois_key);
  }
}

void UnifiedEvaluator::complex_conjugate_inplace(
    UnifiedCiphertext &encrypted, const UnifiedGaloisKeys &galois_key) const {
  if (backend_ == LOCATION::HOST) {
    seal_eval_->complex_conjugate_inplace(encrypted, galois_key);
  } else {
    phantom_eval_->complex_conjugate_inplace(encrypted, galois_key);
  }
}

// function template specializations
template void UnifiedEvaluator::relinearize_inplace<seal::RelinKeys>(
    UnifiedCiphertext &encrypted, const seal::RelinKeys &relin_keys) const;

template void UnifiedEvaluator::relinearize_inplace<PhantomRelinKey>(
    UnifiedCiphertext &encrypted, const PhantomRelinKey &relin_keys) const;

#endif