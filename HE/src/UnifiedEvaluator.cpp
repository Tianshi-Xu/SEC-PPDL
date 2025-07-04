#ifdef USE_HE_GPU

#include "HE/unified/UnifiedEvaluator.h"

using namespace HE::unified;

void UnifiedEvaluator::negate_inplace(UnifiedCiphertext &encrypted) const {
  backend_check(encrypted);
  if (encrypted.on_host()) {
    seal_eval_->negate_inplace(encrypted);
  } else {
    phantom_eval_->negate_inplace(encrypted);
  }
}

void UnifiedEvaluator::add_inplace(UnifiedCiphertext &encrypted1,
                                   const UnifiedCiphertext &encrypted2) const {
  backend_check(encrypted1, encrypted2);
  if (encrypted1.on_host() && encrypted2.on_host()) {
    seal_eval_->add_inplace(encrypted1, encrypted2);
  } else {
    phantom_eval_->add_inplace(encrypted1, encrypted2);
  }
}

void UnifiedEvaluator::sub_inplace(UnifiedCiphertext &encrypted1,
                                   const UnifiedCiphertext &encrypted2) const {
  backend_check(encrypted1, encrypted2);
  if (encrypted1.on_host() && encrypted2.on_host()) {
    seal_eval_->sub_inplace(encrypted1, encrypted2);
  } else {
    phantom_eval_->sub_inplace(encrypted1, encrypted2);
  }
}

void UnifiedEvaluator::multiply_inplace(
    UnifiedCiphertext &encrypted1, const UnifiedCiphertext &encrypted2) const {
  backend_check(encrypted1, encrypted2);
  if (encrypted1.on_host() && encrypted2.on_host()) {
    seal_eval_->multiply_inplace(encrypted1, encrypted2);
  } else {
    phantom_eval_->multiply_inplace(encrypted1, encrypted2);
  }
}

void UnifiedEvaluator::relinearize_inplace(UnifiedCiphertext &encrypted,
                                           const UnifiedRelinKeys &relin_keys) const {
  backend_check(encrypted);
  if (encrypted.on_host()) {
    seal_eval_->relinearize_inplace(encrypted, relin_keys);
  } else {
    phantom_eval_->relinearize_inplace(encrypted, relin_keys);
  }
}

void UnifiedEvaluator::mod_switch_to_next(
    const UnifiedCiphertext &encrypted, UnifiedCiphertext &destination) const {
  backend_check(encrypted);
  if (encrypted.on_host()) {
    seal_eval_->mod_switch_to_next(encrypted, destination);
  } else {
    phantom_eval_->mod_switch_to_next(encrypted, destination);
  }
}

void UnifiedEvaluator::rescale_to_next(const UnifiedCiphertext &encrypted,
                                       UnifiedCiphertext &destination) const {
  backend_check(encrypted);
  if (encrypted.on_host()) {
    seal_eval_->rescale_to_next(encrypted, destination);
  } else {
    phantom_eval_->rescale_to_next(encrypted, destination);
  }
}

void UnifiedEvaluator::add_plain_inplace(UnifiedCiphertext &encrypted,
                                         const UnifiedPlaintext &plain) const {
  backend_check(encrypted, plain);
  if (encrypted.on_host() && plain.on_host()) {
    // std::cout << "add_plain_inplace on host" << std::endl;
    seal_eval_->add_plain_inplace(encrypted, plain);
  } else {
    phantom_eval_->add_plain_inplace(encrypted, plain);
  }
}

void UnifiedEvaluator::sub_plain_inplace(UnifiedCiphertext &encrypted,
                                         const UnifiedPlaintext &plain) const {
  backend_check(encrypted, plain);
  if (encrypted.on_host() && plain.on_host()) {
    seal_eval_->sub_plain_inplace(encrypted, plain);
  } else {
    phantom_eval_->sub_plain_inplace(encrypted, plain);
  }
}

void UnifiedEvaluator::multiply_plain_inplace(
    UnifiedCiphertext &encrypted, const UnifiedPlaintext &plain) const {
  backend_check(encrypted, plain);
  if (encrypted.on_host() && plain.on_host()) {
    seal_eval_->multiply_plain_inplace(encrypted, plain);
  } else {
    phantom_eval_->multiply_plain_inplace(encrypted, plain);
  }
}

void UnifiedEvaluator::rotate_vector_inplace(
    UnifiedCiphertext &encrypted, int step,
    const UnifiedGaloisKeys &galois_key) const {
  backend_check(encrypted, galois_key);
  if (encrypted.on_host() && galois_key.on_host()) {
    seal_eval_->rotate_vector_inplace(encrypted, step, galois_key);
  } else {
    phantom_eval_->rotate_vector_inplace(encrypted, step, galois_key);
  }
}

void UnifiedEvaluator::complex_conjugate_inplace(
    UnifiedCiphertext &encrypted, const UnifiedGaloisKeys &galois_key) const {
  backend_check(encrypted, galois_key);
  if (encrypted.on_host() && galois_key.on_host()) {
    seal_eval_->complex_conjugate_inplace(encrypted, galois_key);
  } else {
    phantom_eval_->complex_conjugate_inplace(encrypted, galois_key);
  }
}

void UnifiedEvaluator::rotate_rows_inplace(
    UnifiedCiphertext &encrypted, int step,
    const UnifiedGaloisKeys &galois_key) const {
  backend_check(encrypted, galois_key);
  if (encrypted.on_host() && galois_key.on_host()) {
    seal_eval_->rotate_rows_inplace(encrypted, step, galois_key);
  } else {
    phantom_eval_->rotate_vector_inplace(encrypted, step, galois_key);
  }
}

void UnifiedEvaluator::rotate_columns_inplace(
    UnifiedCiphertext &encrypted, const UnifiedGaloisKeys &galois_key) const {
  backend_check(encrypted, galois_key);
  if (encrypted.on_host() && galois_key.on_host()) {
    seal_eval_->rotate_columns_inplace(encrypted, galois_key);
  } else {
    phantom_eval_->complex_conjugate_inplace(encrypted, galois_key);
  }
}

#endif