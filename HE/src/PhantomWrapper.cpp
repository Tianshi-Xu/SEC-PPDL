#include "HE/unified/PhantomWrapper.h"
#include <phantom/evaluate.cuh>

using namespace HE;

void PhantomEvaluator::negate_inplace(PhantomCiphertext &encrypted) const {
  phantom::negate_inplace(context_, encrypted);
}

void PhantomEvaluator::add_inplace(PhantomCiphertext &encrypted1,
                                   const PhantomCiphertext &encrypted2) const {
  phantom::add_inplace(context_, encrypted1, encrypted2);
}

void PhantomEvaluator::sub_inplace(PhantomCiphertext &encrypted1,
                                   const PhantomCiphertext &encrypted2,
                                   bool negate) const {
  phantom::sub_inplace(context_, encrypted1, encrypted2, negate);
}

void PhantomEvaluator::add_plain_inplace(PhantomCiphertext &encrypted,
                                         const PhantomPlaintext &plain) const {
  phantom::add_plain_inplace(context_, encrypted, plain);
}

void PhantomEvaluator::sub_plain_inplace(PhantomCiphertext &encrypted,
                                         const PhantomPlaintext &plain) const {
  phantom::sub_plain_inplace(context_, encrypted, plain);
}

void PhantomEvaluator::multiply_plain_inplace(
    PhantomCiphertext &encrypted, const PhantomPlaintext &plain) const {
  phantom::multiply_plain_inplace(context_, encrypted, plain);
}

void PhantomEvaluator::multiply_inplace(
    PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) const {
  phantom::multiply_inplace(context_, encrypted1, encrypted2);
}

void PhantomEvaluator::relinearize_inplace(
    PhantomCiphertext &encrypted, const PhantomRelinKey &relin_keys) const {
  phantom::relinearize_inplace(context_, encrypted, relin_keys);
}

void PhantomEvaluator::rescale_to_next(const PhantomCiphertext &encrypted,
                                       PhantomCiphertext &destination) const {
  destination = std::move(phantom::rescale_to_next(context_, encrypted));
}

void PhantomEvaluator::mod_switch_to_next(
    const PhantomCiphertext &encrypted, PhantomCiphertext &destination) const {
  destination = std::move(phantom::mod_switch_to_next(context_, encrypted));
}

void PhantomEvaluator::rotate_vector_inplace(
    PhantomCiphertext &encrypted, int step,
    const PhantomGaloisKey &galois_key) const {
  const auto n_full_slots =
      context_.key_context_data().parms().poly_modulus_degree() >> 1;
  step %= n_full_slots;
  if (((step + n_full_slots) % n_full_slots) == 0) {
    return;
  }
  phantom::rotate_inplace(context_, encrypted, step, galois_key);
}

void PhantomEvaluator::complex_conjugate_inplace(
    PhantomCiphertext &encrypted, const PhantomGaloisKey &galois_key) const {
  phantom::rotate_inplace(context_, encrypted, 0, galois_key);
}