#pragma once

#include "HE/unified/Define.h"
#include "HE/unified/UnifiedEvk.h"
#include <HE/unified/UnifiedContext.h>
#include <seal/evaluator.h>

#ifdef USE_HE_GPU
#include "HE/unified/PhantomWrapper.h"
#include "HE/unified/UnifiedCiphertext.h"
#include "HE/unified/UnifiedPlaintext.h"
#endif

namespace HE {
namespace unified {

#ifndef USE_HE_GPU

class UnifiedEvaluator : public seal::Evaluator {
public:
  // Explicitly inherit all constructors from the Base class
  using seal::Evaluator::Evaluator;

  inline LOCATION backend() const { return LOCATION::HOST; }
};

#else

class UnifiedEvaluator {
public:
  explicit UnifiedEvaluator(const UnifiedContext &context)
      : backend_(context.backend()) {
    switch (backend_) {
    case LOCATION::HOST:
      register_evaluator(context.hcontext());
      break;
    case LOCATION::DEVICE:
      register_evaluator(context.dcontext());
      break;
    default:
      throw std::invalid_argument("Invalid backend");
    }
  }

  explicit UnifiedEvaluator(const seal::SEALContext &context)
      : backend_(LOCATION::HOST) {
    register_evaluator(context);
  }

  explicit UnifiedEvaluator(const PhantomContext &context)
      : backend_(LOCATION::DEVICE) {
    register_evaluator(context);
  }

  inline LOCATION backend() const { return backend_; }

  template <typename context_t>
  void register_evaluator(const context_t &context) {
    if constexpr (std::is_same_v<context_t, seal::SEALContext>) {
      seal_eval_ = std::make_unique<seal::Evaluator>(context);
    } else if constexpr (std::is_same_v<context_t, PhantomEvaluator>) {
      phantom_eval_ = std::make_unique<PhantomEvaluator>(context);
    }
  }

  void activate_backend(LOCATION backend) {
    switch (backend) {
    case LOCATION::HOST:
      if (seal_eval_ == nullptr) {
        throw std::runtime_error(
            "Evaluator on the HOST side is not registered");
      }
      break;
    case LOCATION::DEVICE:
      if (seal_eval_ == nullptr) {
        throw std::runtime_error(
            "Evaluator on the HOST side is not registered");
      }
      break;
    default:
      throw std::invalid_argument("Invalid backend");
    }
    backend_ = backend;
  }

  void negate_inplace(UnifiedCiphertext &encrypted) const;

  inline void negate(const UnifiedCiphertext &encrypted,
                     UnifiedCiphertext &destination) const {
    destination = encrypted;
    negate_inplace(destination);
  }

  void add_inplace(UnifiedCiphertext &encrypted1,
                   const UnifiedCiphertext &encrypted2) const;

  inline void add(const UnifiedCiphertext &encrypted1,
                  const UnifiedCiphertext &encrypted2,
                  UnifiedCiphertext &destination) const {
    destination = encrypted1;
    add_inplace(destination, encrypted2);
  }

  void sub_inplace(UnifiedCiphertext &encrypted1,
                   const UnifiedCiphertext &encrypted2) const;

  inline void sub(const UnifiedCiphertext &encrypted1,
                  const UnifiedCiphertext &encrypted2,
                  UnifiedCiphertext &destination) const {
    destination = encrypted1;
    sub_inplace(destination, encrypted2);
  }

  void multiply_inplace(UnifiedCiphertext &encrypted1,
                        const UnifiedCiphertext &encrypted2) const;

  inline void multiply(const UnifiedCiphertext &encrypted1,
                       const UnifiedCiphertext &encrypted2,
                       UnifiedCiphertext &destination) const {
    destination = encrypted1;
    multiply_inplace(destination, encrypted2);
  }

  inline void square_inplace(UnifiedCiphertext &encrypted) const {
    multiply_inplace(encrypted, encrypted);
  }

  inline void square(const UnifiedCiphertext &encrypted,
                     UnifiedCiphertext &destination) {
    destination = encrypted;
    multiply_inplace(destination, encrypted);
  }

  template <typename RelinKey_t>
  void relinearize_inplace(UnifiedCiphertext &encrypted,
                           const RelinKey_t &relin_keys) const;

  template <typename RelinKey_t>
  inline void relinearize(const UnifiedCiphertext &encrypted,
                          const RelinKey_t &relin_keys,
                          UnifiedCiphertext &destination) const {
    destination = encrypted;
    relinearize_inplace(destination, relin_keys);
  }

  void mod_switch_to_next(const UnifiedCiphertext &encrypted,
                          UnifiedCiphertext &destination) const;

  inline void mod_switch_to_next_inplace(UnifiedCiphertext &encrypted) const {
    UnifiedCiphertext destination;
    mod_switch_to_next(encrypted, destination);
    encrypted = std::move(destination);
  }

  void rescale_to_next(const UnifiedCiphertext &encrypted,
                       UnifiedCiphertext &destination) const;

  inline void rescale_to_next_inplace(UnifiedCiphertext &encrypted) const {
    UnifiedCiphertext destination;
    rescale_to_next(encrypted, destination);
    encrypted = std::move(destination);
  }

  void add_plain_inplace(UnifiedCiphertext &encrypted,
                         const UnifiedPlaintext &plain) const;

  inline void add_plain(const UnifiedCiphertext &encrypted,
                        const UnifiedPlaintext &plain,
                        UnifiedCiphertext &destination) const {
    destination = encrypted;
    add_plain_inplace(destination, plain);
  }

  void sub_plain_inplace(UnifiedCiphertext &encrypted,
                         const UnifiedPlaintext &plain) const;

  inline void sub_plain(const UnifiedCiphertext &encrypted,
                        const UnifiedPlaintext &plain,
                        UnifiedCiphertext &destination) const {
    destination = encrypted;
    sub_plain_inplace(destination, plain);
  }

  void multiply_plain_inplace(UnifiedCiphertext &encrypted,
                              const UnifiedPlaintext &plain) const;

  inline void multiply_plain(const UnifiedCiphertext &encrypted,
                             const UnifiedPlaintext &plain,
                             UnifiedCiphertext &destination) const {
    destination = encrypted;
    multiply_plain_inplace(destination, plain);
  }

  void rotate_vector_inplace(UnifiedCiphertext &encrypted, int step,
                             const UnifiedGaloisKeys &galois_key) const;

  inline void rotate_vector(const UnifiedCiphertext &encrypted, int step,
                            const UnifiedGaloisKeys &galois_key,
                            UnifiedCiphertext &destination) const {
    destination = encrypted;
    rotate_vector_inplace(destination, step, galois_key);
  }

  void complex_conjugate_inplace(UnifiedCiphertext &encrypted,
                                 const UnifiedGaloisKeys &galois_key) const;

  inline void complex_conjugate(const UnifiedCiphertext &encrypted,
                                const UnifiedGaloisKeys &galois_key,
                                UnifiedCiphertext &destination) const {
    destination = encrypted;
    complex_conjugate_inplace(destination, galois_key);
  }

  inline void rotate_rows_inplace(UnifiedCiphertext &encrypted, int step,
                                  const UnifiedGaloisKeys &galois_key) const {
    rotate_vector_inplace(encrypted, step, galois_key);
  }

  inline void rotate_rows(const UnifiedCiphertext &encrypted, int step,
                          const UnifiedGaloisKeys &galois_key,
                          UnifiedCiphertext &destination) const {
    destination = encrypted;
    rotate_rows_inplace(destination, step, galois_key);
  }

  inline void
  rotate_columns_inplace(UnifiedCiphertext &encrypted,
                         const UnifiedGaloisKeys &galois_key) const {
    complex_conjugate_inplace(encrypted, galois_key);
  }

  inline void rotate_columns(const UnifiedCiphertext &encrypted,
                             const UnifiedGaloisKeys &galois_key,
                             UnifiedCiphertext &destination) const {
    destination = encrypted;
    rotate_columns_inplace(destination, galois_key);
  }

private:
  LOCATION backend_;
  std::unique_ptr<seal::Evaluator> seal_eval_;
  std::unique_ptr<PhantomEvaluator> phantom_eval_;
};
#endif

} // namespace unified
} // namespace HE
