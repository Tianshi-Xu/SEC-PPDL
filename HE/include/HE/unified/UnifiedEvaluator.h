#pragma once

#include "HE/unified/Define.h"
#include "HE/unified/PhantomWrapper.h"
#include "HE/unified/UnifiedCiphertext.h"
#include "HE/unified/UnifiedPlaintext.h"
#include <context.cuh>
#include <seal/evaluator.h>

#ifndef USE_HE_GPU

using UnifiedEvaluator = seal::Evaluator;

#else
namespace HE {
namespace unified {

class UnifiedEvaluator {
public:
  explicit UnifiedEvaluator(const seal::SEALContext &context)
      : backend_(LOCATION::HOST) {
    register_evaluator(context);
  }

  explicit UnifiedEvaluator(const PhantomContext &context)
      : backend_(LOCATION::DEVICE) {
    register_evaluator(context);
  }

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

private:
  LOCATION backend_;
  std::unique_ptr<seal::Evaluator> seal_eval_;
  std::unique_ptr<PhantomEvaluator> phantom_eval_;
};

} // namespace unified
} // namespace HE

#endif