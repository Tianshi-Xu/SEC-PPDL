#pragma once

#include <seal/evaluator.h>
#include "HE/unified/UnifiedCiphertext.h"
#include "HE/unified/UnifiedEvk.h"
#include "HE/unified/UnifiedPlaintext.h"

#ifdef USE_HE_GPU
#include "HE/unified/PhantomWrapper.h"
#else
#include "Datatype/UnifiedType.h"
#endif

namespace HE {
namespace unified {

#ifndef USE_HE_GPU

class UnifiedEvaluator : public seal::Evaluator {
public:
  // Explicitly inherit all constructors from the Base class
  using seal::Evaluator::Evaluator;

  inline Datatype::LOCATION backend() const { return Datatype::LOCATION::HOST; }
};

#else

class UnifiedEvaluator {
public:
  explicit UnifiedEvaluator(const UnifiedContext &context) {
    register_evaluator(context.hcontext());
    if (context.is_gpu_enable()) {
      register_evaluator(context.dcontext());
    }
  }

  ~UnifiedEvaluator() = default;

  template <typename context_t>
  void register_evaluator(const context_t &context) {
    if constexpr (std::is_same_v<context_t, seal::SEALContext>) {
      seal_eval_ = std::make_unique<seal::Evaluator>(context);
    } else if constexpr (std::is_same_v<context_t, PhantomContext>) {
      phantom_eval_ = std::make_unique<PhantomEvaluator>(context);
    } else {
      static_assert(std::is_same_v<context_t, void>,
                    "UnifiedEvaluator: Invalid context");
    }
  }

  template <typename unified_data_t>
  inline void backend_check(const unified_data_t &data) const {
    if (data.on_device() && !phantom_eval_) {
      throw std::invalid_argument("Unregistered GPU backend");
    }
  }

  template <typename T1, typename T2>
  inline void backend_check(const T1 &x, const T2 &y) const {
    if (x.on_device() && y.on_device() && !phantom_eval_) {
      throw std::invalid_argument("Unregistered GPU backend");
    }
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

  void relinearize_inplace(UnifiedCiphertext &encrypted,
                           const UnifiedRelinKeys &relin_keys) const;

  inline void relinearize(const UnifiedCiphertext &encrypted,
                          const UnifiedRelinKeys &relin_keys,
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

  void rotate_rows_inplace(UnifiedCiphertext &encrypted, int step,
                           const UnifiedGaloisKeys &galois_key) const;

  inline void rotate_rows(const UnifiedCiphertext &encrypted, int step,
                          const UnifiedGaloisKeys &galois_key,
                          UnifiedCiphertext &destination) const {
    destination = encrypted;
    rotate_rows_inplace(destination, step, galois_key);
  }

  void rotate_columns_inplace(UnifiedCiphertext &encrypted,
                              const UnifiedGaloisKeys &galois_key) const;

  inline void rotate_columns(const UnifiedCiphertext &encrypted,
                             const UnifiedGaloisKeys &galois_key,
                             UnifiedCiphertext &destination) const {
    destination = encrypted;
    rotate_columns_inplace(destination, galois_key);
  }

private:
  std::unique_ptr<seal::Evaluator> seal_eval_ = nullptr;
  std::unique_ptr<PhantomEvaluator> phantom_eval_ = nullptr;
};
#endif

} // namespace unified
} // namespace HE
