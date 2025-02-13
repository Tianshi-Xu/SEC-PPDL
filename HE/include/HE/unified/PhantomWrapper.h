#pragma once

#include <phantom/batchencoder.h>
#include <phantom/ciphertext.h>
#include <phantom/context.cuh>
#include <phantom/plaintext.h>
#include <phantom/secretkey.h>

namespace HE {

// ************************ PhantomEvaluator ***********************

class PhantomEvaluator {

public:
  PhantomEvaluator() = delete;

  PhantomEvaluator(const PhantomContext &context) : context_(context) {}

  ~PhantomEvaluator() = default;

  // encrypted = -encrypted
  void negate_inplace(PhantomCiphertext &encrypted) const;

  inline void negate(const PhantomCiphertext &encrypted,
                     PhantomCiphertext &destination) const {
    destination = encrypted;
    negate_inplace(destination);
  }

  // encrypted1 += encrypted2
  void add_inplace(PhantomCiphertext &encrypted1,
                   const PhantomCiphertext &encrypted2) const;

  inline void add(const PhantomCiphertext &encrypted1,
                  const PhantomCiphertext &encrypted2,
                  PhantomCiphertext &destination) const {
    destination = encrypted1;
    add_inplace(destination, encrypted2);
  }

  // if negate = false (default): encrypted1 -= encrypted2
  // if negate = true: encrypted1 = encrypted2 - encrypted1
  void sub_inplace(PhantomCiphertext &encrypted1,
                   const PhantomCiphertext &encrypted2,
                   bool negate = false) const;

  inline void sub(const PhantomCiphertext &encrypted1,
                  const PhantomCiphertext &encrypted2,
                  PhantomCiphertext &destination, bool negate = false) const {
    destination = encrypted1;
    sub_inplace(destination, encrypted2, negate);
  }

  // encrypted += plain
  void add_plain_inplace(PhantomCiphertext &encrypted,
                         const PhantomPlaintext &plain) const;

  inline void add_plain(const PhantomCiphertext &encrypted,
                        const PhantomPlaintext &plain,
                        PhantomCiphertext &destination) const {
    destination = encrypted;
    add_plain_inplace(destination, plain);
  }

  // encrypted -= plain
  void sub_plain_inplace(PhantomCiphertext &encrypted,
                         const PhantomPlaintext &plain) const;

  inline void sub_plain(const PhantomCiphertext &encrypted,
                        const PhantomPlaintext &plain,
                        PhantomCiphertext &destination) const {
    destination = encrypted;
    sub_plain_inplace(destination, plain);
  }

  // encrypted *= plain
  void multiply_plain_inplace(PhantomCiphertext &encrypted,
                              const PhantomPlaintext &plain) const;

  inline void multiply_plain(const PhantomCiphertext &encrypted,
                             const PhantomPlaintext &plain,
                             PhantomCiphertext &destination) const {
    destination = encrypted;
    multiply_plain_inplace(destination, plain);
  }

  // encrypted1 *= encrypted2
  void multiply_inplace(PhantomCiphertext &encrypted1,
                        const PhantomCiphertext &encrypted2) const;

  inline void multiply(const PhantomCiphertext &encrypted1,
                       const PhantomCiphertext &encrypted2,
                       PhantomCiphertext &destination) const {
    destination = encrypted1;
    multiply_inplace(destination, encrypted2);
  }

  inline void square_inplace(PhantomCiphertext &encrypted) const {
    multiply_inplace(encrypted, encrypted);
  }

  inline void square(const PhantomCiphertext &encrypted,
                     PhantomCiphertext &destination) {
    destination = encrypted;
    multiply_inplace(destination, encrypted);
  }

  void relinearize_inplace(PhantomCiphertext &encrypted,
                           const PhantomRelinKey &relin_keys) const;

  inline void relinearize(const PhantomCiphertext &encrypted,
                          const PhantomRelinKey &relin_keys,
                          PhantomCiphertext &destination) const {
    destination = encrypted;
    relinearize_inplace(destination, relin_keys);
  }

  void rescale_to_next(const PhantomCiphertext &encrypted,
                       PhantomCiphertext &destination) const;

  inline void rescale_to_next_inplace(PhantomCiphertext &encrypted) const {
    PhantomCiphertext destination;
    rescale_to_next(encrypted, destination);
    encrypted = std::move(destination);
  }

  void mod_switch_to_next(const PhantomCiphertext &encrypted,
                          PhantomCiphertext &destination) const;

  inline void mod_switch_to_next_inplace(PhantomCiphertext &encrypted) const {
    PhantomCiphertext destination;
    mod_switch_to_next(encrypted, destination);
    encrypted = std::move(destination);
  }

  void rotate_vector_inplace(PhantomCiphertext &encrypted, int step,
                             const PhantomGaloisKey &galois_key) const;

  inline void rotate_vector(const PhantomCiphertext &encrypted, int step,
                            const PhantomGaloisKey &galois_key,
                            PhantomCiphertext &destination) const {
    destination = encrypted;
    rotate_vector_inplace(destination, step, galois_key);
  }

  void complex_conjugate_inplace(PhantomCiphertext &encrypted,
                                 const PhantomGaloisKey &galois_key) const;

  inline void complex_conjugate(const PhantomCiphertext &encrypted,
                                const PhantomGaloisKey &galois_key,
                                PhantomCiphertext &destination) const {
    destination = encrypted;
    complex_conjugate_inplace(destination, galois_key);
  }

private:
  const PhantomContext &context_;
};

// ********************** PhantomBatchEncoder **********************

// For BFV/BGV
class PhantomIntegerEncoder : PhantomBatchEncoder {

public:
  PhantomIntegerEncoder(const PhantomContext &context)
      : PhantomBatchEncoder(context), context_(context) {}

  inline void encode(const std::vector<uint64_t> &values_matrix,
                     PhantomPlaintext &destination) const {
    PhantomBatchEncoder::encode(context_, values_matrix, destination);
  }

  inline void decode(const PhantomPlaintext &plain,
                     std::vector<uint64_t> &destination) const {
    PhantomBatchEncoder::decode(context_, plain, destination);
  }

private:
  const PhantomContext &context_;
};

// *********************** PhantomCKKSEncoder **********************

// TODO:

} // namespace HE