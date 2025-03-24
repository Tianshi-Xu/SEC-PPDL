#pragma once

#include <Datatype/UnifiedType.h>
#include <seal/context.h>

#ifdef USE_HE_GPU
#include <phantom/context.cuh>
#endif

using namespace Datatype;

namespace HE {
namespace unified {

class UnifiedContext {
public:
  UnifiedContext(uint64_t poly_modulus_degree, int bit_size, bool batch = true,
                 LOCATION backend = HOST)
      : is_gpu_enable_(backend == DEVICE) {
#ifndef USE_HE_GPU
    if (backend != LOCATION::HOST) {
      throw std::invalid_argument("Non GPU version");
    }
#else
    if (backend != HOST && backend != DEVICE) {
      throw std::invalid_argument("UnifiedContext: Invalid backend");
    }
#endif
    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(
        seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(
        batch ? seal::PlainModulus::Batching(poly_modulus_degree, bit_size)
              : 1 << bit_size);
    seal_context_ = std::make_unique<seal::SEALContext>(parms);

#ifdef USE_HE_GPU
    if (backend == LOCATION::DEVICE) {
      phantom::EncryptionParameters parms(phantom::scheme_type::bfv);
      parms.set_poly_modulus_degree(poly_modulus_degree);
      parms.set_coeff_modulus(
          phantom::arith::CoeffModulus::BFVDefault(poly_modulus_degree));
      parms.set_plain_modulus(batch ? phantom::arith::PlainModulus::Batching(
                                          poly_modulus_degree, bit_size)
                                    : 1 << bit_size);
      phantom_context_ = std::make_unique<PhantomContext>(parms);
    }
#endif
  }

  ~UnifiedContext() = default;

  inline bool is_gpu_enable() const { return is_gpu_enable_; }

  inline const seal::SEALContext &hcontext() const { return *seal_context_; }

  operator const seal::SEALContext &() const { return hcontext(); };

#ifdef USE_HE_GPU
  inline const PhantomContext &dcontext() const { return *phantom_context_; }

  operator const PhantomContext &() const { return dcontext(); };
#endif

private:
  bool is_gpu_enable_ = false;
  std::unique_ptr<seal::SEALContext> seal_context_ = nullptr;
#ifdef USE_HE_GPU
  std::unique_ptr<PhantomContext> phantom_context_ = nullptr;
#endif
};

} // namespace unified
} // namespace HE