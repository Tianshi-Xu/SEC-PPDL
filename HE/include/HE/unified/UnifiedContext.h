#pragma once

#include "HE/unified/Define.h"
#include <seal/context.h>

#ifdef USE_HE_GPU
#include <context.cuh>
#endif

namespace HE {
namespace unified {

class UnifiedContext {
public:
  UnifiedContext(uint64_t poly_modulus_degree, int bit_size,
                 LOCATION backend = LOCATION::HOST)
      : backend_(backend) {
#ifndef USE_HE_GPU
    if (backend != LOCATION::HOST) {
      throw std::invalid_argument("Non GPU version");
    }
#else
    if (backend != LOCATION::HOST || backend != LOCATION::DEVICE) {
      throw std::invalid_argument("Invalid backend");
    }
#endif
    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(
        seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(
        seal::PlainModulus::Batching(poly_modulus_degree, bit_size));
    seal_context_ = std::make_unique<seal::SEALContext>(parms);

#ifdef USE_HE_GPU
    if (backend == LOCATION::DEVICE) {
      phantom::EncryptionParameters parms(phantom::scheme_type::bfv);
      parms.set_coeff_modulus(
          phantom::arith::CoeffModulus::BFVDefault(poly_modulus_degree));
      parms.set_plain_modulus(phantom::arith::PlainModulus::Batching(
          poly_modulus_degree, bit_size));
      phantom_context_ = std::make_unique<PhantomContext>(parms);
    }
#endif
  }

  inline LOCATION backend() const { return backend_; }

  inline const seal::SEALContext &hcontext() const { return *seal_context_; }

  operator const seal::SEALContext &() const { return hcontext(); };

#ifdef USE_HE_GPU
  inline const PhantomContext &dcontext() const { return *phantom_context_; }

  operator const PhantomContext &() const { return dcontext(); };
#endif

private:
  LOCATION backend_;
  std::unique_ptr<seal::SEALContext> seal_context_;

#ifdef USE_HE_GPU
  std::unique_ptr<PhantomContext> phantom_context_;
#endif
};

} // namespace unified
} // namespace HE