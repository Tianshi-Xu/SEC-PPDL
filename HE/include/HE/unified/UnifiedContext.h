#pragma once

#include <Datatype/UnifiedType.h>
#include <seal/context.h>

#ifdef USE_HE_GPU
#include <phantom/context.cuh>
#endif

using namespace Datatype;
using namespace std;

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

  UnifiedContext(const seal::EncryptionParameters &parms,
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
    seal_context_ = std::make_unique<seal::SEALContext>(parms);
#ifdef USE_HE_GPU
    if (backend == LOCATION::DEVICE) {
      phantom_context_ = std::make_unique<PhantomContext>(get_parms(parms));
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
#ifdef USE_HE_GPU
  inline static phantom::EncryptionParameters
  get_parms(const seal::EncryptionParameters &parms) {

    auto get_scheme_type = [](seal::scheme_type type) {
      switch (type) {
      case seal::scheme_type::bfv:
        return phantom::scheme_type::bfv;
      case seal::scheme_type::ckks:
        return phantom::scheme_type::ckks;
      default:
        throw std::invalid_argument("Not support");
      }
    };

    auto convert_seal_to_phantom_modulus =
        [](const seal::Modulus &seal_modulus) {
          phantom::arith::Modulus phantom_modulus(seal_modulus.value());
          return phantom::arith::Modulus(seal_modulus.value());
        };

    phantom::EncryptionParameters target_parms(get_scheme_type(parms.scheme()));
    std::vector<phantom::arith::Modulus> modulus(parms.coeff_modulus().size());
    for (size_t i = 0; i < modulus.size(); i++) {
      modulus[i] = convert_seal_to_phantom_modulus(parms.coeff_modulus()[i]);
    }
    target_parms.set_poly_modulus_degree(parms.poly_modulus_degree());
    target_parms.set_coeff_modulus(modulus);
    target_parms.set_plain_modulus(
        convert_seal_to_phantom_modulus(parms.plain_modulus()));
    return target_parms;
  }
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