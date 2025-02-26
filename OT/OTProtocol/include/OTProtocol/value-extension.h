#include "aux-protocols.h"
#include "millionaire.h"
#pragma once
using namespace OTPrimitive;
using namespace Utils;
namespace OTProtocol {
class XTProtocol {
public:
  NetIO *io = nullptr;
  OTPack<NetIO> *otpack;
  TripleGenerator<NetIO> *triple_gen = nullptr;
  MillionaireProtocol<NetIO> *millionaire = nullptr;
  AuxProtocols *aux = nullptr;
  bool del_aux = false;
  int party;

  // Constructor
  XTProtocol(int party, NetIO *io, OTPack<NetIO> *otpack,
             AuxProtocols *auxp = nullptr);

  // Destructor
  ~XTProtocol();

  void z_extend(int32_t dim, uint64_t *inA, uint64_t *outB, int32_t bwA,
                int32_t bwB, uint8_t *msbA = nullptr);

  void s_extend(int32_t dim, uint64_t *inA, uint64_t *outB, int32_t bwA,
                int32_t bwB, uint8_t *msbA = nullptr);
};

} // namespace OTProtocol
