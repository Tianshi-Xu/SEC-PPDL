#pragma once
#include "emp_ot.h"
#include "ot_pack.h"
#include <Utils/emp-tool.h>
using namespace OTPrimitive;
using namespace std;

namespace OTPrimitive {
template <typename T>
class IKNPOTPack : public OTPack<T> {

 public:
  IKNPOTPack(T *io, int party, bool do_setup = true) : OTPack<T>(io, party, do_setup) {
  IKNPOTPack(T *io, int party, bool do_setup = true): OTPack<T>(io, party, do_setup) {
    std::cout << "using kkot pack" << std::endl;
    this->party = party;
    this->do_setup = do_setup;
    this->io = io;

    for (int i = 0; i < KKOT_TYPES; i++) {
      this->kkot[i] = new SplitKKOT<NetIO>(this->party, io, 1 << (i + 1));
      // cout << this->kkot[i]->te << endl;
    }
cd
    this->iknp_straight = new SplitIKNP<NetIO>(this->party, io);
    this->iknp_reversed = new SplitIKNP<NetIO>(3 - this->party, io);
    this->do_setup = false;
    if (do_setup) {
      SetupBaseOTs();
    }
  }

  ~IKNPOTPack() {
    for (int i = 0; i < KKOT_TYPES; i++) delete this->kkot[i];
    delete this->iknp_straight;
    delete this->iknp_reversed;
  }

  void SetupBaseOTs() {
    switch (this->party) {
      case 1:
        this->kkot[0]->setup_send();
        this->iknp_straight->setup_send();
        this->iknp_reversed->setup_recv();
        for (int i = 1; i < KKOT_TYPES; i++) {
          this->kkot[i]->setup_send();
        }
        break;
      case 2:
        this->kkot[0]->setup_recv();
        this->iknp_straight->setup_recv();
        this->iknp_reversed->setup_send();
        for (int i = 1; i < KKOT_TYPES; i++) {
          this->kkot[i]->setup_recv();
        }
        break;
    }
    this->io->flush();
  }

  /*
   * DISCLAIMER:
   * OTPack copy method avoids computing setup keys for each OT instance by
   * reusing the keys generated (through base OTs) for another OT instance.
   * Ideally, the PRGs within OT instances, using the same keys, should use
   * mutually exclusive counters for security. However, the current
   * implementation does not support this.
   */

  void copy(OTPack<T> *copy_from) {
    assert(this->do_setup == false && copy_from->do_setup == true);
    SplitKKOT<T> *kkot_base = copy_from->kkot[0];
    SplitIKNP<T> *iknp_s_base = copy_from->iknp_straight;
    SplitIKNP<T> *iknp_r_base = copy_from->iknp_reversed;

    switch (this->party) {
      case 1:
        for (int i = 0; i < KKOT_TYPES; i++) {
          this->kkot[i]->setup_send(kkot_base->k0, kkot_base->s);
        }
        this->iknp_straight->setup_send(iknp_s_base->k0, iknp_s_base->s);
        this->iknp_reversed->setup_recv(iknp_r_base->k0, iknp_r_base->k1);
        break;
      case 2:
        for (int i = 0; i < KKOT_TYPES; i++) {
          this->kkot[i]->setup_recv(kkot_base->k0, kkot_base->k1);
        }
        this->iknp_straight->setup_recv(iknp_s_base->k0, iknp_s_base->k1);
        this->iknp_reversed->setup_send(iknp_r_base->k0, iknp_r_base->s);
        break;
    }
    this->do_setup = true;
    return;
  }
};

}  // namespace OTPrimitive
