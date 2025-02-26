#include <OTProtocol/protocol.h>
#include <HE/HE.h>
#include <NonlinearLayer/ReLU.h>
#include <NonlinearOperator/Truncation.h>

namespace Model{
template <typename T, typename IO=Utils::NetIO>
class CryptoPrimitive{
    public:
        HE::HEEvaluator* HE;
        NonlinearLayer::ReLU<IO, T>* relu;
        NonlinearOperator::Truncation<T>* truncation;
        int32_t num_threads;
        int party;
        Datatype::CONV_TYPE conv_type;
        CryptoPrimitive(int party, HE::HEEvaluator* HE, Datatype::CONV_TYPE conv_type, NonlinearLayer::ReLU<IO, T>* relu, NonlinearOperator::Truncation<T>* truncation, int32_t num_threads){
            this->HE = HE;
            this->relu = relu;
            this->truncation = truncation;
            this->num_threads = num_threads;
            this->party = party;
        }

        CryptoPrimitive(int party, HE::HEEvaluator* HE, int32_t num_threads, Datatype::OT_TYPE ot_type){
            this->party = party;
            this->HE = HE;
            this->num_threads = num_threads;
            this->ioArr = new Utils::NetIO*[num_threads];
            this->otpackArr = new OTPrimitive::OTPack<Utils::NetIO>*[num_threads];
            this->reluprotocol = new NonlinearLayer::ReLUProtocol<Utils::NetIO, T>*[num_threads];
            this->truncationProtocol = new NonlinearOperator::TruncationProtocol*[num_threads];
            for (int i = 0; i < num_threads; i++) {
                this->ioArr[i] = new Utils::NetIO(party == Utils::ALICE ? nullptr : address.c_str(), port + i);
                // TODO: change to VOLE OT
                if (i & 1) {
                    this->otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], 3 - party);
                } else {
                    this->otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], party);
                }
                this->reluprotocol[i] = new ReLURingProtocol<IO, T>(party, ioArr[i], 4, MILL_PARAM, otpackArr[i], ot_type);
                this->truncationProtocol[i] = new TruncationProtocol(party, ioArr[i], otpackArr[i]);
            }
            this->relu = new NonlinearLayer::ReLU<IO, T>(party, HE, num_threads, ot_type);
            this->truncation = new NonlinearOperator::Truncation<T>(party, HE, num_threads, ot_type);
        }
    private:
        Utils::NetIO **ioArr;
        OTPrimitive::OTPack<Utils::NetIO> **otpackArr;
        NonlinearLayer::ReLUProtocol<Utils::NetIO, T> **reluprotocol;
        NonlinearOperator::TruncationProtocol **truncationProtocol;
};
}