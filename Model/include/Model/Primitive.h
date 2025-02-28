#include <OTProtocol/protocol.h>
#include <HE/HE.h>
#include <NonlinearLayer/ReLU.h>
#include <NonlinearOperator/Truncation.h>
using namespace NonlinearLayer;
namespace Model{
template <typename T, typename IO=Utils::NetIO>
class CryptoPrimitive{
    public:
        HE::HEEvaluator* HE;
        NonlinearLayer::ReLU<T, IO>* relu;
        NonlinearOperator::Truncation<T>* truncation;
        int32_t num_threads;
        int party;
        Datatype::CONV_TYPE conv_type = Datatype::CONV_TYPE::Nest;
        CryptoPrimitive(int party, HE::HEEvaluator* HE, Datatype::CONV_TYPE conv_type, NonlinearLayer::ReLU<T, IO>* relu, NonlinearOperator::Truncation<T>* truncation, int32_t num_threads){
            this->HE = HE;
            this->relu = relu;
            this->truncation = truncation;
            this->num_threads = num_threads;
            this->party = party;
            this->conv_type = conv_type;
        }

        CryptoPrimitive(int party, HE::HEEvaluator* HE, int32_t num_threads, int32_t bit_length, Datatype::OT_TYPE ot_type, Datatype::CONV_TYPE conv_type, string address, int port){
            this->party = party;
            this->conv_type = conv_type;
            this->HE = HE;
            this->num_threads = num_threads;
            this->ioArr = new Utils::NetIO*[num_threads];
            this->otpackArr = new OTPrimitive::OTPack<Utils::NetIO>*[num_threads];
            this->reluprotocol = new NonlinearLayer::ReLUProtocol<T, IO>*[num_threads];
            this->truncationProtocol = new OTProtocol::TruncationProtocol*[num_threads];
            for (int i = 0; i < num_threads; i++) {
                if(i==0){
                    this->ioArr[i] = HE->IO;
                }else{
                    this->ioArr[i] = new Utils::NetIO(party == Utils::ALICE ? nullptr : address.c_str(), port + i);
                }
                // TODO: change to VOLE OT
                if (i & 1) {
                    this->otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], 3 - party);
                } else {
                    this->otpackArr[i] = new IKNPOTPack<Utils::NetIO>(ioArr[i], party);
                }
                this->reluprotocol[i] = new NonlinearLayer::ReLURingProtocol<T, IO>(party, ioArr[i], bit_length, MILL_PARAM, otpackArr[i], ot_type);
                this->truncationProtocol[i] = new TruncationProtocol(party, ioArr[i], otpackArr[i]);
            }
            this->relu = new NonlinearLayer::ReLU<T, IO>(reluprotocol, bit_length, num_threads);
            this->truncation = new NonlinearOperator::Truncation<T>(truncationProtocol, num_threads);
        }
    private:
        IO **ioArr;
        OTPrimitive::OTPack<IO> **otpackArr;
        NonlinearLayer::ReLUProtocol<T, IO> **reluprotocol;
        OTProtocol::TruncationProtocol **truncationProtocol;
};
}