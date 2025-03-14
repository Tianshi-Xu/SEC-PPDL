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

        CryptoPrimitive(int party, int32_t num_threads, int32_t bit_length, Datatype::OT_TYPE ot_type, int32_t polyModulusDegree, int32_t plainWidth, Datatype::CONV_TYPE conv_type,Datatype::LOCATION backend, string address, int port){
            this->party = party;
            this->conv_type = conv_type;
            this->num_threads = num_threads;
            this->ioArr = new IO*[num_threads];
            this->otpackArr = new OTPrimitive::OTPack<IO>*[num_threads];
            this->reluprotocol = new NonlinearLayer::ReLUProtocol<T, IO>*[num_threads];
            this->truncationProtocol = new OTProtocol::TruncationProtocol*[num_threads];
            for (int i = 0; i < num_threads; i++) {
                std::cout << "before, i = " << i << std::endl;
                this->ioArr[i] = new IO(party == ALICE ? nullptr : address.c_str(), port + i + 1);
                std::cout << "i = " << i << std::endl;
                // TODO: change to VOLE OT
                if (ot_type == Datatype::VOLE) {
                    this->otpackArr[i] = new VOLEOTPack<Utils::NetIO>(this->ioArr[i], party);
                } else {
                    this->otpackArr[i] = new IKNPOTPack<Utils::NetIO>(this->ioArr[i], party);
                }
                this->reluprotocol[i] = new NonlinearLayer::ReLURingProtocol<T, IO>(party, this->ioArr[i], bit_length, MILL_PARAM, this->otpackArr[i], ot_type);
                this->truncationProtocol[i] = new TruncationProtocol(party, this->ioArr[i], this->otpackArr[i]);
            }
            this->relu = new NonlinearLayer::ReLU<T, IO>(reluprotocol, bit_length, num_threads);
            this->truncation = new NonlinearOperator::Truncation<T>(truncationProtocol, num_threads);
            cout << "begin to generate HEIO" << endl;
            this->io = ioArr[0];
            this->HE = new HE::HEEvaluator(io, party, polyModulusDegree, plainWidth, backend);
            this->HE->GenerateNewKey();
            cout << "CryptoPrimitive constructor finished" << endl;
        }
    private:
        IO *io;
        IO **ioArr;
        OTPrimitive::OTPack<IO> **otpackArr;
        NonlinearLayer::ReLUProtocol<T, IO> **reluprotocol;
        OTProtocol::TruncationProtocol **truncationProtocol;
};

}