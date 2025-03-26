#include <LinearOperator/Polynomial.h>
#include <Utils/ArgMapping/ArgMapping.h>

int party, port = 32000;
int num_threads = 2;
string address = "127.0.0.1";

Utils::NetIO* netio;
HE::HEEvaluator* he;
using namespace std;
using namespace LinearOperator;
using namespace HE;

void test_poly(HE::HEEvaluator* he){
    Tensor<uint64_t> x({8192});
    Tensor<uint64_t> y({8192});
    Tensor<uint64_t> z({8192});
    if(party == ALICE){
        for(uint32_t i = 0; i < 8192; i++){
            x(i) = i;
            y(i) = i;
        }
    }
    x.print(10);
    y.print(10);
    z = LinearOperator::ElementWiseMul(x, x, he);
    if (party == ALICE){
        netio->send_tensor(z);
    }else{
        Tensor<uint64_t> z0({8192});
        netio->recv_tensor(z0);
        auto z_result = z + z0;
        for(uint32_t i = 0; i < 8192; i++){
            z_result(i) = z_result(i) % he->plain_mod;
        }
        z_result.print(10);
    }
}

int main(int argc, char **argv){
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2"); // 1 is server, 2 is client
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.parse(argc, argv);
    
    netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
    std::cout << "netio generated" << std::endl;
    he = new HE::HEEvaluator(netio, party, 8192,32,Datatype::HOST);
    he->GenerateNewKey();
    
    test_poly(he);

    return 0;
}