#include <HE/HE.h>
#include <Utils/ArgMapping/ArgMapping.h>
using namespace std;
using namespace HE;

int party, port = 32000;
int num_threads = 2;
string address = "127.0.0.1";

int main(int argc, char* argv[]) {
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2"); // 1 is server, 2 is client
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.parse(argc, argv);
    
    Utils::NetIO* netio = new Utils::NetIO(party == ALICE ? nullptr : address.c_str(), port);
    std::cout << "netio generated" << std::endl;
    HE::HEEvaluator HE(netio, party, 8192,60,Datatype::HOST);
    HE.GenerateNewKey();
    return 0;
}