//#include "NetIO.h"
#include "HE.h"
#include <cstring>
#include "seal/seal.h"
#include <seal/secretkey.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/rlwe.h>
#include <seal/secretkey.h>
#include <seal/serializable.h>
using namespace seal;
using namespace seal::util;
using namespace std;



int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <server/client> <IP> <port>\n";
        return 1;
    }

    bool is_server = std::string(argv[1]) == "server";
    const char* ip = argv[2];
    int port = std::stoi(argv[3]);
    NetIO netio(ip, port, is_server);
    std::cout << "start test" << std::endl;
    HEEvaluator HE(netio, is_server);
    std::cout << "build";
    HE.GenerateNewKey(8192);
    std::cout << "gen";
    return 0;
}