#include <iostream>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#pragma once
namespace HE {

class HEIO {
private:
    int sockfd;  // 套接字文件描述符
    struct sockaddr_in address;
    bool is_server;
    int counter = 0;

public:
    HEIO(const char* ip, int port, bool server = false);
    ~HEIO();

    void send_data(const void* data, int nbyte);
    void recv_data(void* data, int nbyte);
};

} // namespace HE
