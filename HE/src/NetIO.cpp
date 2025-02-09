#include <HE/NetIO.h>
using namespace HE;

NetIO::NetIO(const char* ip, int port, bool server) {
    is_server = server;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(port);

    if (is_server) {
        address.sin_addr.s_addr = INADDR_ANY;
        if (bind(sockfd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            perror("Bind failed");
            exit(EXIT_FAILURE);
        }
        listen(sockfd, 1);
        std::cout << "Server listening on port " << port << "...\n";
        sockfd = accept(sockfd, nullptr, nullptr);
        if (sockfd < 0) {
            perror("Accept failed");
            exit(EXIT_FAILURE);
        }
    } else {
        address.sin_addr.s_addr = inet_addr(ip);
        if (connect(sockfd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            perror("Connection failed");
            exit(EXIT_FAILURE);
        }
        std::cout << "Connected to server at " << ip << ":" << port << "\n";
    }
}

NetIO::~NetIO() {
    close(sockfd);
    std::cout << "Connection closed.\n";
}


void NetIO::send_data(const void* data, int nbyte) {
    counter += nbyte;
    int sent = 0;
    while (sent < nbyte) {
        int res = send(sockfd, (const char*)data + sent, nbyte - sent, 0);
        if (res < 0) {
            perror("Send failed");
            exit(EXIT_FAILURE);
        }
        sent += res;
    }
    std::cout << "Sent " << nbyte << " bytes, total: " << counter << " bytes\n";
}


void NetIO::recv_data(void* data, int nbyte) {
    int received = 0;
    while (received < nbyte) {
        int res = recv(sockfd, (char*)data + received, nbyte - received, 0);
        if (res <= 0) {
            perror("Receive failed");
            //exit(EXIT_FAILURE);
            break;
        }
        received += res;
    }
    counter += nbyte;
    std::cout << "Received " << nbyte << " bytes, total: " << counter << " bytes\n";
}
