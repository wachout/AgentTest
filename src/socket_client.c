#include "socket_client.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <errno.h>

int socket_connect(const char* ip, int port) {
    int sockfd;
    struct sockaddr_in serv_addr;

    // Create socket
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation error");
        return -1;
    }

    memset(&serv_addr, '0', sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        close(sockfd);
        return -1;
    }

    // Connect to the server
    if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection Failed");
        close(sockfd);
        return -1;
    }

    return sockfd;
}

int socket_read_fully(int sockfd, void* buffer, size_t len) {
    char* p_buffer = (char*)buffer;
    size_t bytes_read = 0;
    while (bytes_read < len) {
        ssize_t result = read(sockfd, p_buffer + bytes_read, len - bytes_read);
        if (result < 0) {
            // Error
            perror("Socket read error");
            return -1;
        }
        if (result == 0) {
            // Connection closed by peer
            fprintf(stderr, "Socket connection closed by peer.\n");
            return -1;
        }
        bytes_read += result;
    }
    return 0;
}

void socket_disconnect(int sockfd) {
    if (sockfd >= 0) {
        close(sockfd);
    }
}
