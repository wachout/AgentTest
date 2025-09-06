#ifndef SOCKET_CLIENT_H
#define SOCKET_CLIENT_H

#include <stddef.h> // For size_t

/**
 * @brief Connects to a TCP server.
 *
 * @param ip The IP address of the server.
 * @param port The port number of the server.
 * @return The socket file descriptor on success, or -1 on failure.
 */
int socket_connect(const char* ip, int port);

/**
 * @brief Reads a specified number of bytes from the socket.
 *
 * This function will block until all requested bytes are read.
 *
 * @param sockfd The socket file descriptor.
 * @param buffer The buffer to read data into.
 * @param len The number of bytes to read.
 * @return 0 on success, -1 on failure (e.g., connection closed).
 */
int socket_read_fully(int sockfd, void* buffer, size_t len);

/**
 * @brief Disconnects from the server and closes the socket.
 *
 * @param sockfd The socket file descriptor to close.
 */
void socket_disconnect(int sockfd);

#endif // SOCKET_CLIENT_H
