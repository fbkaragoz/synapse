#ifndef SERVER_H
#define SERVER_H

#include <string>
#include <memory>
#include <vector>
#include "ring_buffer.h"

class Server {
public:
    virtual ~Server() = default;

    // Start the server in a background thread.
    // Port: WebSocket port to listen on.
    // buffer: Shared ring buffer to poll for data to broadcast.
    virtual void start(int port, std::shared_ptr<RingBuffer> buffer) = 0;

    // Stop the server and join the background thread.
    virtual void stop() = 0;

    // Check if server is running
    virtual bool is_running() const = 0;
};

// Factory to create the real implementation
std::unique_ptr<Server> create_uwebsockets_server();

#endif // SERVER_H
