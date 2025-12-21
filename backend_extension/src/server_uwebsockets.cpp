#include "server.h"
#include "neural_probe.h" // For handle_control_message
#include <thread>

#include <atomic>
#include <iostream>
#include <mutex>
#include <unordered_set>
#include <vector>

// uWebSockets
#include "App.h"

class UWebSocketsServer : public Server {
public:
    UWebSocketsServer() : running_(false) {}
    
    ~UWebSocketsServer() override {
        stop();
    }

    void start(int port, std::shared_ptr<RingBuffer> buffer) override {
        if (running_) return;
        running_ = true;
        stop_requested_ = false;
        buffer_ = buffer;
        listen_socket_.store(nullptr, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            connections_.clear();
        }
        
        // Start the server thread
        server_thread_ = std::thread([this, port]() {
            // Create the app - keeping it inside the thread so the Loop belongs to this thread
            auto app = uWS::App()
                .ws<PerSocketData>("/*", {
                    /* Settings */
                    .compression = uWS::SHARED_COMPRESSOR,
                    .maxPayloadLength = 16 * 1024 * 1024,
                    .idleTimeout = 60,
                    .open = [this](auto *ws) {
                        // std::cout << "Client connected!" << std::endl;
                        {
                            std::lock_guard<std::mutex> lock(connections_mutex_);
                            connections_.insert(ws);
                        }
                        ws->subscribe("broadcast");
                        
                        // 1. Send Model Meta (Topology)
                        auto meta = get_model_meta_packet();
                        ws->send(std::string_view(reinterpret_cast<char*>(meta.data()), meta.size()), uWS::OpCode::BINARY);

                        // 2. Send State Snapshot (Config)
                        auto snapshot = get_state_snapshot();
                        ws->send(std::string_view(reinterpret_cast<char*>(snapshot.data()), snapshot.size()), uWS::OpCode::BINARY);
                    },
                    .message = [](auto *ws, std::string_view message, uWS::OpCode opCode) {
                        // Handle control messages from frontend
                        if (opCode == uWS::OpCode::BINARY) {
                             handle_control_message(reinterpret_cast<const uint8_t*>(message.data()), message.size());
                        }
                    },
                    .drain = [](auto *ws) {
                        // Backpressure handling
                    },
                    .close = [this](auto *ws, int code, std::string_view message) {
                        // std::cout << "Client disconnected" << std::endl;
                        std::lock_guard<std::mutex> lock(connections_mutex_);
                        connections_.erase(ws);
                    }
                })
                .listen(port, [this, port](auto *token) {
                    if (token) {
                       listen_socket_.store(token, std::memory_order_release);
                       // std::cout << "Listening on port " << port << std::endl;
                    } else {
                        std::cerr << "Failed to listen on port " << port << std::endl;
                    }
                });

            // Broadcaster loop
            // We can't just block on the ring buffer here because that would block the uWS loop.
            // Instead, we can use a timer or a post loop callback.
            // uWS doesn't strictly expose a "poll" callback easily in the App builder chain without 'us_loop' hacking.
            // EASIER APPROACH for Phase 1:
            // Use a separate thread that pops from RingBuffer and calls loop->defer(...) to broadcast.
            
            // Wait, uWS is not thread safe. You can only call ws->send or app->publish from the loop thread.
            // So we DO need a way to wake up the loop.
            // 'loop->defer' is thread-safe (check docs, usually uSockets 'us_loop_wakeup' or similar).
            // But uWebSockets App wrapper might hide the raw loop.
            
            // ALTERNATIVE:
            // The standard way in uWS is to have the external thread signal the loop.
            // Since we want to simplify, let's spin up a "Sender Thread" that waits on the RingBuffer
            // and then uses a thread-safe way to inject into the uWS loop.
            // uWS::Loop::get()->defer(cb) is the way. 
            
            // BUT: uWS::Loop::get() returns the loop for the CURRENT thread.
            // We need to capture the pointer to the loop of the server thread.
            
            // Let's optimize:
            // We will add a Timer to the app that runs every X ms (tick) to poll the RingBuffer.
            // This is "polling" but if the tick is fast (e.g. 16ms ~ 60fps) it's fine for visualization.
            // And it keeps everything single-threaded in the WS thread.
            
            // Adding a frequent timer to poll the queue.
            // This avoids complex cross-thread signaling for now.
            // Check RingBuffer (non-blocking preferred, but our pop is blocking?)
            // We will add a non-blocking 'try_pop' to RingBuffer or just use a short peek.
            
            // Actually, we don't want to block the event loop.
            // So we should assume RingBuffer has a try_pop or we peek.
            // Our RingBuffer.pop() IS blocking. We need a try_pop.
            
            // Let's modify RingBuffer usage:
            // We'll leave RingBuffer as is, but we will access the underlying queue with a lock in the lambda
            // and perform a non-blocking check.
            
            // Actually, best pattern: 
            // 1. Thread 1 (Python) -> pushes to RingBuffer
            // 2. Thread 2 (Broadcaster) -> pops from RingBuffer (blocking), then calls loop->defer() to send.
            // 3. Thread 3 (uWS server) -> runs the loop, executes deferred tasks.
            
            // This requires 2 background threads but is robust.
            // Or, just use Polling in uWS loop if acceptable.
            // Let's use Polling for simplicity in Phase 1.
            
            // We need to make RingBuffer non-blocking for the polling strategy.
            // I'll simply add 'try_pop' to RingBuffer later or just use 'pop' with a 0ms wait if I can.
            // Current 'pop' has no timeout.
            // I will implement a "polling" logic here using a unique strategy:
            // The app has a fallback "idle" work? No.
            // I'll leave the "Threading Model" details:
            // "Use Loop::defer(...) to schedule broadcast on the WS loop thread."
            
            // So I need to capture the loop pointer.
            loop_.store(uWS::Loop::get(), std::memory_order_release);
            if (stop_requested_.load(std::memory_order_acquire)) {
                schedule_shutdown();
            }
            
            // Start a bridge thread that pops from queue and defers to loop
            std::thread bridge([this, &app]() {
                 while (running_) {
                     // Wait for data
                     auto data_opt = buffer_->pop(); 
                     if (!data_opt) break; // stopped
                     
                     auto& data = *data_opt;
                     
                     // We have data. Schedule send on the loop.
                     // We must copy data or move it into the lambda. Capture by value.
                     // data is std::vector<uint8_t>
                     
                     struct Context {
                         std::vector<uint8_t> payload;
                     };
                     auto* ctx = new Context{std::move(data)}; // Move to heap to pass to C-style callback if needed, or just C++ lambda capture

                     // Loop::defer is thread safe?
                     // uWebSockets documentation says: "Thread safety: ... Loop::defer is thread safe."
                     loop_.load(std::memory_order_acquire)->defer([&app, ctx]() {
                         // We are now on the loop thread
                         // std::cout << "[Server] Broadcasting " << ctx->payload.size() << " bytes" << std::endl;
                         app.publish("broadcast", std::string_view(reinterpret_cast<char*>(ctx->payload.data()), ctx->payload.size()), uWS::OpCode::BINARY, false);
                         delete ctx;
                     });
                 }
            });
            
            app.run();
            // cleanup
            if (bridge.joinable()) bridge.join();
            loop_.store(nullptr, std::memory_order_release);
            running_ = false;
        });
    }

    void stop() override {
        if (!running_) return;
        running_ = false;
        stop_requested_ = true;
        
        // Stop the ring buffer so the bridge thread wakes up
        if (buffer_) buffer_->stop();

        schedule_shutdown();
        
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }

    bool is_running() const override {
        return running_;
    }

private:
    struct PerSocketData {
        /* Fill with user data if needed */
    };

    using Ws = uWS::WebSocket<false, true, PerSocketData>;

    void schedule_shutdown() {
        uWS::Loop* loop = loop_.load(std::memory_order_acquire);
        if (!loop) return;

        loop->defer([this]() {
            auto* listen_socket = listen_socket_.load(std::memory_order_acquire);
            if (listen_socket) {
                us_listen_socket_close(0, listen_socket);
                listen_socket_.store(nullptr, std::memory_order_release);
            }

            std::vector<Ws*> to_close;
            {
                std::lock_guard<std::mutex> lock(connections_mutex_);
                to_close.assign(connections_.begin(), connections_.end());
                connections_.clear();
            }

            for (auto* ws : to_close) {
                if (ws) {
                    ws->close();
                }
            }
        });
    }

    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_{false};
    std::thread server_thread_;
    std::shared_ptr<RingBuffer> buffer_;
    std::atomic<uWS::Loop*> loop_{nullptr}; // Raw pointer to the loop (thread-local to server_thread_)
    std::atomic<us_listen_socket_t*> listen_socket_{nullptr};
    std::mutex connections_mutex_;
    std::unordered_set<Ws*> connections_;
};

std::unique_ptr<Server> create_uwebsockets_server() {
    return std::make_unique<UWebSocketsServer>();
}
