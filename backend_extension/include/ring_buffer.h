#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <vector>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <iostream>

// Simple thread-safe queue for buffering frames between Python thread and WS thread.
// Not a strict "ring buffer" in the POD sense, but a bounded queue of byte buffers.
// In a production "zero-copy" ring buffer we'd write directly to a mmapped region,
// but for simplicity and safety, we'll store std::vector<uint8_t> packets.

class RingBuffer {
public:
    explicit RingBuffer(size_t max_size = 100) : max_size_(max_size), stopped_(false) {}

    // Producer
    bool push(std::vector<uint8_t>&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stopped_) return false;
        
        // If full, drop oldest (or block? dropping is better for real-time viz)
        if (queue_.size() >= max_size_) {
            // Strategy: Drop oldest to make room for new. Real-time > complete history.
            queue_.erase(queue_.begin()); 
        }

        queue_.push_back(std::move(item));
        lock.unlock();
        cv_.notify_one();
        return true;
    }

    // Consumer (blocking)
    std::optional<std::vector<uint8_t>> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });

        if (queue_.empty() && stopped_) {
            return std::nullopt;
        }

        if (queue_.empty()) return std::nullopt; // should not happen if wait predicate correct

        std::vector<uint8_t> item = std::move(queue_.front());
        queue_.erase(queue_.begin());
        return item;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

private:
    size_t max_size_;
    std::vector<std::vector<uint8_t>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopped_;
};

#endif // RING_BUFFER_H
