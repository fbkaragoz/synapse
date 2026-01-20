#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <optional>

class RingBuffer {
public:
    explicit RingBuffer(size_t max_size = 100);

    bool push(std::vector<uint8_t>&& item);

    std::optional<std::vector<uint8_t>> pop();

    void stop();

private:
    size_t max_size_;
    std::vector<std::vector<uint8_t>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopped_;
};

#endif // RING_BUFFER_H
