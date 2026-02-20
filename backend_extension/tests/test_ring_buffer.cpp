#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include "ring_buffer.h"

TEST_CASE("RingBuffer basic operations", "[ring_buffer]") {
    
    SECTION("Push and pop single item") {
        RingBuffer buffer(10);
        std::vector<uint8_t> data = {1, 2, 3, 4, 5};
        
        REQUIRE(buffer.push(std::move(data)));
        
        auto result = buffer.pop();
        REQUIRE(result.has_value());
        REQUIRE(result->size() == 5);
        REQUIRE((*result)[0] == 1);
        REQUIRE((*result)[4] == 5);
    }
    
    SECTION("Empty buffer pop blocks until data or stop") {
        RingBuffer buffer(10);
        std::atomic<bool> popped{false};
        
        std::thread consumer([&]() {
            auto result = buffer.pop();
            popped = true;
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        REQUIRE(!popped);
        
        buffer.stop();
        consumer.join();
    }
    
    SECTION("Stop wakes up blocked pop") {
        RingBuffer buffer(10);
        std::atomic<bool> pop_returned{false};
        
        std::thread consumer([&]() {
            auto result = buffer.pop();
            pop_returned = true;
            REQUIRE(!result.has_value());
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        REQUIRE(!pop_returned);
        
        buffer.stop();
        consumer.join();
        REQUIRE(pop_returned);
    }
}

TEST_CASE("RingBuffer overflow behavior", "[ring_buffer]") {
    
    SECTION("Drop oldest when buffer full") {
        RingBuffer buffer(3);
        
        std::vector<uint8_t> first = {1};
        std::vector<uint8_t> second = {2};
        std::vector<uint8_t> third = {3};
        std::vector<uint8_t> fourth = {4};
        
        REQUIRE(buffer.push(std::move(first)));
        REQUIRE(buffer.push(std::move(second)));
        REQUIRE(buffer.push(std::move(third)));
        REQUIRE(buffer.push(std::move(fourth)));
        
        auto r1 = buffer.pop();
        auto r2 = buffer.pop();
        auto r3 = buffer.pop();
        
        REQUIRE(r1.has_value());
        REQUIRE(r2.has_value());
        REQUIRE(r3.has_value());
        
        REQUIRE((*r1)[0] == 2);
        REQUIRE((*r2)[0] == 3);
        REQUIRE((*r3)[0] == 4);
    }
    
    SECTION("Push after stop returns false") {
        RingBuffer buffer(10);
        buffer.stop();
        
        std::vector<uint8_t> data = {1, 2, 3};
        REQUIRE(!buffer.push(std::move(data)));
    }
}

TEST_CASE("RingBuffer concurrent access", "[ring_buffer]") {
    
    SECTION("Multiple producers single consumer") {
        RingBuffer buffer(1000);
        std::atomic<bool> running{true};
        std::atomic<int> produced_count{0};
        std::atomic<int> consumed_count{0};
        const int items_per_producer = 50;
        const int num_producers = 4;
        
        std::vector<std::thread> producers;
        for (int p = 0; p < num_producers; ++p) {
            producers.emplace_back([&, p]() {
                for (int i = 0; i < items_per_producer; ++i) {
                    std::vector<uint8_t> data(8, static_cast<uint8_t>(p * 100 + i));
                    buffer.push(std::move(data));
                    produced_count++;
                }
            });
        }
        
        std::thread consumer([&]() {
            while (running) {
                auto result = buffer.pop();
                if (result.has_value()) {
                    consumed_count++;
                }
            }
        });
        
        for (auto& t : producers) {
            t.join();
        }
        
        int wait_cycles = 0;
        while (consumed_count < produced_count && wait_cycles < 100) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_cycles++;
        }
        
        running = false;
        buffer.stop();
        consumer.join();
        
        REQUIRE(produced_count == num_producers * items_per_producer);
        REQUIRE(consumed_count >= produced_count * 0.9);
    }
    
    SECTION("No deadlock under stress") {
        RingBuffer buffer(50);
        std::atomic<bool> running{true};
        std::atomic<int> total_ops{0};
        
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&, t]() {
                while (running) {
                    if (t % 2 == 0) {
                        std::vector<uint8_t> data(16, static_cast<uint8_t>(t));
                        buffer.push(std::move(data));
                    } else {
                        auto result = buffer.pop();
                        if (!result.has_value() && !running) break;
                    }
                    total_ops++;
                }
            });
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        running = false;
        buffer.stop();
        
        for (auto& t : threads) {
            t.join();
        }
        
        REQUIRE(total_ops > 0);
    }
}

TEST_CASE("RingBuffer edge cases", "[ring_buffer]") {
    
    SECTION("Size one buffer") {
        RingBuffer buffer(1);
        
        std::vector<uint8_t> a = {1};
        std::vector<uint8_t> b = {2};
        std::vector<uint8_t> c = {3};
        
        buffer.push(std::move(a));
        buffer.push(std::move(b));
        buffer.push(std::move(c));
        
        auto result = buffer.pop();
        REQUIRE(result.has_value());
        REQUIRE((*result)[0] == 3);
    }
    
    SECTION("Large data items") {
        RingBuffer buffer(10);
        
        std::vector<uint8_t> large(1024 * 1024, 0xAB);
        buffer.push(std::move(large));
        
        auto result = buffer.pop();
        REQUIRE(result.has_value());
        REQUIRE(result->size() == 1024 * 1024);
        REQUIRE((*result)[0] == 0xAB);
    }
    
    SECTION("Double stop is safe") {
        RingBuffer buffer(10);
        buffer.stop();
        buffer.stop();
    }
}
