#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "protocol_parser.h"

using namespace nf;
using Catch::Approx;

TEST_CASE("Protocol header parsing", "[protocol]") {
    
    SECTION("Valid header is parsed correctly") {
        std::vector<uint8_t> buffer = build_packet(
            NF_MSG_LAYER_SUMMARY_BATCH,
            NF_FLAG_FP32,
            42,
            1234567890,
            nullptr,
            0
        );
        
        ParsedHeader header;
        ParseResult result = parse_header(buffer, header);
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(header.magic == 0x574C464E);
        REQUIRE(header.version == 1);
        REQUIRE(header.msg_type == NF_MSG_LAYER_SUMMARY_BATCH);
        REQUIRE(header.flags == NF_FLAG_FP32);
        REQUIRE(header.seq == 42);
        REQUIRE(header.timestamp_ns == 1234567890);
        REQUIRE(header.payload_bytes == 0);
    }
    
    SECTION("Buffer too short returns error") {
        std::vector<uint8_t> short_buffer(31, 0);
        
        ParsedHeader header;
        ParseResult result = parse_header(short_buffer, header);
        
        REQUIRE(result == ParseResult::ERR_BUFFER_TOO_SHORT);
    }
    
    SECTION("Invalid magic returns error") {
        std::vector<uint8_t> buffer(32, 0);
        buffer[0] = 'X';
        buffer[1] = 'Y';
        buffer[2] = 'Z';
        buffer[3] = 'W';
        
        ParsedHeader header;
        ParseResult result = parse_header(buffer, header);
        
        REQUIRE(result == ParseResult::ERR_INVALID_MAGIC);
    }
    
    SECTION("Invalid version returns error") {
        std::vector<uint8_t> buffer = build_packet(
            NF_MSG_LAYER_SUMMARY_BATCH, 0, 0, 0, nullptr, 0
        );
        *reinterpret_cast<uint16_t*>(buffer.data() + 4) = 99;
        
        ParsedHeader header;
        ParseResult result = parse_header(buffer, header);
        
        REQUIRE(result == ParseResult::ERR_INVALID_VERSION);
    }
    
    SECTION("Truncated payload returns error") {
        std::vector<uint8_t> buffer = build_packet(
            NF_MSG_LAYER_SUMMARY_BATCH, 0, 0, 0, nullptr, 100
        );
        buffer.resize(40);
        
        ParsedHeader header;
        ParseResult result = parse_header(buffer, header);
        
        REQUIRE(result == ParseResult::ERR_TRUNCATED_PAYLOAD);
    }
}

TEST_CASE("Layer summary batch parsing", "[protocol]") {
    
    SECTION("Empty batch") {
        std::vector<uint8_t> payload(8, 0);
        ParsedLayerSummaryBatch batch;
        ParseResult result = parse_layer_summary_batch(payload.data(), payload.size(), batch);
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(batch.summaries.empty());
    }
    
    SECTION("Single layer summary") {
        std::vector<ParsedLayerSummary> summaries = {
            {1, 1000, 0.5f, 2.0f}
        };
        auto packet = build_layer_summary_packet(summaries);
        
        ParsedHeader header;
        REQUIRE(parse_header(packet, header) == ParseResult::OK);
        
        ParsedLayerSummaryBatch batch;
        ParseResult result = parse_layer_summary_batch(
            packet.data() + 32, header.payload_bytes, batch
        );
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(batch.summaries.size() == 1);
        REQUIRE(batch.summaries[0].layer_id == 1);
        REQUIRE(batch.summaries[0].neuron_count == 1000);
        REQUIRE(batch.summaries[0].mean == Approx(0.5f));
        REQUIRE(batch.summaries[0].max == Approx(2.0f));
    }
    
    SECTION("Multiple layer summaries") {
        std::vector<ParsedLayerSummary> summaries = {
            {0, 512, 0.1f, 1.0f},
            {1, 1024, 0.2f, 2.0f},
            {2, 2048, 0.3f, 3.0f}
        };
        auto packet = build_layer_summary_packet(summaries);
        
        ParsedHeader header;
        REQUIRE(parse_header(packet, header) == ParseResult::OK);
        
        ParsedLayerSummaryBatch batch;
        ParseResult result = parse_layer_summary_batch(
            packet.data() + 32, header.payload_bytes, batch
        );
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(batch.summaries.size() == 3);
        REQUIRE(batch.summaries[0].layer_id == 0);
        REQUIRE(batch.summaries[1].layer_id == 1);
        REQUIRE(batch.summaries[2].layer_id == 2);
        REQUIRE(batch.summaries[2].neuron_count == 2048);
        REQUIRE(batch.summaries[2].mean == Approx(0.3f));
    }
    
    SECTION("Truncated batch returns error") {
        std::vector<uint8_t> payload(10, 0);
        *reinterpret_cast<uint32_t*>(payload.data()) = 5;
        
        ParsedLayerSummaryBatch batch;
        ParseResult result = parse_layer_summary_batch(payload.data(), payload.size(), batch);
        
        REQUIRE(result == ParseResult::ERR_TRUNCATED_PAYLOAD);
    }
}

TEST_CASE("Control packet parsing", "[protocol]") {
    
    SECTION("Threshold control packet") {
        auto packet = build_control_packet(NF_OP_SET_THRESHOLD, 0, 0.75f);
        
        ParsedHeader header;
        REQUIRE(parse_header(packet, header) == ParseResult::OK);
        
        ParsedControlPacket ctrl;
        ParseResult result = parse_control_packet(
            packet.data() + 32, header.payload_bytes, ctrl
        );
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(ctrl.opcode == NF_OP_SET_THRESHOLD);
        REQUIRE(ctrl.value_f32 == Approx(0.75f));
    }
    
    SECTION("Accumulation steps control packet") {
        auto packet = build_control_packet(NF_OP_SET_ACCUMULATION_STEPS, 10, 0.0f);
        
        ParsedHeader header;
        REQUIRE(parse_header(packet, header) == ParseResult::OK);
        
        ParsedControlPacket ctrl;
        ParseResult result = parse_control_packet(
            packet.data() + 32, header.payload_bytes, ctrl
        );
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(ctrl.opcode == NF_OP_SET_ACCUMULATION_STEPS);
        REQUIRE(ctrl.value_u32 == 10);
    }
    
    SECTION("Truncated control packet returns error") {
        std::vector<uint8_t> payload(8, 0);
        
        ParsedControlPacket ctrl;
        ParseResult result = parse_control_packet(payload.data(), payload.size(), ctrl);
        
        REQUIRE(result == ParseResult::ERR_BUFFER_TOO_SHORT);
    }
}

TEST_CASE("Sparse activation parsing", "[protocol]") {
    
    SECTION("Empty sparse activations") {
        uint8_t payload[16] = {0};
        *reinterpret_cast<uint32_t*>(payload) = 1;
        *reinterpret_cast<uint32_t*>(payload + 4) = 0;
        
        ParsedSparseActivation sparse;
        ParseResult result = parse_sparse_activation(payload, 16, sparse);
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(sparse.layer_id == 1);
        REQUIRE(sparse.indices.empty());
        REQUIRE(sparse.values.empty());
    }
    
    SECTION("Sparse activations with entries") {
        struct Entry { uint32_t idx; float val; };
        std::vector<Entry> entries = {
            {0, 0.5f},
            {10, 0.8f},
            {100, 1.2f}
        };
        
        std::vector<uint8_t> payload(16 + entries.size() * 8);
        *reinterpret_cast<uint32_t*>(payload.data()) = 5;
        *reinterpret_cast<uint32_t*>(payload.data() + 4) = static_cast<uint32_t>(entries.size());
        
        for (size_t i = 0; i < entries.size(); ++i) {
            uint8_t* entry = payload.data() + 16 + i * 8;
            *reinterpret_cast<uint32_t*>(entry) = entries[i].idx;
            *reinterpret_cast<float*>(entry + 4) = entries[i].val;
        }
        
        ParsedSparseActivation sparse;
        ParseResult result = parse_sparse_activation(payload.data(), payload.size(), sparse);
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(sparse.layer_id == 5);
        REQUIRE(sparse.indices.size() == 3);
        REQUIRE(sparse.indices[0] == 0);
        REQUIRE(sparse.indices[1] == 10);
        REQUIRE(sparse.indices[2] == 100);
        REQUIRE(sparse.values[0] == Approx(0.5f));
        REQUIRE(sparse.values[1] == Approx(0.8f));
        REQUIRE(sparse.values[2] == Approx(1.2f));
    }
    
    SECTION("Truncated sparse activations returns error") {
        uint8_t payload[16] = {0};
        *reinterpret_cast<uint32_t*>(payload + 4) = 10;
        
        ParsedSparseActivation sparse;
        ParseResult result = parse_sparse_activation(payload, 16, sparse);
        
        REQUIRE(result == ParseResult::ERR_TRUNCATED_PAYLOAD);
    }
}

TEST_CASE("Packet building roundtrip", "[protocol]") {
    
    SECTION("Layer summary batch roundtrip") {
        std::vector<ParsedLayerSummary> original = {
            {42, 4096, 0.123f, 5.67f}
        };
        
        auto packet = build_layer_summary_packet(original, 100, 99999);
        
        ParsedHeader header;
        REQUIRE(parse_header(packet, header) == ParseResult::OK);
        REQUIRE(header.seq == 100);
        REQUIRE(header.timestamp_ns == 99999);
        
        ParsedLayerSummaryBatch batch;
        REQUIRE(parse_layer_summary_batch(packet.data() + 32, header.payload_bytes, batch) == ParseResult::OK);
        
        REQUIRE(batch.summaries.size() == 1);
        REQUIRE(batch.summaries[0].layer_id == 42);
        REQUIRE(batch.summaries[0].neuron_count == 4096);
        REQUIRE(batch.summaries[0].mean == Approx(0.123f));
        REQUIRE(batch.summaries[0].max == Approx(5.67f));
    }
}

TEST_CASE("Gradient batch parsing", "[protocol]") {
    
    SECTION("Empty gradient batch") {
        std::vector<uint8_t> payload(12, 0);
        ParsedGradientBatch batch;
        ParseResult result = parse_gradient_batch(payload.data(), payload.size(), batch);
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(batch.training_step == 0);
        REQUIRE(batch.global_grad_norm == Approx(0.0f));
        REQUIRE(batch.gradients.empty());
    }
    
    SECTION("Single gradient summary") {
        std::vector<ParsedGradientSummary> summaries = {
            {0, 1000, 0.01f, 0.02f, -0.1f, 0.5f, 1.5f, 10.0f, 0.15f}
        };
        
        auto packet = build_gradient_batch_packet(summaries, 42, 2.5f);
        
        ParsedHeader header;
        REQUIRE(parse_header(packet, header) == ParseResult::OK);
        REQUIRE(header.msg_type == NF_MSG_GRADIENT_BATCH);
        
        ParsedGradientBatch batch;
        ParseResult result = parse_gradient_batch(packet.data() + 32, header.payload_bytes, batch);
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(batch.training_step == 42);
        REQUIRE(batch.global_grad_norm == Approx(2.5f));
        REQUIRE(batch.gradients.size() == 1);
        REQUIRE(batch.gradients[0].layer_id == 0);
        REQUIRE(batch.gradients[0].param_count == 1000);
        REQUIRE(batch.gradients[0].grad_mean == Approx(0.01f));
        REQUIRE(batch.gradients[0].grad_std == Approx(0.02f));
        REQUIRE(batch.gradients[0].grad_min == Approx(-0.1f));
        REQUIRE(batch.gradients[0].grad_max == Approx(0.5f));
        REQUIRE(batch.gradients[0].grad_l2_norm == Approx(1.5f));
        REQUIRE(batch.gradients[0].weight_l2_norm == Approx(10.0f));
        REQUIRE(batch.gradients[0].grad_to_weight == Approx(0.15f));
    }
    
    SECTION("Multiple gradient summaries") {
        std::vector<ParsedGradientSummary> summaries = {
            {0, 512, 0.001f, 0.002f, -0.05f, 0.1f, 0.5f, 5.0f, 0.1f},
            {1, 1024, 0.002f, 0.003f, -0.1f, 0.2f, 1.0f, 10.0f, 0.1f},
            {2, 2048, 0.003f, 0.004f, -0.2f, 0.3f, 2.0f, 20.0f, 0.1f}
        };
        
        auto packet = build_gradient_batch_packet(summaries, 100, 5.0f, 50, 1000000);
        
        ParsedHeader header;
        REQUIRE(parse_header(packet, header) == ParseResult::OK);
        REQUIRE(header.seq == 50);
        REQUIRE(header.timestamp_ns == 1000000);
        
        ParsedGradientBatch batch;
        ParseResult result = parse_gradient_batch(packet.data() + 32, header.payload_bytes, batch);
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(batch.training_step == 100);
        REQUIRE(batch.global_grad_norm == Approx(5.0f));
        REQUIRE(batch.gradients.size() == 3);
        
        REQUIRE(batch.gradients[0].layer_id == 0);
        REQUIRE(batch.gradients[1].layer_id == 1);
        REQUIRE(batch.gradients[2].layer_id == 2);
        
        REQUIRE(batch.gradients[2].param_count == 2048);
        REQUIRE(batch.gradients[2].grad_l2_norm == Approx(2.0f));
    }
    
    SECTION("Truncated gradient batch returns error") {
        std::vector<uint8_t> payload(20, 0);
        *reinterpret_cast<uint32_t*>(payload.data()) = 5;
        
        ParsedGradientBatch batch;
        ParseResult result = parse_gradient_batch(payload.data(), payload.size(), batch);
        
        REQUIRE(result == ParseResult::ERR_TRUNCATED_PAYLOAD);
    }
    
    SECTION("Buffer too short returns error") {
        uint8_t payload[8] = {0};
        
        ParsedGradientBatch batch;
        ParseResult result = parse_gradient_batch(payload, 8, batch);
        
        REQUIRE(result == ParseResult::ERR_BUFFER_TOO_SHORT);
    }
}

TEST_CASE("Gradient statistics computation", "[protocol]") {
    
    SECTION("Basic gradient stats") {
        std::vector<float> data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        GradientStats stats;
        compute_gradient_stats(data.data(), data.size(), stats);
        
        REQUIRE(stats.mean == Approx(0.3f));
        REQUIRE(stats.min == Approx(0.1f));
        REQUIRE(stats.max == Approx(0.5f));
        REQUIRE(stats.l2_norm > 0);
    }
    
    SECTION("Gradient stats with negative values") {
        std::vector<float> data = {-0.1f, 0.0f, 0.1f, 0.2f, 0.3f};
        GradientStats stats;
        compute_gradient_stats(data.data(), data.size(), stats);
        
        REQUIRE(stats.min == Approx(-0.1f));
        REQUIRE(stats.max == Approx(0.3f));
    }
    
    SECTION("Empty data returns zeros") {
        GradientStats stats;
        compute_gradient_stats(static_cast<float*>(nullptr), 0, stats);
        
        REQUIRE(stats.mean == Approx(0.0f));
        REQUIRE(stats.std == Approx(0.0f));
    }
    
    SECTION("Gradient explosion detection scenario") {
        std::vector<float> data(1000, 0.001f);
        data[500] = 100.0f;
        
        GradientStats stats;
        compute_gradient_stats(data.data(), data.size(), stats);
        
        REQUIRE(stats.max > 50.0f);
        REQUIRE(stats.max / (stats.mean + 1e-10f) > 100);
    }
    
    SECTION("Vanishing gradient detection scenario") {
        std::vector<float> data(1000, 1e-10f);
        
        GradientStats stats;
        compute_gradient_stats(data.data(), data.size(), stats);
        
        REQUIRE(stats.l2_norm < 1e-5f);
    }
}
