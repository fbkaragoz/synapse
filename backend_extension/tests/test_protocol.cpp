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

TEST_CASE("Layer summary batch V2 parsing", "[protocol]") {
    
    SECTION("Empty V2 batch") {
        std::vector<uint8_t> payload(8, 0);
        ParsedLayerSummaryBatchV2 batch;
        ParseResult result = parse_layer_summary_batch_v2(payload.data(), payload.size(), batch);
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(batch.summaries.empty());
    }
    
    SECTION("Single V2 summary") {
        ParsedLayerSummaryV2 s;
        s.layer_id = 1;
        s.neuron_count = 1000;
        s.mean = 0.5f;
        s.std = 0.2f;
        s.min = 0.0f;
        s.max = 2.0f;
        s.l2_norm = 10.0f;
        s.zero_ratio = 0.1f;
        s.p5 = 0.05f;
        s.p25 = 0.25f;
        s.p75 = 0.75f;
        s.p95 = 1.9f;
        s.kurtosis = 0.5f;
        s.skewness = 0.3f;
        s.flags = 0;
        
        auto packet = build_layer_summary_packet_v2({s}, 100, 1000000);
        
        ParsedHeader header;
        REQUIRE(parse_header(packet, header) == ParseResult::OK);
        REQUIRE(header.msg_type == NF_MSG_LAYER_SUMMARY_BATCH_V2);
        REQUIRE(header.seq == 100);
        
        ParsedLayerSummaryBatchV2 batch;
        ParseResult result = parse_layer_summary_batch_v2(
            packet.data() + 32, header.payload_bytes, batch
        );
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(batch.summaries.size() == 1);
        REQUIRE(batch.summaries[0].layer_id == 1);
        REQUIRE(batch.summaries[0].neuron_count == 1000);
        REQUIRE(batch.summaries[0].mean == Approx(0.5f));
        REQUIRE(batch.summaries[0].std == Approx(0.2f));
        REQUIRE(batch.summaries[0].min == Approx(0.0f));
        REQUIRE(batch.summaries[0].max == Approx(2.0f));
        REQUIRE(batch.summaries[0].l2_norm == Approx(10.0f));
        REQUIRE(batch.summaries[0].zero_ratio == Approx(0.1f));
        REQUIRE(batch.summaries[0].p5 == Approx(0.05f));
        REQUIRE(batch.summaries[0].p25 == Approx(0.25f));
        REQUIRE(batch.summaries[0].p75 == Approx(0.75f));
        REQUIRE(batch.summaries[0].p95 == Approx(1.9f));
        REQUIRE(batch.summaries[0].kurtosis == Approx(0.5f));
        REQUIRE(batch.summaries[0].skewness == Approx(0.3f));
    }
    
    SECTION("Multiple V2 summaries") {
        std::vector<ParsedLayerSummaryV2> summaries(3);
        for (int i = 0; i < 3; ++i) {
            summaries[i].layer_id = i;
            summaries[i].neuron_count = 512 * (i + 1);
            summaries[i].mean = 0.1f * (i + 1);
            summaries[i].std = 0.05f * (i + 1);
            summaries[i].min = -0.5f;
            summaries[i].max = 1.0f + i * 0.5f;
            summaries[i].l2_norm = 5.0f + i;
            summaries[i].zero_ratio = 0.2f;
            summaries[i].p5 = 0.02f;
            summaries[i].p25 = 0.15f;
            summaries[i].p75 = 0.85f;
            summaries[i].p95 = 0.98f;
            summaries[i].kurtosis = 0.0f;
            summaries[i].skewness = 0.0f;
            summaries[i].flags = 0;
        }
        
        auto packet = build_layer_summary_packet_v2(summaries);
        
        ParsedHeader header;
        REQUIRE(parse_header(packet, header) == ParseResult::OK);
        
        ParsedLayerSummaryBatchV2 batch;
        ParseResult result = parse_layer_summary_batch_v2(
            packet.data() + 32, header.payload_bytes, batch
        );
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(batch.summaries.size() == 3);
        REQUIRE(batch.summaries[0].layer_id == 0);
        REQUIRE(batch.summaries[1].layer_id == 1);
        REQUIRE(batch.summaries[2].layer_id == 2);
        REQUIRE(batch.summaries[2].neuron_count == 1536);
        REQUIRE(batch.summaries[2].l2_norm == Approx(7.0f));
    }
    
    SECTION("Truncated V2 batch returns error") {
        std::vector<uint8_t> payload(20, 0);
        *reinterpret_cast<uint32_t*>(payload.data()) = 5;
        
        ParsedLayerSummaryBatchV2 batch;
        ParseResult result = parse_layer_summary_batch_v2(payload.data(), payload.size(), batch);
        
        REQUIRE(result == ParseResult::ERR_TRUNCATED_PAYLOAD);
    }
    
    SECTION("Statistics to V2 summary conversion") {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Statistics stats;
        compute_statistics(data, stats);
        
        ParsedLayerSummaryV2 s = statistics_to_v2_summary(42, 5, stats);
        
        REQUIRE(s.layer_id == 42);
        REQUIRE(s.neuron_count == 5);
        REQUIRE(s.mean == Approx(3.0f));
        REQUIRE(s.min == Approx(1.0f));
        REQUIRE(s.max == Approx(5.0f));
    }
    
    SECTION("Dead layer flag detection") {
        std::vector<float> data(100, 0.0f);
        for (int i = 0; i < 30; ++i) data[i] = 0.1f;
        
        Statistics stats;
        compute_statistics(data, stats);
        
        ParsedLayerSummaryV2 s = statistics_to_v2_summary(0, 100, stats);
        
        REQUIRE(s.flags & NF_LAYER_FLAG_DEAD);
    }
    
    SECTION("Exploding layer flag detection") {
        std::vector<float> data(1000, 0.001f);
        data[500] = 1000.0f;
        
        Statistics stats;
        compute_statistics(data, stats);
        
        ParsedLayerSummaryV2 s = statistics_to_v2_summary(0, 1000, stats);
        
        REQUIRE(s.flags & NF_LAYER_FLAG_EXPLODING);
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

TEST_CASE("Attention pattern parsing", "[protocol]") {
    
    SECTION("Empty attention pattern") {
        std::vector<uint8_t> payload(20, 0);
        ParsedAttentionPattern pattern;
        ParseResult result = parse_attention_pattern(payload.data(), payload.size(), pattern);
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(pattern.entries.empty());
    }
    
    SECTION("Single attention entry") {
        std::vector<ParsedAttentionEntry> entries = {
            {0, 0, 1.0f}
        };
        
        auto packet = build_attention_packet(5, 2, 10, 10, NF_ATTENTION_TOP_K, entries, 100, 1000000);
        
        ParsedHeader header;
        REQUIRE(parse_header(packet, header) == ParseResult::OK);
        REQUIRE(header.msg_type == NF_MSG_ATTENTION_PATTERN);
        REQUIRE(header.seq == 100);
        
        ParsedAttentionPattern pattern;
        ParseResult result = parse_attention_pattern(packet.data() + 32, header.payload_bytes, pattern);
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(pattern.layer_id == 5);
        REQUIRE(pattern.head_id == 2);
        REQUIRE(pattern.seq_len == 10);
        REQUIRE(pattern.tgt_len == 10);
        REQUIRE(pattern.mode == NF_ATTENTION_TOP_K);
        REQUIRE(pattern.entries.size() == 1);
        REQUIRE(pattern.entries[0].src_idx == 0);
        REQUIRE(pattern.entries[0].tgt_idx == 0);
        REQUIRE(pattern.entries[0].weight == Approx(1.0f));
    }
    
    SECTION("Multiple attention entries") {
        std::vector<ParsedAttentionEntry> entries = {
            {0, 1, 0.5f},
            {0, 2, 0.3f},
            {1, 0, 0.8f},
            {2, 3, 0.2f}
        };
        
        auto packet = build_attention_packet(0, 0, 4, 4, NF_ATTENTION_THRESHOLD, entries);
        
        ParsedHeader header;
        REQUIRE(parse_header(packet, header) == ParseResult::OK);
        
        ParsedAttentionPattern pattern;
        ParseResult result = parse_attention_pattern(packet.data() + 32, header.payload_bytes, pattern);
        
        REQUIRE(result == ParseResult::OK);
        REQUIRE(pattern.entries.size() == 4);
        REQUIRE(pattern.entries[0].src_idx == 0);
        REQUIRE(pattern.entries[0].tgt_idx == 1);
        REQUIRE(pattern.entries[3].src_idx == 2);
        REQUIRE(pattern.entries[3].tgt_idx == 3);
    }
    
    SECTION("Truncated attention pattern returns error") {
        std::vector<uint8_t> payload(25, 0);
        *reinterpret_cast<uint16_t*>(payload.data() + 14) = 10;  // entry_count = 10
        
        ParsedAttentionPattern pattern;
        ParseResult result = parse_attention_pattern(payload.data(), payload.size(), pattern);
        
        REQUIRE(result == ParseResult::ERR_TRUNCATED_PAYLOAD);
    }
    
    SECTION("Buffer too short returns error") {
        uint8_t payload[10] = {0};
        
        ParsedAttentionPattern pattern;
        ParseResult result = parse_attention_pattern(payload, 10, pattern);
        
        REQUIRE(result == ParseResult::ERR_BUFFER_TOO_SHORT);
    }
}

TEST_CASE("Attention extraction functions", "[protocol]") {
    
    SECTION("Top-k extraction") {
        std::vector<float> weights = {
            0.1f, 0.2f, 0.3f,
            0.4f, 0.5f, 0.6f,
            0.7f, 0.8f, 0.9f
        };
        
        auto entries = extract_attention_top_k(weights.data(), 3, 3, 3);
        
        REQUIRE(entries.size() == 3);
        REQUIRE(entries[0].weight == Approx(0.9f));
        REQUIRE(entries[1].weight == Approx(0.8f));
        REQUIRE(entries[2].weight == Approx(0.7f));
    }
    
    SECTION("Top-k with threshold") {
        std::vector<float> weights = {
            0.1f, 0.2f, 0.3f,
            0.4f, 0.5f, 0.6f
        };
        
        auto entries = extract_attention_top_k(weights.data(), 2, 3, 10, 0.5f);
        
        REQUIRE(entries.size() == 2);  // Only 0.5 and 0.6 pass threshold
        REQUIRE(entries[0].weight >= 0.5f);
        REQUIRE(entries[1].weight >= 0.5f);
    }
    
    SECTION("Threshold extraction") {
        std::vector<float> weights = {
            0.1f, 0.5f, 0.2f,
            0.8f, 0.3f, 0.9f
        };
        
        auto entries = extract_attention_threshold(weights.data(), 2, 3, 0.5f);
        
        REQUIRE(entries.size() == 3);
        for (const auto& e : entries) {
            REQUIRE(e.weight >= 0.5f);
        }
    }
    
    SECTION("Threshold extraction - all below threshold") {
        std::vector<float> weights = {0.1f, 0.2f, 0.3f, 0.4f};
        
        auto entries = extract_attention_threshold(weights.data(), 2, 2, 0.5f);
        
        REQUIRE(entries.empty());
    }
    
    SECTION("Threshold extraction - all above threshold") {
        std::vector<float> weights = {0.6f, 0.7f, 0.8f, 0.9f};
        
        auto entries = extract_attention_threshold(weights.data(), 2, 2, 0.5f);
        
        REQUIRE(entries.size() == 4);
    }
}
