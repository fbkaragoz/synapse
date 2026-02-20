#ifndef PROTOCOL_PARSER_H
#define PROTOCOL_PARSER_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <optional>
#include <cmath>
#include <algorithm>
#include "protocol.h"

namespace nf {

enum class ParseResult {
    OK = 0,
    ERR_BUFFER_TOO_SHORT = 1,
    ERR_INVALID_MAGIC = 2,
    ERR_INVALID_VERSION = 3,
    ERR_TRUNCATED_PAYLOAD = 4,
    ERR_INVALID_MSG_TYPE = 5,
};

struct ParsedHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t msg_type;
    uint32_t flags;
    uint64_t seq;
    uint64_t timestamp_ns;
    uint32_t payload_bytes;
};

inline ParseResult parse_header(const uint8_t* data, size_t len, ParsedHeader& out) {
    if (len < 32) {
        return ParseResult::ERR_BUFFER_TOO_SHORT;
    }
    
    out.magic = *reinterpret_cast<const uint32_t*>(data);
    if (out.magic != 0x574C464E) {
        return ParseResult::ERR_INVALID_MAGIC;
    }
    
    out.version = *reinterpret_cast<const uint16_t*>(data + 4);
    if (out.version != 1) {
        return ParseResult::ERR_INVALID_VERSION;
    }
    
    out.msg_type = *reinterpret_cast<const uint16_t*>(data + 6);
    out.flags = *reinterpret_cast<const uint32_t*>(data + 8);
    out.seq = *reinterpret_cast<const uint64_t*>(data + 12);
    out.timestamp_ns = *reinterpret_cast<const uint64_t*>(data + 20);
    out.payload_bytes = *reinterpret_cast<const uint32_t*>(data + 28);
    
    if (len < 32 + out.payload_bytes) {
        return ParseResult::ERR_TRUNCATED_PAYLOAD;
    }
    
    return ParseResult::OK;
}

inline ParseResult parse_header(const std::vector<uint8_t>& data, ParsedHeader& out) {
    return parse_header(data.data(), data.size(), out);
}

struct ParsedLayerSummary {
    uint32_t layer_id;
    uint32_t neuron_count;
    float mean;
    float max;
};

struct ParsedLayerSummaryBatch {
    std::vector<ParsedLayerSummary> summaries;
};

inline ParseResult parse_layer_summary_batch(const uint8_t* payload, size_t payload_len, ParsedLayerSummaryBatch& out) {
    if (payload_len < 8) {
        return ParseResult::ERR_BUFFER_TOO_SHORT;
    }
    
    uint32_t count = *reinterpret_cast<const uint32_t*>(payload);
    
    const size_t entry_size = 16;
    const size_t required_size = 8 + (count * entry_size);
    if (payload_len < required_size) {
        return ParseResult::ERR_TRUNCATED_PAYLOAD;
    }
    
    out.summaries.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* entry = payload + 8 + (i * entry_size);
        out.summaries[i].layer_id = *reinterpret_cast<const uint32_t*>(entry);
        out.summaries[i].neuron_count = *reinterpret_cast<const uint32_t*>(entry + 4);
        out.summaries[i].mean = *reinterpret_cast<const float*>(entry + 8);
        out.summaries[i].max = *reinterpret_cast<const float*>(entry + 12);
    }
    
    return ParseResult::OK;
}

struct ParsedSparseActivation {
    uint32_t layer_id;
    std::vector<uint32_t> indices;
    std::vector<float> values;
};

inline ParseResult parse_sparse_activation(const uint8_t* payload, size_t payload_len, ParsedSparseActivation& out) {
    if (payload_len < 16) {
        return ParseResult::ERR_BUFFER_TOO_SHORT;
    }
    
    out.layer_id = *reinterpret_cast<const uint32_t*>(payload);
    uint32_t count = *reinterpret_cast<const uint32_t*>(payload + 4);
    
    const size_t entry_size = 8;
    const size_t required_size = 16 + (count * entry_size);
    if (payload_len < required_size) {
        return ParseResult::ERR_TRUNCATED_PAYLOAD;
    }
    
    out.indices.resize(count);
    out.values.resize(count);
    
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* entry = payload + 16 + (i * entry_size);
        out.indices[i] = *reinterpret_cast<const uint32_t*>(entry);
        out.values[i] = *reinterpret_cast<const float*>(entry + 4);
    }
    
    return ParseResult::OK;
}

struct ParsedControlPacket {
    uint32_t opcode;
    uint32_t value_u32;
    float value_f32;
};

inline ParseResult parse_control_packet(const uint8_t* payload, size_t payload_len, ParsedControlPacket& out) {
    if (payload_len < 16) {
        return ParseResult::ERR_BUFFER_TOO_SHORT;
    }
    
    out.opcode = *reinterpret_cast<const uint32_t*>(payload);
    out.value_u32 = *reinterpret_cast<const uint32_t*>(payload + 4);
    out.value_f32 = *reinterpret_cast<const float*>(payload + 8);
    
    return ParseResult::OK;
}

struct Statistics {
    float mean;
    float std;
    float min;
    float max;
    float l2_norm;
    float zero_ratio;
    float p5;
    float p25;
    float p75;
    float p95;
    float kurtosis;
    float skewness;
};

inline void compute_statistics(const float* data, size_t count, Statistics& out) {
    if (count == 0) {
        out = {};
        return;
    }
    
    double sum = 0.0;
    out.min = data[0];
    out.max = data[0];
    size_t zero_count = 0;
    double sum_sq = 0.0;
    
    for (size_t i = 0; i < count; ++i) {
        float v = data[i];
        sum += v;
        sum_sq += static_cast<double>(v) * v;
        if (v < out.min) out.min = v;
        if (v > out.max) out.max = v;
        if (v == 0.0f) ++zero_count;
    }
    
    out.mean = static_cast<float>(sum / count);
    out.l2_norm = static_cast<float>(std::sqrt(sum_sq));
    out.zero_ratio = static_cast<float>(static_cast<double>(zero_count) / count);
    
    double variance = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double diff = data[i] - out.mean;
        variance += diff * diff;
    }
    variance /= count;
    out.std = static_cast<float>(std::sqrt(variance));
    
    std::vector<float> sorted(data, data + count);
    std::sort(sorted.begin(), sorted.end());
    
    auto percentile = [&sorted, count](double p) -> float {
        if (count == 1) return sorted[0];
        double idx = (count - 1) * p / 100.0;
        size_t lower = static_cast<size_t>(std::floor(idx));
        size_t upper = static_cast<size_t>(std::ceil(idx));
        double frac = idx - lower;
        return static_cast<float>(sorted[lower] * (1 - frac) + sorted[upper] * frac);
    };
    
    out.p5 = percentile(5);
    out.p25 = percentile(25);
    out.p75 = percentile(75);
    out.p95 = percentile(95);
    
    if (out.std > 0) {
        double m3 = 0.0;
        double m4 = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double diff = (data[i] - out.mean) / out.std;
            m3 += diff * diff * diff;
            m4 += diff * diff * diff * diff;
        }
        m3 /= count;
        m4 /= count;
        
        out.skewness = static_cast<float>(m3);
        out.kurtosis = static_cast<float>(m4 - 3.0);
    } else {
        out.skewness = 0.0f;
        out.kurtosis = 0.0f;
    }
}

inline void compute_statistics(const std::vector<float>& data, Statistics& out) {
    compute_statistics(data.data(), data.size(), out);
}

inline std::vector<uint8_t> build_packet(
    uint16_t msg_type,
    uint32_t flags,
    uint64_t seq,
    uint64_t timestamp_ns,
    const uint8_t* payload,
    uint32_t payload_size
) {
    std::vector<uint8_t> packet(32 + payload_size);
    
    *reinterpret_cast<uint32_t*>(packet.data()) = 0x574C464E;
    *reinterpret_cast<uint16_t*>(packet.data() + 4) = 1;
    *reinterpret_cast<uint16_t*>(packet.data() + 6) = msg_type;
    *reinterpret_cast<uint32_t*>(packet.data() + 8) = flags;
    *reinterpret_cast<uint64_t*>(packet.data() + 12) = seq;
    *reinterpret_cast<uint64_t*>(packet.data() + 20) = timestamp_ns;
    *reinterpret_cast<uint32_t*>(packet.data() + 28) = payload_size;
    
    if (payload_size > 0 && payload != nullptr) {
        std::memcpy(packet.data() + 32, payload, payload_size);
    }
    
    return packet;
}

inline std::vector<uint8_t> build_layer_summary_packet(
    const std::vector<ParsedLayerSummary>& summaries,
    uint64_t seq = 0,
    uint64_t timestamp_ns = 0
) {
    const size_t payload_size = 8 + summaries.size() * 16;
    std::vector<uint8_t> payload(payload_size);
    
    *reinterpret_cast<uint32_t*>(payload.data()) = static_cast<uint32_t>(summaries.size());
    
    for (size_t i = 0; i < summaries.size(); ++i) {
        uint8_t* entry = payload.data() + 8 + i * 16;
        *reinterpret_cast<uint32_t*>(entry) = summaries[i].layer_id;
        *reinterpret_cast<uint32_t*>(entry + 4) = summaries[i].neuron_count;
        *reinterpret_cast<float*>(entry + 8) = summaries[i].mean;
        *reinterpret_cast<float*>(entry + 12) = summaries[i].max;
    }
    
    return build_packet(NF_MSG_LAYER_SUMMARY_BATCH, NF_FLAG_FP32, seq, timestamp_ns, 
                        payload.data(), static_cast<uint32_t>(payload_size));
}

inline std::vector<uint8_t> build_control_packet(
    uint32_t opcode,
    uint32_t value_u32,
    float value_f32,
    uint64_t seq = 0,
    uint64_t timestamp_ns = 0
) {
    uint8_t payload[16];
    *reinterpret_cast<uint32_t*>(payload) = opcode;
    *reinterpret_cast<uint32_t*>(payload + 4) = value_u32;
    *reinterpret_cast<float*>(payload + 8) = value_f32;
    *reinterpret_cast<uint32_t*>(payload + 12) = 0;
    
    return build_packet(NF_MSG_CONTROL, 0, seq, timestamp_ns, payload, 16);
}

}

#endif
