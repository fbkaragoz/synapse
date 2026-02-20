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

struct ParsedLayerSummaryV2 {
    uint32_t layer_id;
    uint32_t neuron_count;
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
    uint32_t flags;
};

struct ParsedLayerSummaryBatchV2 {
    std::vector<ParsedLayerSummaryV2> summaries;
};

inline ParseResult parse_layer_summary_batch_v2(const uint8_t* payload, size_t payload_len, ParsedLayerSummaryBatchV2& out) {
    if (payload_len < 8) {
        return ParseResult::ERR_BUFFER_TOO_SHORT;
    }
    
    uint32_t count = *reinterpret_cast<const uint32_t*>(payload);
    
    const size_t entry_size = 64;
    const size_t required_size = 8 + (count * entry_size);
    if (payload_len < required_size) {
        return ParseResult::ERR_TRUNCATED_PAYLOAD;
    }
    
    out.summaries.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* entry = payload + 8 + (i * entry_size);
        ParsedLayerSummaryV2& s = out.summaries[i];
        s.layer_id = *reinterpret_cast<const uint32_t*>(entry);
        s.neuron_count = *reinterpret_cast<const uint32_t*>(entry + 4);
        s.mean = *reinterpret_cast<const float*>(entry + 8);
        s.std = *reinterpret_cast<const float*>(entry + 12);
        s.min = *reinterpret_cast<const float*>(entry + 16);
        s.max = *reinterpret_cast<const float*>(entry + 20);
        s.l2_norm = *reinterpret_cast<const float*>(entry + 24);
        s.zero_ratio = *reinterpret_cast<const float*>(entry + 28);
        s.p5 = *reinterpret_cast<const float*>(entry + 32);
        s.p25 = *reinterpret_cast<const float*>(entry + 36);
        s.p75 = *reinterpret_cast<const float*>(entry + 40);
        s.p95 = *reinterpret_cast<const float*>(entry + 44);
        s.kurtosis = *reinterpret_cast<const float*>(entry + 48);
        s.skewness = *reinterpret_cast<const float*>(entry + 52);
        s.flags = *reinterpret_cast<const uint32_t*>(entry + 56);
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

class WelfordAccumulator {
public:
    WelfordAccumulator() : count_(0), mean_(0.0), m2_(0.0), min_(1e30f), max_(-1e30f), 
                           sum_sq_(0.0), zero_count_(0) {}
    
    void add(float value) {
        count_++;
        double delta = value - mean_;
        mean_ += delta / count_;
        double delta2 = value - mean_;
        m2_ += delta * delta2;
        
        if (value < min_) min_ = value;
        if (value > max_) max_ = value;
        sum_sq_ += static_cast<double>(value) * value;
        if (value == 0.0f) zero_count_++;
        
        values_.push_back(value);
    }
    
    void add(const float* data, size_t n) {
        values_.reserve(values_.size() + n);
        for (size_t i = 0; i < n; ++i) {
            add(data[i]);
        }
    }
    
    void reset() {
        count_ = 0;
        mean_ = 0.0;
        m2_ = 0.0;
        min_ = 1e30f;
        max_ = -1e30f;
        sum_sq_ = 0.0;
        zero_count_ = 0;
        values_.clear();
    }
    
    Statistics get_statistics() const {
        Statistics out;
        if (count_ == 0) {
            out = {};
            return out;
        }
        
        out.mean = static_cast<float>(mean_);
        out.min = min_;
        out.max = max_;
        out.l2_norm = static_cast<float>(std::sqrt(sum_sq_));
        out.zero_ratio = static_cast<float>(static_cast<double>(zero_count_) / count_);
        
        double variance = (count_ > 1) ? m2_ / count_ : 0.0;
        out.std = static_cast<float>(std::sqrt(variance));
        
        std::vector<float> sorted = values_;
        std::sort(sorted.begin(), sorted.end());
        
        auto percentile = [&sorted, this](double p) -> float {
            if (count_ == 1) return sorted[0];
            double idx = (count_ - 1) * p / 100.0;
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
            for (float v : values_) {
                double diff = (v - mean_) / out.std;
                m3 += diff * diff * diff;
                m4 += diff * diff * diff * diff;
            }
            m3 /= count_;
            m4 /= count_;
            out.skewness = static_cast<float>(m3);
            out.kurtosis = static_cast<float>(m4 - 3.0);
        } else {
            out.skewness = 0.0f;
            out.kurtosis = 0.0f;
        }
        
        return out;
    }
    
    size_t count() const { return count_; }
    double mean() const { return mean_; }
    double variance() const { return (count_ > 1) ? m2_ / count_ : 0.0; }
    
private:
    size_t count_;
    double mean_;
    double m2_;
    float min_;
    float max_;
    double sum_sq_;
    size_t zero_count_;
    std::vector<float> values_;
};

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

inline std::vector<uint8_t> build_layer_summary_packet_v2(
    const std::vector<ParsedLayerSummaryV2>& summaries,
    uint64_t seq = 0,
    uint64_t timestamp_ns = 0
) {
    const size_t payload_size = 8 + summaries.size() * 64;
    std::vector<uint8_t> payload(payload_size);
    
    *reinterpret_cast<uint32_t*>(payload.data()) = static_cast<uint32_t>(summaries.size());
    *reinterpret_cast<uint32_t*>(payload.data() + 4) = 2;  // version
    
    for (size_t i = 0; i < summaries.size(); ++i) {
        uint8_t* entry = payload.data() + 8 + i * 64;
        const ParsedLayerSummaryV2& s = summaries[i];
        *reinterpret_cast<uint32_t*>(entry) = s.layer_id;
        *reinterpret_cast<uint32_t*>(entry + 4) = s.neuron_count;
        *reinterpret_cast<float*>(entry + 8) = s.mean;
        *reinterpret_cast<float*>(entry + 12) = s.std;
        *reinterpret_cast<float*>(entry + 16) = s.min;
        *reinterpret_cast<float*>(entry + 20) = s.max;
        *reinterpret_cast<float*>(entry + 24) = s.l2_norm;
        *reinterpret_cast<float*>(entry + 28) = s.zero_ratio;
        *reinterpret_cast<float*>(entry + 32) = s.p5;
        *reinterpret_cast<float*>(entry + 36) = s.p25;
        *reinterpret_cast<float*>(entry + 40) = s.p75;
        *reinterpret_cast<float*>(entry + 44) = s.p95;
        *reinterpret_cast<float*>(entry + 48) = s.kurtosis;
        *reinterpret_cast<float*>(entry + 52) = s.skewness;
        *reinterpret_cast<uint32_t*>(entry + 56) = s.flags;
        *reinterpret_cast<uint32_t*>(entry + 60) = 0;  // reserved
    }
    
    return build_packet(NF_MSG_LAYER_SUMMARY_BATCH_V2, NF_FLAG_FP32, seq, timestamp_ns,
                        payload.data(), static_cast<uint32_t>(payload_size));
}

inline ParsedLayerSummaryV2 statistics_to_v2_summary(
    uint32_t layer_id,
    uint32_t neuron_count,
    const Statistics& stats
) {
    ParsedLayerSummaryV2 s;
    s.layer_id = layer_id;
    s.neuron_count = neuron_count;
    s.mean = stats.mean;
    s.std = stats.std;
    s.min = stats.min;
    s.max = stats.max;
    s.l2_norm = stats.l2_norm;
    s.zero_ratio = stats.zero_ratio;
    s.p5 = stats.p5;
    s.p25 = stats.p25;
    s.p75 = stats.p75;
    s.p95 = stats.p95;
    s.kurtosis = stats.kurtosis;
    s.skewness = stats.skewness;
    
    s.flags = NF_LAYER_FLAG_NONE;
    if (stats.zero_ratio > 0.5f) s.flags |= NF_LAYER_FLAG_DEAD;
    if (stats.max > 100.0f * (std::abs(stats.mean) + 1e-10f)) s.flags |= NF_LAYER_FLAG_EXPLODING;
    
    return s;
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

struct ParsedGradientSummary {
    uint32_t layer_id;
    uint32_t param_count;
    float grad_mean;
    float grad_std;
    float grad_min;
    float grad_max;
    float grad_l2_norm;
    float weight_l2_norm;
    float grad_to_weight;
};

struct ParsedGradientBatch {
    uint32_t training_step;
    float global_grad_norm;
    std::vector<ParsedGradientSummary> gradients;
};

inline ParseResult parse_gradient_batch(const uint8_t* payload, size_t payload_len, ParsedGradientBatch& out) {
    if (payload_len < 12) {
        return ParseResult::ERR_BUFFER_TOO_SHORT;
    }
    
    uint32_t count = *reinterpret_cast<const uint32_t*>(payload);
    out.training_step = *reinterpret_cast<const uint32_t*>(payload + 4);
    out.global_grad_norm = *reinterpret_cast<const float*>(payload + 8);
    
    const size_t entry_size = 36;
    const size_t required_size = 12 + (count * entry_size);
    if (payload_len < required_size) {
        return ParseResult::ERR_TRUNCATED_PAYLOAD;
    }
    
    out.gradients.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* entry = payload + 12 + (i * entry_size);
        ParsedGradientSummary& s = out.gradients[i];
        s.layer_id = *reinterpret_cast<const uint32_t*>(entry);
        s.param_count = *reinterpret_cast<const uint32_t*>(entry + 4);
        s.grad_mean = *reinterpret_cast<const float*>(entry + 8);
        s.grad_std = *reinterpret_cast<const float*>(entry + 12);
        s.grad_min = *reinterpret_cast<const float*>(entry + 16);
        s.grad_max = *reinterpret_cast<const float*>(entry + 20);
        s.grad_l2_norm = *reinterpret_cast<const float*>(entry + 24);
        s.weight_l2_norm = *reinterpret_cast<const float*>(entry + 28);
        s.grad_to_weight = *reinterpret_cast<const float*>(entry + 32);
    }
    
    return ParseResult::OK;
}

struct GradientStats {
    float mean;
    float std;
    float min;
    float max;
    float l2_norm;
};

inline void compute_gradient_stats(const float* data, size_t count, GradientStats& out) {
    if (count == 0) {
        out = {};
        return;
    }
    
    double sum = 0.0;
    out.min = data[0];
    out.max = data[0];
    double sum_sq = 0.0;
    
    for (size_t i = 0; i < count; ++i) {
        float v = data[i];
        sum += v;
        sum_sq += static_cast<double>(v) * v;
        if (v < out.min) out.min = v;
        if (v > out.max) out.max = v;
    }
    
    out.mean = static_cast<float>(sum / count);
    out.l2_norm = static_cast<float>(std::sqrt(sum_sq));
    
    double variance = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double diff = data[i] - out.mean;
        variance += diff * diff;
    }
    variance /= count;
    out.std = static_cast<float>(std::sqrt(variance));
}

inline std::vector<uint8_t> build_gradient_batch_packet(
    const std::vector<ParsedGradientSummary>& gradients,
    uint32_t training_step,
    float global_grad_norm,
    uint64_t seq = 0,
    uint64_t timestamp_ns = 0
) {
    const size_t payload_size = 12 + gradients.size() * 36;
    std::vector<uint8_t> payload(payload_size);
    
    *reinterpret_cast<uint32_t*>(payload.data()) = static_cast<uint32_t>(gradients.size());
    *reinterpret_cast<uint32_t*>(payload.data() + 4) = training_step;
    *reinterpret_cast<float*>(payload.data() + 8) = global_grad_norm;
    
    for (size_t i = 0; i < gradients.size(); ++i) {
        uint8_t* entry = payload.data() + 12 + i * 36;
        const ParsedGradientSummary& s = gradients[i];
        *reinterpret_cast<uint32_t*>(entry) = s.layer_id;
        *reinterpret_cast<uint32_t*>(entry + 4) = s.param_count;
        *reinterpret_cast<float*>(entry + 8) = s.grad_mean;
        *reinterpret_cast<float*>(entry + 12) = s.grad_std;
        *reinterpret_cast<float*>(entry + 16) = s.grad_min;
        *reinterpret_cast<float*>(entry + 20) = s.grad_max;
        *reinterpret_cast<float*>(entry + 24) = s.grad_l2_norm;
        *reinterpret_cast<float*>(entry + 28) = s.weight_l2_norm;
        *reinterpret_cast<float*>(entry + 32) = s.grad_to_weight;
    }
    
    return build_packet(NF_MSG_GRADIENT_BATCH, NF_FLAG_FP32, seq, timestamp_ns,
                        payload.data(), static_cast<uint32_t>(payload_size));
}

struct ParsedAttentionEntry {
    uint16_t src_idx;
    uint16_t tgt_idx;
    float weight;
};

struct ParsedAttentionPattern {
    uint32_t layer_id;
    uint32_t head_id;
    uint16_t seq_len;
    uint16_t tgt_len;
    uint8_t mode;
    std::vector<ParsedAttentionEntry> entries;
};

inline ParseResult parse_attention_pattern(const uint8_t* payload, size_t payload_len, ParsedAttentionPattern& out) {
    if (payload_len < 20) {
        return ParseResult::ERR_BUFFER_TOO_SHORT;
    }
    
    out.layer_id = *reinterpret_cast<const uint32_t*>(payload);
    out.head_id = *reinterpret_cast<const uint32_t*>(payload + 4);
    out.seq_len = *reinterpret_cast<const uint16_t*>(payload + 8);
    out.tgt_len = *reinterpret_cast<const uint16_t*>(payload + 10);
    out.mode = payload[12];
    uint16_t entry_count = *reinterpret_cast<const uint16_t*>(payload + 14);
    
    const size_t entry_size = 8;
    const size_t required_size = 20 + (entry_count * entry_size);
    if (payload_len < required_size) {
        return ParseResult::ERR_TRUNCATED_PAYLOAD;
    }
    
    out.entries.resize(entry_count);
    for (uint16_t i = 0; i < entry_count; ++i) {
        const uint8_t* entry = payload + 20 + (i * entry_size);
        out.entries[i].src_idx = *reinterpret_cast<const uint16_t*>(entry);
        out.entries[i].tgt_idx = *reinterpret_cast<const uint16_t*>(entry + 2);
        out.entries[i].weight = *reinterpret_cast<const float*>(entry + 4);
    }
    
    return ParseResult::OK;
}

inline std::vector<ParsedAttentionEntry> extract_attention_top_k(
    const float* weights,
    uint16_t seq_len,
    uint16_t tgt_len,
    uint16_t k,
    float threshold = 0.0f
) {
    std::vector<std::pair<float, size_t>> weighted_indices;
    weighted_indices.reserve(static_cast<size_t>(seq_len) * tgt_len);
    
    for (uint16_t s = 0; s < seq_len; ++s) {
        for (uint16_t t = 0; t < tgt_len; ++t) {
            float w = weights[static_cast<size_t>(s) * tgt_len + t];
            if (w >= threshold) {
                weighted_indices.push_back({w, static_cast<size_t>(s) * tgt_len + t});
            }
        }
    }
    
    std::partial_sort(
        weighted_indices.begin(),
        weighted_indices.begin() + std::min(static_cast<size_t>(k), weighted_indices.size()),
        weighted_indices.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    std::vector<ParsedAttentionEntry> entries;
    size_t count = std::min(static_cast<size_t>(k), weighted_indices.size());
    entries.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        size_t idx = weighted_indices[i].second;
        entries.push_back({
            static_cast<uint16_t>(idx / tgt_len),
            static_cast<uint16_t>(idx % tgt_len),
            weighted_indices[i].first
        });
    }
    
    return entries;
}

inline std::vector<ParsedAttentionEntry> extract_attention_threshold(
    const float* weights,
    uint16_t seq_len,
    uint16_t tgt_len,
    float threshold
) {
    std::vector<ParsedAttentionEntry> entries;
    
    for (uint16_t s = 0; s < seq_len; ++s) {
        for (uint16_t t = 0; t < tgt_len; ++t) {
            float w = weights[static_cast<size_t>(s) * tgt_len + t];
            if (w >= threshold) {
                entries.push_back({s, t, w});
            }
        }
    }
    
    return entries;
}

inline std::vector<uint8_t> build_attention_packet(
    uint32_t layer_id,
    uint32_t head_id,
    uint16_t seq_len,
    uint16_t tgt_len,
    uint8_t mode,
    const std::vector<ParsedAttentionEntry>& entries,
    uint64_t seq = 0,
    uint64_t timestamp_ns = 0
) {
    const size_t payload_size = 20 + entries.size() * 8;
    std::vector<uint8_t> payload(payload_size);
    
    *reinterpret_cast<uint32_t*>(payload.data()) = layer_id;
    *reinterpret_cast<uint32_t*>(payload.data() + 4) = head_id;
    *reinterpret_cast<uint16_t*>(payload.data() + 8) = seq_len;
    *reinterpret_cast<uint16_t*>(payload.data() + 10) = tgt_len;
    payload[12] = mode;
    payload[13] = 0;
    *reinterpret_cast<uint16_t*>(payload.data() + 14) = static_cast<uint16_t>(entries.size());
    *reinterpret_cast<uint32_t*>(payload.data() + 16) = 0;
    
    for (size_t i = 0; i < entries.size(); ++i) {
        uint8_t* entry = payload.data() + 20 + i * 8;
        *reinterpret_cast<uint16_t*>(entry) = entries[i].src_idx;
        *reinterpret_cast<uint16_t*>(entry + 2) = entries[i].tgt_idx;
        *reinterpret_cast<float*>(entry + 4) = entries[i].weight;
    }
    
    return build_packet(NF_MSG_ATTENTION_PATTERN, NF_FLAG_FP32, seq, timestamp_ns,
                        payload.data(), static_cast<uint32_t>(payload_size));
}

}

#endif
