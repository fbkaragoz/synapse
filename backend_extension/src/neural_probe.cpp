#include "neural_probe.h"
#include "server.h"
#include "protocol.h"
#include "protocol_parser.h"
#include "ring_buffer.h"
#include <memory>
#include <iostream>
#include <cstring>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <vector>
#include <cmath>

static std::shared_ptr<RingBuffer> g_buffer;
static std::unique_ptr<Server> g_server;
static uint64_t g_seq = 0;
static std::atomic<float> g_threshold = 0.0f;
static std::atomic<uint32_t> g_accum_steps = 0;
static std::atomic<uint32_t> g_broadcast_interval = 0;
static std::atomic<uint32_t> g_max_sparse = 0;
static std::atomic<bool> g_use_v2 = true;

void set_threshold(float threshold) {
    g_threshold = threshold;
    std::cout << "[NeuralProbe] Threshold set to " << threshold << std::endl;
}

NF_Result set_accumulation_steps(uint32_t steps) {
    g_accum_steps = steps;
    std::cout << "[NeuralProbe] Accumulation steps set to " << steps << std::endl;
    return NF_OK;
}

NF_Result set_broadcast_interval(uint32_t interval_ms) {
    g_broadcast_interval = interval_ms;
    std::cout << "[NeuralProbe] Broadcast interval set to " << interval_ms << "ms" << std::endl;
    return NF_OK;
}

NF_Result set_max_sparse_points(uint32_t max_points) {
    g_max_sparse = max_points;
    std::cout << "[NeuralProbe] Max sparse points set to " << max_points << std::endl;
    return NF_OK;
}

void set_use_v2(bool use_v2) {
    g_use_v2 = use_v2;
    std::cout << "[NeuralProbe] V2 mode " << (use_v2 ? "enabled" : "disabled") << std::endl;
}

bool get_use_v2() {
    return g_use_v2.load();
}

static std::vector<uint8_t> build_control_packet(uint32_t opcode, uint32_t value_u32, float value_f32) {
    size_t payload_size = sizeof(NF_ControlPacketV1);
    size_t total_size = sizeof(NF_PacketHeader) + payload_size;
    std::vector<uint8_t> packet(total_size);
    uint8_t* raw = packet.data();

    NF_PacketHeader* hdr = reinterpret_cast<NF_PacketHeader*>(raw);
    hdr->magic = 0x574C464E;
    hdr->version = 1;
    hdr->msg_type = NF_MSG_CONTROL;
    hdr->flags = 0;
    hdr->seq = g_seq++;
    auto now = std::chrono::steady_clock::now();
    hdr->timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    hdr->payload_bytes = payload_size;

    NF_ControlPacketV1* ctrl = reinterpret_cast<NF_ControlPacketV1*>(raw + sizeof(NF_PacketHeader));
    ctrl->opcode = opcode;
    ctrl->value_u32 = value_u32;
    ctrl->value_f32 = value_f32;
    ctrl->reserved = 0;

    return packet;
}

// Layer Topology (Dynamic)
#include <map>
#include <mutex>
#include <cmath>

static std::map<uint32_t, NF_LayerInfo> g_layer_map;
static std::mutex g_layer_mutex;

void handle_control_message(const uint8_t* data, size_t len) {
    if (len < sizeof(NF_PacketHeader) + sizeof(NF_ControlPacketV1)) return;
    
    // We assume the header has already been stripped or we need to skip it?
    // The uWS message callback provides the full payload.
    // Let's assume passed data is the FULL WEBSOCKET MESSAGE.
    
    const NF_PacketHeader* hdr = reinterpret_cast<const NF_PacketHeader*>(data);
    if (hdr->magic != 0x574C464E) return;
    if (hdr->msg_type != NF_MSG_CONTROL) return;
    
    const NF_ControlPacketV1* ctrl = reinterpret_cast<const NF_ControlPacketV1*>(data + sizeof(NF_PacketHeader));
    
    // std::cout << "[NeuralProbe] Received Opcode: " << ctrl->opcode << std::endl;
    
    switch (ctrl->opcode) {
        case NF_OP_SET_THRESHOLD:
            set_threshold(ctrl->value_f32);
            break;
        case NF_OP_SET_ACCUMULATION_STEPS:
            g_accum_steps = ctrl->value_u32;
            std::cout << "[NeuralProbe] Accumulation Steps set to " << g_accum_steps << std::endl;
            break;
        case NF_OP_SET_BROADCAST_INTERVAL:
            g_broadcast_interval = ctrl->value_u32;
            std::cout << "[NeuralProbe] Broadcast Interval set to " << g_broadcast_interval << std::endl;
            break;
        case NF_OP_SET_MAX_SPARSE_POINTS:
            g_max_sparse = ctrl->value_u32;
            std::cout << "[NeuralProbe] Max Sparse set to " << g_max_sparse << std::endl;
            break;
        default:
            break;
    }
}

std::vector<uint8_t> get_state_snapshot() {
    return build_control_packet(NF_OP_SET_THRESHOLD, 0, g_threshold.load());
}

std::vector<std::vector<uint8_t>> get_state_snapshot_packets() {
    std::vector<std::vector<uint8_t>> packets;
    packets.reserve(4);
    packets.push_back(build_control_packet(NF_OP_SET_THRESHOLD, 0, g_threshold.load()));
    packets.push_back(build_control_packet(NF_OP_SET_ACCUMULATION_STEPS, g_accum_steps.load(), 0.0f));
    packets.push_back(build_control_packet(NF_OP_SET_BROADCAST_INTERVAL, g_broadcast_interval.load(), 0.0f));
    packets.push_back(build_control_packet(NF_OP_SET_MAX_SPARSE_POINTS, g_max_sparse.load(), 0.0f));
    return packets;
}

std::vector<uint8_t> get_model_meta_packet() {
    std::lock_guard<std::mutex> lock(g_layer_mutex);
    
    size_t layer_count = g_layer_map.size();
    size_t payload_size = sizeof(NF_ModelMetaPacket) + (layer_count * sizeof(NF_LayerInfo));
    size_t total_size = sizeof(NF_PacketHeader) + payload_size;
    
    std::vector<uint8_t> packet(total_size);
    uint8_t* raw = packet.data();
    
    // Header
    NF_PacketHeader* hdr = reinterpret_cast<NF_PacketHeader*>(raw);
    hdr->magic = 0x574C464E;
    hdr->version = 1;
    hdr->msg_type = NF_MSG_MODEL_META;
    hdr->flags = 0;
    hdr->seq = g_seq++;
    auto now = std::chrono::steady_clock::now();
    hdr->timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    hdr->payload_bytes = payload_size;
    
    // Payload
    NF_ModelMetaPacket* meta = reinterpret_cast<NF_ModelMetaPacket*>(raw + sizeof(NF_PacketHeader));
    meta->total_layers = (uint32_t)layer_count;
    
    NF_LayerInfo* infos = reinterpret_cast<NF_LayerInfo*>(raw + sizeof(NF_PacketHeader) + sizeof(NF_ModelMetaPacket));
    
    int i = 0;
    for (const auto& kv : g_layer_map) {
        infos[i] = kv.second;
        i++;
    }
    
    return packet;
}

NF_Result start_server(const std::string& host, int port) {
    if (g_server && g_server->is_running()) {
        std::cout << "[NeuralProbe] Server already running." << std::endl;
        return NF_OK;
    }

    g_buffer = std::make_shared<RingBuffer>(1000); 

    g_server = create_uwebsockets_server();
    g_server->start(port, g_buffer);
    std::cout << "[NeuralProbe] Server started on port " << port << std::endl;
    return NF_OK;
}

void stop_server() {
    if (g_server) {
        g_server->stop();
        g_server.reset();
        std::cout << "[NeuralProbe] Server stopped." << std::endl;
    }
    g_buffer.reset();
}

NF_Result log_activation(int layer_id, py::array_t<float> tensor) {
    if (!g_buffer) return NF_ERR_NOT_STARTED;

    py::buffer_info buf = tensor.request();

    if (buf.ndim < 1 || buf.ndim > 3) {
        return NF_ERR_INVALID_DIMS;
    }
    
    if (buf.ptr == nullptr) {
        return NF_ERR_NULL_POINTER;
    }

    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;

    bool added_new_layer = false;
    {
        std::lock_guard<std::mutex> lock(g_layer_mutex);
        if (g_layer_map.find(layer_id) == g_layer_map.end()) {
            NF_LayerInfo info;
            info.layer_id = (uint32_t)layer_id;
            info.neuron_count = (uint32_t)size;
            
            float side = std::sqrt((float)size);
            info.recommended_width = (uint16_t)std::ceil(side);
            info.recommended_height = (uint16_t)std::ceil((float)size / info.recommended_width);
            
            g_layer_map[layer_id] = info;
            added_new_layer = true;
            std::cout << "[NeuralProbe] Registered Layer " << layer_id 
                      << " (Size: " << size 
                      << ", Grid: " << info.recommended_width << "x" << info.recommended_height << ")" << std::endl;
        }
    }

    if (added_new_layer && g_buffer) {
        g_buffer->push(get_model_meta_packet());
    }

    float threshold = g_threshold.load();
    std::vector<NF_ActivationEntryF32V1> sparse_entries;
    sparse_entries.reserve(std::min(size, (size_t)1024));

    for (size_t i = 0; i < size; ++i) {
        float v = ptr[i];
        if (v > threshold) {
            sparse_entries.push_back({(uint32_t)i, v});
        }
    }

    if (g_use_v2.load()) {
        nf::Statistics stats;
        nf::compute_statistics(ptr, size, stats);
        
        nf::ParsedLayerSummaryV2 v2_summary = nf::statistics_to_v2_summary(
            static_cast<uint32_t>(layer_id),
            static_cast<uint32_t>(size),
            stats
        );
        
        auto packet = nf::build_layer_summary_packet_v2({v2_summary}, g_seq++, 0);
        g_buffer->push(std::move(packet));
    } else {
        float sum = 0.0f;
        float max_val = -1e9f;
        for (size_t i = 0; i < size; ++i) {
            float v = ptr[i];
            sum += v;
            if (v > max_val) max_val = v;
        }
        float mean = sum / size;
        
        size_t payload_size = sizeof(NF_LayerSummaryBatchV1) + sizeof(NF_LayerSummaryV1);
        size_t total_size = sizeof(NF_PacketHeader) + payload_size;
        
        std::vector<uint8_t> packet(total_size);
        uint8_t* raw = packet.data();
        
        NF_PacketHeader* hdr = reinterpret_cast<NF_PacketHeader*>(raw);
        hdr->magic = 0x574C464E; 
        hdr->version = 1;
        hdr->msg_type = NF_MSG_LAYER_SUMMARY_BATCH;
        hdr->flags = NF_FLAG_FP32;
        hdr->seq = g_seq++;
        auto now = std::chrono::steady_clock::now();
        hdr->timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        hdr->payload_bytes = payload_size;
        
        uint8_t* payload_ptr = raw + sizeof(NF_PacketHeader);
        NF_LayerSummaryBatchV1* batch = reinterpret_cast<NF_LayerSummaryBatchV1*>(payload_ptr);
        batch->count = 1; 
        batch->reserved = 0;
        
        NF_LayerSummaryV1* summary = reinterpret_cast<NF_LayerSummaryV1*>(payload_ptr + sizeof(NF_LayerSummaryBatchV1));
        summary->layer_id = (uint32_t)layer_id;
        summary->neuron_count = (uint32_t)size;
        summary->mean = mean;
        summary->max = max_val;
        
        g_buffer->push(std::move(packet));
    }

    if (!sparse_entries.empty()) {
        size_t entries_size = sparse_entries.size() * sizeof(NF_ActivationEntryF32V1);
        size_t payload_size = sizeof(NF_SparseActivationsV1) + entries_size;
        size_t total_size = sizeof(NF_PacketHeader) + payload_size;
        
        std::vector<uint8_t> packet(total_size);
        uint8_t* raw = packet.data();
        
        NF_PacketHeader* hdr = reinterpret_cast<NF_PacketHeader*>(raw);
        hdr->magic = 0x574C464E; 
        hdr->version = 1;
        hdr->msg_type = NF_MSG_SPARSE_ACTIVATIONS;
        hdr->flags = NF_FLAG_FP32;
        hdr->seq = g_seq++;
        auto now = std::chrono::steady_clock::now();
        hdr->timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        hdr->payload_bytes = payload_size;
        
        uint8_t* payload_ptr = raw + sizeof(NF_PacketHeader);
        NF_SparseActivationsV1* sparse_hdr = reinterpret_cast<NF_SparseActivationsV1*>(payload_ptr);
        sparse_hdr->layer_id = (uint32_t)layer_id;
        sparse_hdr->count = (uint32_t)sparse_entries.size();
        sparse_hdr->reserved0 = 0;
        sparse_hdr->reserved1 = 0;
        
        uint8_t* entries_ptr = payload_ptr + sizeof(NF_SparseActivationsV1);
        std::memcpy(entries_ptr, sparse_entries.data(), entries_size);
        
        g_buffer->push(std::move(packet));
    }
    
    return NF_OK;
}

static std::atomic<uint64_t> g_training_step{0};

void set_training_step(uint64_t step) {
    g_training_step = step;
}

uint64_t get_training_step() {
    return g_training_step.load();
}

static std::vector<NF_GradientSummaryV1> g_pending_gradients;
static std::mutex g_gradient_mutex;
static std::atomic<float> g_global_grad_norm{0.0f};
static std::atomic<int> g_gradient_count{0};

NF_Result log_gradient(int layer_id, py::array_t<float> grad, py::array_t<float> weight) {
    if (!g_buffer) return NF_ERR_NOT_STARTED;
    
    py::buffer_info grad_buf = grad.request();
    py::buffer_info weight_buf = weight.request();
    
    if (grad_buf.ndim < 1 || grad_buf.ndim > 2) {
        return NF_ERR_INVALID_DIMS;
    }
    if (weight_buf.ndim < 1 || weight_buf.ndim > 2) {
        return NF_ERR_INVALID_DIMS;
    }
    if (grad_buf.ptr == nullptr || weight_buf.ptr == nullptr) {
        return NF_ERR_NULL_POINTER;
    }
    
    float* grad_ptr = static_cast<float*>(grad_buf.ptr);
    float* weight_ptr = static_cast<float*>(weight_buf.ptr);
    size_t grad_size = grad_buf.size;
    size_t weight_size = weight_buf.size;
    
    double grad_sum = 0.0;
    double grad_sum_sq = 0.0;
    float grad_min = grad_ptr[0];
    float grad_max = grad_ptr[0];
    
    for (size_t i = 0; i < grad_size; ++i) {
        float v = grad_ptr[i];
        grad_sum += v;
        grad_sum_sq += static_cast<double>(v) * v;
        if (v < grad_min) grad_min = v;
        if (v > grad_max) grad_max = v;
    }
    
    float grad_mean = static_cast<float>(grad_sum / grad_size);
    float grad_l2_norm = static_cast<float>(std::sqrt(grad_sum_sq));
    
    double grad_var = 0.0;
    for (size_t i = 0; i < grad_size; ++i) {
        double diff = grad_ptr[i] - grad_mean;
        grad_var += diff * diff;
    }
    float grad_std = static_cast<float>(std::sqrt(grad_var / grad_size));
    
    double weight_sum_sq = 0.0;
    for (size_t i = 0; i < weight_size; ++i) {
        weight_sum_sq += static_cast<double>(weight_ptr[i]) * weight_ptr[i];
    }
    float weight_l2_norm = static_cast<float>(std::sqrt(weight_sum_sq));
    
    float grad_to_weight = (weight_l2_norm > 1e-10f) ? (grad_l2_norm / weight_l2_norm) : 0.0f;
    
    NF_GradientSummaryV1 summary;
    summary.layer_id = static_cast<uint32_t>(layer_id);
    summary.param_count = static_cast<uint32_t>(grad_size);
    summary.grad_mean = grad_mean;
    summary.grad_std = grad_std;
    summary.grad_min = grad_min;
    summary.grad_max = grad_max;
    summary.grad_l2_norm = grad_l2_norm;
    summary.weight_l2_norm = weight_l2_norm;
    summary.grad_to_weight = grad_to_weight;
    
    {
        std::lock_guard<std::mutex> lock(g_gradient_mutex);
        g_pending_gradients.push_back(summary);
        
        float old_norm = g_global_grad_norm.load();
        float new_norm = std::sqrt(old_norm * old_norm + grad_l2_norm * grad_l2_norm);
        g_global_grad_norm = new_norm;
        g_gradient_count++;
    }
    
    return NF_OK;
}

void flush_gradient_batch() {
    if (!g_buffer) return;
    
    std::vector<NF_GradientSummaryV1> gradients;
    float global_norm;
    uint32_t count;
    
    {
        std::lock_guard<std::mutex> lock(g_gradient_mutex);
        gradients = std::move(g_pending_gradients);
        g_pending_gradients.clear();
        global_norm = g_global_grad_norm.exchange(0.0f);
        count = static_cast<uint32_t>(g_gradient_count.exchange(0));
    }
    
    if (gradients.empty()) return;
    
    size_t payload_size = sizeof(NF_GradientBatchV1) + gradients.size() * sizeof(NF_GradientSummaryV1);
    size_t total_size = sizeof(NF_PacketHeader) + payload_size;
    
    std::vector<uint8_t> packet(total_size);
    uint8_t* raw = packet.data();
    
    NF_PacketHeader* hdr = reinterpret_cast<NF_PacketHeader*>(raw);
    hdr->magic = 0x574C464E;
    hdr->version = 1;
    hdr->msg_type = NF_MSG_GRADIENT_BATCH;
    hdr->flags = NF_FLAG_FP32;
    hdr->seq = g_seq++;
    auto now = std::chrono::steady_clock::now();
    hdr->timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    hdr->payload_bytes = static_cast<uint32_t>(payload_size);
    
    uint8_t* payload_ptr = raw + sizeof(NF_PacketHeader);
    NF_GradientBatchV1* batch = reinterpret_cast<NF_GradientBatchV1*>(payload_ptr);
    batch->count = static_cast<uint32_t>(gradients.size());
    batch->training_step = static_cast<uint32_t>(g_training_step.load());
    batch->global_grad_norm = global_norm;
    
    NF_GradientSummaryV1* summaries = reinterpret_cast<NF_GradientSummaryV1*>(payload_ptr + sizeof(NF_GradientBatchV1));
    std::memcpy(summaries, gradients.data(), gradients.size() * sizeof(NF_GradientSummaryV1));
    
    g_buffer->push(std::move(packet));
}
