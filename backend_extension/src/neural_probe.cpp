#include "neural_probe.h"
#include "server.h"
#include "protocol.h"
#include "ring_buffer.h"
#include <memory>
#include <iostream>
#include <cstring>
#include <chrono>
#include <atomic>
#include <algorithm> // for std::min
#include <vector>

// Global state
// In a real library, might wrap this in a singleton class
static std::shared_ptr<RingBuffer> g_buffer;
static std::unique_ptr<Server> g_server;
static uint64_t g_seq = 0;
static std::atomic<float> g_threshold = 0.0f; // Default capture everything > 0? Or just 0.0

void set_threshold(float threshold) {
    g_threshold = threshold;
    std::cout << "[NeuralProbe] Threshold set to " << threshold << std::endl;
}

// Global configuration
static std::atomic<uint32_t> g_accum_steps = 0;
static std::atomic<uint32_t> g_broadcast_interval = 0;
static std::atomic<uint32_t> g_max_sparse = 0;

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
    ctrl->opcode = NF_OP_SET_THRESHOLD;
    ctrl->value_f32 = g_threshold.load();
    ctrl->value_u32 = 0;
    ctrl->reserved = 0;
    
    return packet;
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

void start_server(const std::string& host, int port) {
    if (g_server && g_server->is_running()) {
        std::cout << "[NeuralProbe] Server already running." << std::endl;
        return;
    }

    // Initialize buffer with some capacity (e.g., 50MB worth of packets or Count)
    // 1000 items? Depends on frame size.
    g_buffer = std::make_shared<RingBuffer>(1000); 

    g_server = create_uwebsockets_server();
    // Host logic not fully used by uWS in this simple impl (binds to all interfaces usually), 
    // but port is used.
    g_server->start(port, g_buffer);
    std::cout << "[NeuralProbe] Server started on port " << port << std::endl;
}

void stop_server() {
    if (g_server) {
        g_server->stop();
        g_server.reset();
        std::cout << "[NeuralProbe] Server stopped." << std::endl;
    }
    g_buffer.reset();
}

// Helper to serialize data
void log_activation(int layer_id, py::array_t<float> tensor) {
    if (!g_buffer) return;

    // Request raw buffer info (fast, no copy yet)
    py::buffer_info buf = tensor.request();

    if (buf.ndim != 2 && buf.ndim != 3 && buf.ndim != 1) {
        return; 
    }

    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size; // Total elements

    // Auto-Register Layer Topology IF NEW
    bool added_new_layer = false;
    {
        std::lock_guard<std::mutex> lock(g_layer_mutex);
        if (g_layer_map.find(layer_id) == g_layer_map.end()) {
            NF_LayerInfo info;
            info.layer_id = (uint32_t)layer_id;
            info.neuron_count = (uint32_t)size;
            
            // Heuristic: Square-ish layout
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

    // If we discovered a new layer after clients connected, broadcast updated topology.
    if (added_new_layer && g_buffer) {
        g_buffer->push(get_model_meta_packet());
    }

    float sum = 0.0f;
    float max_val = -1e9f;
    
    // Collect Sparse Entries
    // We only collect if value > threshold
    float threshold = g_threshold.load();
    std::vector<NF_ActivationEntryF32V1> sparse_entries;
    // Pre-allocate to avoid reallocs if possible? 
    // Heuristic: if threshold is high, sparse_entries is small.
    sparse_entries.reserve(std::min(size, (size_t)1024)); 

    for (size_t i = 0; i < size; ++i) {
        float v = ptr[i];
        sum += v;
        if (v > max_val) max_val = v;
        
        if (v > threshold) {
            sparse_entries.push_back({(uint32_t)i, v});
        }
    }
    float mean = sum / size;

    // 2. Build Packet
    // We will send BOTH Summary and Sparse for now (simplest for Phase 3).
    // Or just one packet with Type = Sparse? 
    // Protocol has: MSG_LAYER_SUMMARY_BATCH and MSG_SPARSE_ACTIVATIONS.
    // Let's send 2 packets back-to-back? Or just Summary if sparse is empty?
    
    // SEND SUMMARY
    {
        size_t payload_size = sizeof(NF_LayerSummaryBatchV1) + sizeof(NF_LayerSummaryV1);
        size_t total_size = sizeof(NF_PacketHeader) + payload_size;
        
        std::vector<uint8_t> packet(total_size);
        uint8_t* raw = packet.data();
        
        // Header
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

    // SEND SPARSE (if any)
    // Limit sparse packet size? 10k entries = 80KB. Fine for localhost.
    if (!sparse_entries.empty()) {
        size_t entries_size = sparse_entries.size() * sizeof(NF_ActivationEntryF32V1);
        size_t payload_size = sizeof(NF_SparseActivationsV1) + entries_size;
        size_t total_size = sizeof(NF_PacketHeader) + payload_size;
        
        std::vector<uint8_t> packet(total_size);
        uint8_t* raw = packet.data();
        
        // Header
        NF_PacketHeader* hdr = reinterpret_cast<NF_PacketHeader*>(raw);
        hdr->magic = 0x574C464E; 
        hdr->version = 1;
        hdr->msg_type = NF_MSG_SPARSE_ACTIVATIONS;
        hdr->flags = NF_FLAG_FP32;
        hdr->seq = g_seq++; // distinct seq
        auto now = std::chrono::steady_clock::now();
        hdr->timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        hdr->payload_bytes = payload_size;
        
        // Payload
        uint8_t* payload_ptr = raw + sizeof(NF_PacketHeader);
        NF_SparseActivationsV1* sparse_hdr = reinterpret_cast<NF_SparseActivationsV1*>(payload_ptr);
        sparse_hdr->layer_id = (uint32_t)layer_id;
        sparse_hdr->count = (uint32_t)sparse_entries.size();
        sparse_hdr->reserved0 = 0;
        sparse_hdr->reserved1 = 0;
        
        // Copy entries
        uint8_t* entries_ptr = payload_ptr + sizeof(NF_SparseActivationsV1);
        std::memcpy(entries_ptr, sparse_entries.data(), entries_size);
        
        g_buffer->push(std::move(packet));
    }
}
