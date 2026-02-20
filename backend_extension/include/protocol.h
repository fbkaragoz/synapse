#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <cstdint>

enum NF_Result : uint32_t {
  NF_OK                    = 0,
  NF_ERR_NOT_STARTED       = 1,
  NF_ERR_INVALID_DIMS      = 2,
  NF_ERR_BUFFER_FULL       = 3,
  NF_ERR_NULL_POINTER      = 4,
  NF_ERR_LAYER_NOT_FOUND   = 5,
  NF_ERR_INVALID_THRESHOLD = 6,
};

// WIRE FORMAT: little-endian, packed-by-definition
// We use manual serialization or ensure compiler does not pad these specific structs.
// For standard types (u32, u64), alignment usually works out, but effectively we treat the wire format as byte-packed.

enum NF_MsgType : uint16_t {
  NF_MSG_LAYER_SUMMARY_BATCH    = 1,   // macro LOD (V1)
  NF_MSG_SPARSE_ACTIVATIONS     = 2,   // micro LOD
  NF_MSG_CONTROL                = 3,   // frontend->backend config
  NF_MSG_MODEL_META             = 4,   // topology
  NF_MSG_GRADIENT_BATCH         = 5,   // gradient statistics
  NF_MSG_LAYER_SUMMARY_BATCH_V2 = 6,   // extended statistics (64 bytes each)
};

enum NF_LayerFlags : uint32_t {
  NF_LAYER_FLAG_NONE       = 0,
  NF_LAYER_FLAG_SATURATED  = 1u << 0,  // Many values at bounds
  NF_LAYER_FLAG_DEAD       = 1u << 1,  // High zero ratio
  NF_LAYER_FLAG_EXPLODING  = 1u << 2,  // Extreme max/mean ratio
  NF_LAYER_FLAG_ANOMALY    = 1u << 3,  // Statistical anomaly detected
};

enum NF_Flags : uint32_t {
  NF_FLAG_NONE       = 0,
  NF_FLAG_FP16       = 1u << 0,  // values are IEEE 754 half in payload (if supported)
  NF_FLAG_FP32       = 1u << 1,  // values are float32 in payload
  NF_FLAG_COMPRESSED = 1u << 2,  // reserved (e.g., zstd)
};

// Header size: 32 bytes
struct __attribute__((packed)) NF_PacketHeader {
  uint32_t magic;          // 0x574C464E ("NFLW" little-endian)
  uint16_t version;        // 1
  uint16_t msg_type;       // NF_MsgType
  uint32_t flags;          // NF_Flags
  uint64_t seq;            // monotonically increasing
  uint64_t timestamp_ns;   // CLOCK_MONOTONIC / steady_clock ns
  uint32_t payload_bytes;  // bytes immediately following this header
};

// Payload: Layer Summary Batch (macro LOD)
struct __attribute__((packed)) NF_LayerSummaryV1 {
  uint32_t layer_id;     
  uint32_t neuron_count; 
  float    mean;         
  float    max;          
};

struct __attribute__((packed)) NF_LayerSummaryBatchV1 {
  uint32_t count;        // number of summaries that follow
  uint32_t reserved;     // 0
  // NF_LayerSummaryV1 summaries[count];
};

// Extended Layer Summary V2 (64 bytes) - 13 statistics per layer
struct __attribute__((packed)) NF_LayerSummaryV2 {
  uint32_t layer_id;        // 0-3
  uint32_t neuron_count;    // 4-7
  float    mean;            // 8-11
  float    std;             // 12-15
  float    min;             // 16-19
  float    max;             // 20-23
  float    l2_norm;         // 24-27
  float    zero_ratio;      // 28-31
  float    p5;              // 32-35
  float    p25;             // 36-39
  float    p75;             // 40-43
  float    p95;             // 44-47
  float    kurtosis;        // 48-51
  float    skewness;        // 52-55
  uint32_t flags;           // 56-59 (NF_LayerFlags)
  uint32_t reserved;        // 60-63
};

struct __attribute__((packed)) NF_LayerSummaryBatchV2 {
  uint32_t count;        // number of summaries that follow
  uint32_t version;      // always 2
  // NF_LayerSummaryV2 summaries[count];
};

// Payload: Sparse Activations (micro LOD)
struct __attribute__((packed)) NF_SparseActivationsV1 {
  uint32_t layer_id;
  uint32_t count;        // number of entries that follow
  uint32_t reserved0;    // 0
  uint32_t reserved1;    // 0
  // entries follow
};

struct __attribute__((packed)) NF_ActivationEntryF32V1 {
  uint32_t neuron_idx;   // index within the layer
  float    value;        // activation value
};

// -------------------------------------------------------------
// Control Packet (frontend -> backend, or backend -> frontend state)
// -------------------------------------------------------------
enum NF_ControlOpcode : uint32_t {
    NF_OP_SET_THRESHOLD            = 1,
    NF_OP_SET_ACCUMULATION_STEPS   = 2,
    NF_OP_SET_BROADCAST_INTERVAL   = 3,
    NF_OP_SET_MAX_SPARSE_POINTS    = 4,
    NF_OP_STATE_SNAPSHOT           = 100, // Backend -> Frontend config sync
    NF_OP_ACK                      = 101, // Backend -> Frontend
};

struct __attribute__((packed)) NF_ControlPacketV1 {
  uint32_t opcode;
  uint32_t value_u32;
  float    value_f32;
  uint32_t reserved;
};

// -------------------------------------------------------------
// Model Meta Packet (backend -> frontend, once on connect)
// -------------------------------------------------------------
struct __attribute__((packed)) NF_LayerInfo {
    uint32_t layer_id;
    uint32_t neuron_count;
    uint16_t recommended_width;
    uint16_t recommended_height;
};

// Header PayloadBytes will cover the size of this struct + (N * sizeof(NF_LayerInfo))
struct __attribute__((packed)) NF_ModelMetaPacket {
    uint32_t total_layers;
    // Followed by NF_LayerInfo layers[total_layers]
};

// -------------------------------------------------------------
// Gradient Batch (backward pass statistics)
// -------------------------------------------------------------
struct __attribute__((packed)) NF_GradientSummaryV1 {
  uint32_t layer_id;
  uint32_t param_count;       // number of parameters aggregated
  float    grad_mean;
  float    grad_std;
  float    grad_min;
  float    grad_max;
  float    grad_l2_norm;      // ||∇L||
  float    weight_l2_norm;    // ||W||
  float    grad_to_weight;    // ||∇L|| / ||W|| - update magnitude
};

struct __attribute__((packed)) NF_GradientBatchV1 {
  uint32_t count;             // number of gradient summaries
  uint32_t training_step;     // global step counter
  float    global_grad_norm;  // total gradient norm (after clipping)
  // Followed by NF_GradientSummaryV1 gradients[count]
};

#endif
