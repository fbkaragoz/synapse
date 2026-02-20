# Synapse Development Roadmap: Analytical Training Visualization

**Version**: 1.0  
**Created**: 2026-02-21  
**Status**: Active  

---

## Executive Summary

**Vision**: Transform Synapse from a basic activation visualizer into an analytical diagnostic tool that detects training pathologies (dead neurons, exploding gradients, vanishing signals) through mathematically grounded metrics.

**Use Case**: Research exploration for medium-scale models (1-10B params) with essential gradient statistics.

**Core Principle**: Every visual element must correspond to a mathematically meaningful statistic from the neural network.

---

## Part I: Current State vs. Required Capabilities

### Protocol Coverage Matrix

| Metric | Currently Captured | Required | Priority |
|--------|-------------------|----------|----------|
| Mean activation | Yes | Yes | - |
| Max activation | Yes | Yes | - |
| Min activation | No | Yes | P0 |
| Std deviation | No | Yes | P0 |
| L2 norm | No | Yes | P0 |
| Zero ratio | No | Yes | P0 |
| Percentiles (p5, p25, p75, p95) | No | Yes | P1 |
| Gradient norm | No | Yes | P0 |
| Kurtosis | No | Yes | P1 |
| Skewness | No | Yes | P1 |

### Implementation Gap Analysis

| Component | Status | Missing |
|-----------|--------|---------|
| Backend statistics | 2 metrics | 8+ metrics |
| Accumulation engine | Protocol only | Actual implementation |
| Gradient hooks | None | Backward hook registration |
| Test coverage | 0 tests | Unit + integration tests |
| Error handling | Silent fails | Status codes, validation |
| Memory management | Manual new | RAII, bounded pools |
| Layer selection | None | Whitelist/sampling |

---

## Part II: Mathematical Framework for Parametric Detection

### 2.1 Training Pathology Detection Functions

```
DEAD_NEURON_RATIO(layer) = count(activation == 0) / neuron_count
  Threshold: >0.5 indicates potential ReLU death cascade

SATURATION_RATIO(layer, bounds) = count(activation in bounds) / neuron_count
  For sigmoid: bounds = [0.95, 1.0] U [0.0, 0.05]
  For tanh: bounds = [0.9, 1.0] U [-1.0, -0.9]

EXPLOSION_INDICATOR(layer) = max / (mean + epsilon)
  Threshold: >100 suggests gradient/activation explosion

VANISHING_INDICATOR(layer_sequence) = product(mean_Li / mean_Li-1)
  Product of successive layer mean ratios
  Threshold: <0.1 indicates vanishing signal

KURTOSIS(layer) = E[(X - mu)^4] / sigma^4 - 3
  >3 indicates heavy tails (outliers)
  <0 indicates light tails (compressed)

SKEWNESS(layer) = E[(X - mu)^3] / sigma^3
  |skew| > 1 indicates asymmetric distribution
  Important for detecting systematic bias

LAYER_HEALTH_SCORE(layer) = weighted_sum([
  (1 - DEAD_NEURON_RATIO) x 0.3,
  (1 - SATURATION_RATIO) x 0.2,
  min(EXPLOSION_INDICATOR / 100, 1) x 0.25,
  min(|KURTOSIS| / 10, 1) x 0.15,
  min(|SKEWNESS| / 2, 1) x 0.10
])

UPDATE_MAGNITUDE(layer) = grad_l2_norm / (weight_l2_norm + epsilon)
  High ratio (>0.1): Potential instability
  Very low ratio (<1e-6): Learning may be stalled

GRADIENT_EXPLOSION(layer) = grad_max / (grad_mean + epsilon)
  >100 suggests outliers, possible explosion

GRADIENT_VANISHING(global) = global_grad_norm
  <1e-8 suggests vanishing gradients
```

### 2.2 Temporal Dynamics

```
TREND(stat, window) = (stat[t] - stat[t-window]) / window
  Positive: increasing, Negative: decreasing
  
VOLATILITY(stat, window) = std(stat[t-window:t])

ANOMALY_SCORE(stat) = |stat[t] - mu_window| / (sigma_window + epsilon)
  >2 indicates statistically significant deviation
```

### 2.3 Protocol Extension Specification

```c
// Extended Layer Summary (24 bytes -> 64 bytes)
struct __attribute__((packed)) NF_LayerSummaryV2 {
  uint32_t layer_id;        // 0-3
  uint32_t neuron_count;    // 4-7
  float    mean;            // 8-11
  float    std;             // 12-15  [NEW]
  float    min;             // 16-19  [NEW]
  float    max;             // 20-23
  float    l2_norm;         // 24-27  [NEW]
  float    zero_ratio;      // 28-31  [NEW]
  float    p5;              // 32-35  [NEW]
  float    p25;             // 36-39  [NEW]
  float    p75;             // 40-43  [NEW]
  float    p95;             // 44-47  [NEW]
  float    kurtosis;        // 48-51  [NEW]
  float    skewness;        // 52-55  [NEW]
  uint32_t flags;           // 56-59  [NEW] saturation flags, anomaly flags
  uint32_t reserved;        // 60-63
};

// Gradient Statistics Message (New)
// Message type: NF_MSG_GRADIENT_BATCH = 6
struct __attribute__((packed)) NF_GradientBatchV1 {
  uint32_t count;               // Number of gradient summaries
  uint32_t training_step;       // Global step counter
  float    global_grad_norm;    // Total gradient norm (clipped)
  // NF_GradientSummaryV1 gradients[count]
};

struct __attribute__((packed)) NF_GradientSummaryV1 {
  uint32_t layer_id;
  uint32_t param_count;         // Number of parameters aggregated
  float    grad_mean;
  float    grad_std;
  float    grad_min;
  float    grad_max;
  float    grad_l2_norm;        // ||grad||
  float    weight_l2_norm;      // ||W||
  float    grad_to_weight;      // ||grad|| / ||W||
  float    grad_to_weight_clip; // After gradient clipping
};

// Layer Selection Control (New)
// Opcode: NF_OP_SELECT_LAYERS = 10
struct NF_LayerSelectionV1 {
  uint32_t mode;         // 0=all, 1=whitelist, 2=blacklist, 3=sample
  uint32_t count;        // Number of layer IDs
  uint32_t layer_ids[];  // Layer IDs based on mode
};

// Attention Pattern (New)
// Message type: NF_MSG_ATTENTION = 5
struct __attribute__((packed)) NF_AttentionPatternV1 {
  uint32_t layer_id;
  uint32_t head_id;
  uint32_t seq_len;         // Source sequence length
  uint32_t tgt_len;         // Target sequence length
  uint8_t  pattern_type;    // 0=full, 1=top-k, 2=thresholded
  uint8_t  reserved[3];
  // Followed by attention entries
};

struct __attribute__((packed)) NF_AttentionEntryV1 {
  uint32_t src_idx;         // Query position
  uint32_t tgt_idx;         // Key position  
  float    weight;          // Attention weight [0, 1]
};
```

---

## Part III: Iteration Schedule

### Iteration 1A: Test Foundation (Week 1)

**Theme**: Correctness and confidence

**Status**: ✅ COMPLETED (2026-02-21)

| Task | Est. | Status |
|------|------|--------|
| Add Catch2 test framework to CMake | 2h | [x] |
| Implement protocol parsing tests | 4h | [x] |
| Implement ring buffer tests | 3h | [x] |
| Create golden packet fixtures | 2h | [x] |
| Error codes implementation | 3h | [x] |
| **Total** | **14h** | |

**Exit Criteria**: 
- [x] All tests pass (139 assertions in 18 test cases)
- [x] CI green (if CI configured)
- [x] Error codes documented

**Deliverables**:
- `tests/test_main.cpp` - Empty entry point for Catch2
- `tests/test_protocol.cpp` - Protocol parsing tests
- `tests/test_ring_buffer.cpp` - Concurrent queue tests
- `tests/test_statistics.cpp` - Statistical computation tests
- `tests/fixtures/golden_packets.h` - Fixture generators
- `include/protocol_parser.h` - Standalone protocol parser for testability
- Error code enum in `include/protocol.h`
- Updated Python API with error returns

**Test Suite Structure**:
```
backend_extension/tests/
├── test_protocol.cpp      # C++ protocol parsing tests
├── test_ring_buffer.cpp   # Concurrent queue behavior
├── test_statistics.cpp    # Statistical computation accuracy
├── test_integration.py    # Python end-to-end tests
└── fixtures/
    └── golden_packets.bin # Recorded packets for regression
```

**Key Test Cases**:

| Test | Input | Expected |
|------|-------|----------|
| ring_buffer_overflow | Push 1001 items to size-1000 buffer | Oldest dropped, newest retained |
| ring_buffer_concurrent | 4 threads push, 1 thread pop | No deadlock, no corruption |
| ring_buffer_stop | Call stop() while blocked on pop | Returns nullopt, doesn't hang |
| protocol_header_magic | Buffer with wrong magic | Parse returns error |
| protocol_truncated | Buffer shorter than header | Parse returns error |
| statistics_mean | [1.0, 2.0, 3.0, 4.0, 5.0] | mean = 3.0 |
| statistics_std | [1.0, 2.0, 3.0, 4.0, 5.0] | std approx 1.414 |
| statistics_percentile | 100 uniform values | p5 approx 5, p95 approx 95 |
| log_activation_invalid | 4D tensor | Returns error code, no crash |

**Error Handling API Change**:
```cpp
enum NF_Result {
  NF_OK = 0,
  NF_ERR_NOT_STARTED = 1,
  NF_ERR_INVALID_DIMS = 2,
  NF_ERR_BUFFER_FULL = 3,
  NF_ERR_NULL_POINTER = 4,
};

NF_Result log_activation(int layer_id, py::array_t<float> tensor);
```

---

### Iteration 1B: Gradient Capture (Week 2)

**Theme**: Essential backward pass statistics

**Status**: ✅ COMPLETED (2026-02-21)

| Task | Est. | Status |
|------|------|--------|
| Define gradient protocol | 2h | [x] |
| Implement log_gradient API | 4h | [x] |
| Create gradient hook example | 2h | [x] |
| Update WASM parser | 2h | [x] |
| Frontend gradient display | 4h | [ ] (deferred to Iteration 2) |
| **Total** | **14h** | |

**Exit Criteria**:
- [x] Gradient statistics captured and transmitted
- [x] WASM parser handles gradient messages
- [ ] Frontend displays gradient metrics (deferred)

**Deliverables**:
- `include/protocol.h` - NF_MSG_GRADIENT_BATCH, NF_GradientSummaryV1, NF_GradientBatchV1
- `include/protocol_parser.h` - Gradient parsing and building functions
- `src/neural_probe.cpp` - log_gradient, flush_gradient_batch, set_training_step
- `python/gradient_hook.py` - GradientCapture class for PyTorch
- `wasm_parser/src/protocol.rs` - Gradient batch parsing

**Python API**:
```python
import neural_probe
from gradient_hook import GradientCapture

# Register gradient hooks
capture = GradientCapture(model, layers=[0, 5, 10])
capture.register_hooks()

# In training loop
for step, batch in enumerate(dataloader):
    neural_probe.set_training_step(step)
    loss.backward()
    capture.flush()  # Sends gradient batch packet
    optimizer.step()
```

---

### Iteration 2: Extended Statistics (Week 3)

**Theme**: Analytical diagnostic capability

**Status**: ✅ COMPLETED (2026-02-21)

| Task | Est. | Status |
|------|------|--------|
| Protocol V2 extension | 2h | [x] |
| Min, std, L2 norm capture | 3h | [x] |
| Percentile computation | 3h | [x] |
| Kurtosis, skewness | 3h | [x] |
| Frontend distribution view | 4h | [ ] (deferred) |
| **Total** | **15h** | |

**Exit Criteria**:
- [x] All 13 statistics captured per layer
- [x] Protocol V2 backward compatible
- [ ] Distribution visualization working (deferred)

**Deliverables**:
- `include/protocol.h` - NF_MSG_LAYER_SUMMARY_BATCH_V2, NF_LayerSummaryV2 (64 bytes)
- `include/protocol_parser.h` - V2 parsing/building, statistics_to_v2_summary helper
- `src/neural_probe.cpp` - Extended statistics in log_activation, set_use_v2/get_use_v2
- `wasm_parser/src/protocol.rs` - LayerSummaryV2 struct and parsing

**New Python API**:
```python
import neural_probe

# Enable V2 mode (default: True)
neural_probe.set_use_v2(True)

# log_activation now sends 13 statistics:
# mean, std, min, max, l2_norm, zero_ratio, p5, p25, p75, p95, kurtosis, skewness, flags
result = neural_probe.log_activation(layer_id, tensor)
```

**Protocol V2 Summary (64 bytes)**:
```
layer_id (4) | neuron_count (4) | mean (4) | std (4)
min (4) | max (4) | l2_norm (4) | zero_ratio (4)
p5 (4) | p25 (4) | p75 (4) | p95 (4)
kurtosis (4) | skewness (4) | flags (4) | reserved (4)
```

---

### Iteration 3: Accumulation & Layer Selection (Week 4)

**Theme**: Configurable fidelity/overhead trade-off

**Status**: ✅ COMPLETED (2026-02-21)

| Task | Est. | Status |
|------|------|--------|
| Welford accumulation | 4h | [x] |
| accum_steps wiring | 3h | [x] |
| Layer selection protocol | 2h | [x] |
| Layer selection UI | 3h | [ ] (deferred) |
| Sampling rate control | 2h | [x] |
| **Total** | **14h** | |

**Exit Criteria**:
- [x] Accumulation reduces overhead measurably
- [x] Layer selection works via API
- [ ] Layer selection UI (deferred)
- [x] Sampling rate configurable

**Deliverables**:
- `include/protocol_parser.h` - WelfordAccumulator class with online statistics
- `include/protocol.h` - NF_OP_SELECT_LAYERS, NF_OP_SET_SAMPLE_RATE opcodes
- `src/neural_probe.cpp` - Accumulation mode, layer selection, sampling
- `tests/test_statistics.cpp` - Welford accumulator tests (5 new test cases)

**New Python API**:
```python
import neural_probe

# Sampling: capture every Nth forward pass
neural_probe.set_sample_rate(10)  # Only capture 1 in 10

# Layer selection modes
neural_probe.set_layer_selection_mode(0)  # All layers (default)
neural_probe.set_layer_selection_mode(1)  # Whitelist mode
neural_probe.set_layer_selection_mode(2)  # Blacklist mode

neural_probe.add_layer_to_whitelist(0)
neural_probe.add_layer_to_whitelist(5)
neural_probe.add_layer_to_blacklist(10)
neural_probe.clear_layer_selection()

# Accumulation: emit after N calls per layer
neural_probe.set_accumulation_steps(10)  # Average over 10 calls
```

**Welford Accumulator Features**:
- Numerically stable mean/variance computation
- Incremental updates (no need to store all values)
- Reset and reuse support
- Matches exact compute_statistics output

---

### Iteration 4: Attention Visualization (Week 5-6)

**Theme**: Transformer-specific diagnostics

| Task | Est. | Status |
|------|------|--------|
| Attention protocol | 2h | [ ] |
| log_attention API | 4h | [ ] |
| Attention hook example | 2h | [ ] |
| WASM parser update | 2h | [ ] |
| Edge rendering | 6h | [ ] |
| Performance optimization | 4h | [ ] |
| **Total** | **20h** | |

**Exit Criteria**:
- Attention patterns visualized
- Works with real transformer
- Performance acceptable with 10k+ edges

---

### Iteration 5: Resilience & Polish (Week 7)

**Theme**: Production readiness

| Task | Est. | Status |
|------|------|--------|
| Reconnection logic | 3h | [ ] |
| Connection metrics | 3h | [ ] |
| Memory pooling | 4h | [ ] |
| RAII broadcast wrapper | 2h | [ ] |
| Documentation | 2h | [ ] |
| **Total** | **14h** | |

**Exit Criteria**:
- Auto-reconnect works
- Memory bounded
- All APIs documented

---

## Part IV: Performance Budget

| Operation | Budget | Measurement |
|-----------|--------|-------------|
| log_activation call | <50us | Time from Python call to buffer push |
| Ring buffer push | <5us | Lock + copy + notify |
| WASM parse | <100us | Full packet decode |
| Three.js update | <8ms | 120fps frame budget |
| Total overhead | <5% | Training time with vs without probe |

---

## Part V: Success Metrics

### Functional Metrics

| Metric | Target |
|--------|--------|
| Test coverage | >80% lines |
| Protocol compatibility | V1 and V2 coexist |
| Error detection | All invalid inputs caught |
| Reconnection time | <5s average |

### Performance Metrics

| Metric | Target |
|--------|--------|
| Training overhead | <5% |
| Memory growth | Zero after warmup |
| Frontend frame rate | >30fps with 10k particles |
| Packet loss | <0.1% |

### Diagnostic Metrics

| Metric | Target |
|--------|--------|
| Dead neuron detection | 95% recall |
| Explosion detection | 90% recall |
| Latency from issue to visible | <100ms |

---

## Part VI: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| WASM parsing bottleneck | Medium | High | Add worker thread |
| Attention capture overhead | High | Medium | Make optional, top-k |
| Memory unbounded | Medium | High | Strict pool limits |
| Protocol V2 breaks V1 | Low | High | Version negotiation |
| GPU tensor copy stalls | High | High | Async staging |

---

## Part VII: Current Phase

**Completed Iterations**: 
- 1A - Test Foundation ✅
- 1B - Gradient Capture ✅
- 2 - Extended Statistics ✅
- 3 - Accumulation & Layer Selection ✅

**Active Iteration**: 4 - Attention Visualization

**Next Steps**:
1. Define attention protocol structures
2. Implement log_attention API
3. Create attention hook for transformers
4. Update WASM parser
5. Frontend edge rendering

**Iteration 3 Summary**:
- WelfordAccumulator class for online statistics
- Accumulation mode (emit every N calls per layer)
- Layer selection (whitelist/blacklist modes)
- Sampling rate control (capture every Nth forward pass)
- 26 test cases with 259 assertions all passing

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-21 | 1.3 | Completed Iteration 3 - Accumulation & layer selection |
| 2026-02-21 | 1.2 | Completed Iteration 2 - Extended statistics |
| 2026-02-21 | 1.1 | Completed Iteration 1B - Gradient capture |
| 2026-02-21 | 1.0 | Initial roadmap creation |
