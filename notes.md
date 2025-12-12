# Neural-Flow — Developer Notes (Scaffolding + Technical Guide)

This repo is intentionally scaffold-only: placeholder files exist, but no implementation is written yet.
The goal is a **real-time LLM activation visualization pipeline** with minimal (<5%) training overhead:

- **PyTorch training loop (Python)** calls a **Python C-extension** (“probe”).
- The **C++ extension** (1) samples/aggregates activations, (2) runs a **uWebSockets** broadcast server in a **background thread**, and (3) ships compact **binary packets**.
- The **SvelteKit frontend** receives WebSocket **ArrayBuffer** frames and forwards them into a **Wasm parser** (Rust or C++→Wasm) to keep JS main-thread work low.

---

## Repository Layout (intentional separation)

- `backend_extension/` — Python C-extension + uWebSockets relay (C++17+)
  - `backend_extension/src/main.cpp` — module entrypoints (placeholder)
  - `backend_extension/src/neural_probe.cpp` — hook ingestion + aggregation (placeholder)
  - `backend_extension/src/server_uwebsockets.cpp` — WS server thread + broadcast (placeholder)
  - `backend_extension/include/protocol.h` — shared packet layout definitions (placeholder)
  - `backend_extension/include/ring_buffer.h` — lock-free / low-lock queue (placeholder)
  - `backend_extension/include/server.h` — server interface (placeholder)
  - `backend_extension/setup.py` — build script placeholder
  - `backend_extension/CMakeLists.txt` — native build placeholder
  - `backend_extension/python/hook_example.py` — usage example placeholder
- `wasm_parser/` — binary decoder compiled to WebAssembly (recommended: Rust + `wasm-bindgen`)
  - `wasm_parser/src/lib.rs` — wasm exports (placeholder)
  - `wasm_parser/src/protocol.rs` — packet parsing helpers (placeholder)
- `frontend_dashboard/` — SvelteKit UI + Three.js/Babylon.js visualization
  - `frontend_dashboard/src/routes/+page.svelte` — main dashboard view (placeholder)
  - `frontend_dashboard/src/lib/ws_client.ts` — WS connect + frame handler (placeholder)
  - `frontend_dashboard/src/lib/wasm_bridge.ts` — calls into Wasm (placeholder)
  - `frontend_dashboard/src/lib/three/scene.ts` — render loop + LOD (placeholder)

---

## Build Order (developer workflow)

The frontend depends on the Wasm artifact; the Wasm artifact depends on the protocol definition; the backend depends on the same protocol definition.

1. **Finalize protocol structs (single source of truth)**
   - Edit: `backend_extension/include/protocol.h`
   - Mirror in Rust parsing: `wasm_parser/src/protocol.rs` (parse-bytes, don’t rely on struct-casts).
2. **Build the C++ Python extension**
   - Output: importable module (e.g., `import neural_probe`)
   - Recommended: `pybind11` or CPython API + CMake (choose one; document it in `backend_extension/setup.py` / `backend_extension/CMakeLists.txt`).
3. **Compile the Wasm parser**
   - Output: `.wasm` + JS glue (or direct WASI-less wasm module)
   - Recommended: Rust `wasm-pack` targeting `web` and emitting into `frontend_dashboard/src/lib/wasm_pkg/` (directory will be generated later).
4. **Run the SvelteKit dashboard**
   - Dev server connects to the backend WS endpoint and renders macro/micro LOD views.

Suggested local commands (fill in build files later):

- Backend extension:
  - `python -m venv .venv && . .venv/bin/activate`
  - `pip install -U pip setuptools wheel`
  - `pip install -e backend_extension`
- Wasm parser (Rust):
  - `cd wasm_parser`
  - `rustup target add wasm32-unknown-unknown`
  - `wasm-pack build --target web --out-dir ../frontend_dashboard/src/lib/wasm_pkg`
- Frontend:
  - `cd frontend_dashboard`
  - `npm install`
  - `npm run dev`

---

## Data Protocol Definition (binary, versioned, zero-JSON)

Design goals:

- **Single WebSocket message = one “frame”** of visualization data.
- **Little-endian** for all integers/floats.
- **No padding** in the transmitted format (treat as byte layout, not C++ ABI layout).
- **Versioned header** so the frontend can reject unknown versions.
- **Sparse payloads** for per-neuron view (thresholded indices + values).
- Keep parsing cheap in Wasm: avoid nested variable-length structures unless necessary.

### Wire Header (fixed-size, 32 bytes)

All messages begin with this header, followed immediately by `payload_bytes` of payload.

```c
// backend_extension/include/protocol.h
// WIRE FORMAT: little-endian, packed-by-definition (do not rely on compiler packing).

enum NF_MsgType : uint16_t {
  NF_MSG_LAYER_SUMMARY_BATCH = 1,   // macro LOD
  NF_MSG_SPARSE_ACTIVATIONS  = 2,   // micro LOD
  NF_MSG_CONTROL             = 3,   // optional: frontend->backend config
};

enum NF_Flags : uint32_t {
  NF_FLAG_NONE       = 0,
  NF_FLAG_FP16       = 1u << 0,  // values are IEEE 754 half in payload (if supported)
  NF_FLAG_FP32       = 1u << 1,  // values are float32 in payload
  NF_FLAG_COMPRESSED = 1u << 2,  // reserved (e.g., zstd) - avoid initially
};

// Header size: 32 bytes (stable).
// magic: ASCII 'N' 'F' 'L' 'W' => 0x4E 0x46 0x4C 0x57
typedef struct NF_PacketHeader {
  uint32_t magic;          // 0x574C464E ("NFLW" when read as little-endian u32)
  uint16_t version;        // 1
  uint16_t msg_type;       // NF_MsgType
  uint32_t flags;          // NF_Flags
  uint64_t seq;            // monotonically increasing
  uint64_t timestamp_ns;   // CLOCK_MONOTONIC or steady_clock converted to ns
  uint32_t payload_bytes;  // bytes immediately following this header
  uint32_t reserved;       // must be 0 for v1
} NF_PacketHeader;
```

**Offset table (v1):**

- `0`  : `u32 magic`
- `4`  : `u16 version`
- `6`  : `u16 msg_type`
- `8`  : `u32 flags`
- `12` : `u64 seq`
- `20` : `u64 timestamp_ns`
- `28` : `u32 payload_bytes`
- `32` : payload begins (note: reserved makes header 32 bytes total; payload begins at byte 32)

### Payload: Layer Summary Batch (macro LOD)

Used for the “layers as blocks” view (mean/max activity).

```c
typedef struct NF_LayerSummaryV1 {
  uint32_t layer_id;     // stable numeric id assigned by Python hook registration
  uint32_t neuron_count; // number of neurons aggregated into this layer
  float    mean;         // mean activation after accumulation/throttle
  float    max;          // max activation after accumulation/throttle
} NF_LayerSummaryV1;

typedef struct NF_LayerSummaryBatchV1 {
  uint32_t count;        // number of summaries that follow
  uint32_t reserved;     // must be 0 for v1
  // NF_LayerSummaryV1 summaries[count];
} NF_LayerSummaryBatchV1;
```

Wire payload layout:

- `NF_LayerSummaryBatchV1` (8 bytes)
- then `count * NF_LayerSummaryV1` (each 16 bytes)

### Payload: Sparse Activations (micro LOD)

Used after user clicks a layer and expands to neuron particles.
Only transmit entries above a configured `threshold`.

```c
typedef struct NF_SparseActivationsV1 {
  uint32_t layer_id;
  uint32_t count;        // number of entries that follow
  uint32_t reserved0;    // must be 0 for v1
  uint32_t reserved1;    // must be 0 for v1
  // entries follow
} NF_SparseActivationsV1;

typedef struct NF_ActivationEntryF32V1 {
  uint32_t neuron_idx;   // index within the layer (0..neuron_count-1)
  float    value;        // activation value
} NF_ActivationEntryF32V1;
```

Wire payload layout (when `flags` includes `NF_FLAG_FP32`):

- `NF_SparseActivationsV1` (16 bytes)
- then `count * NF_ActivationEntryF32V1` (each 8 bytes)

If `NF_FLAG_FP16` is used later, define `NF_ActivationEntryF16V1 { u32 neuron_idx; u16 value_fp16; u16 pad; }` explicitly to keep 8-byte entries and avoid unaligned reads in Wasm.

### Frontend parsing rule (important)

In Wasm/Rust, **do not `transmute`** the header/payload into `repr(C)` structs due to alignment/endianness. Parse from `&[u8]` using explicit little-endian reads and bounds checks. The parsing module (`wasm_parser/src/protocol.rs`) should:

- Validate `magic`, `version`, `payload_bytes`.
- Ensure `buffer_len >= 32 + payload_bytes`.
- Dispatch by `msg_type`.
- Return typed arrays (indices + values) ready for JS/Three.js.

---

## Threading Model (Python ↔ C++ extension ↔ uWebSockets)

Hard requirements:

- The **training loop must not block** on networking.
- The probe must not hold the GIL during any blocking/long work.
- The WS server event loop runs in a **dedicated background thread**.

### Recommended concurrency design

- **Producer(s):** Python training thread calling into extension via forward hooks.
  - Do *minimal* work: validate inputs + enqueue a small record (layer_id + pointer/shape/stride metadata or a copied slice).
- **Aggregator thread (optional):** can be the same as server thread or separate.
  - Performs accumulation over `accumulation_steps` or time window (`broadcast_interval_ms`).
  - Applies sparsity thresholding and builds binary packets.
- **uWebSockets thread:** owns the WS app + loop; broadcasts prepared packets to clients.

Use a **lock-free or low-lock ring buffer** (`backend_extension/include/ring_buffer.h`) between producer and server thread to keep overhead under 5%.

### GIL handling

Rules of thumb:

- Any function called from Python enters with the **GIL held**.
- If that function does anything that can block (waiting on server start/stop, joining threads, flushing queues), wrap it with `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS`.
- Background threads **must not touch Python objects** unless they acquire the GIL via `PyGILState_Ensure`.

Minimal pattern for a blocking call (example: `run_forever()` if you choose to expose it):

```c
// PSEUDOCODE SHAPE ONLY — do not paste as implementation blindly.
static PyObject* nf_run_forever(PyObject* self, PyObject* args) {
  Py_BEGIN_ALLOW_THREADS
  // blocks inside server loop (or waits on a condition variable)
  nf_server_run();
  Py_END_ALLOW_THREADS
  Py_RETURN_NONE;
}
```

For the main recommended design (“start thread and return”), `start_server()` should return quickly and not need `Py_BEGIN_ALLOW_THREADS` unless it waits for readiness.

### uWebSockets cross-thread sending (important)

uWebSockets generally requires that socket operations occur on the loop thread. Common patterns:

- Build packets on producer/aggregator threads, then use `Loop::defer(...)` to schedule broadcast on the WS loop thread.
- Keep “current packet” in a thread-safe structure and only perform `ws->send(...)` from the loop thread.

---

## Component Responsibilities (what goes where)

### `backend_extension/` (C++ + Python)

Targets:

- Export a Python module API similar to:
  - `start_server(host, port, ...)`
  - `stop_server()`
  - `set_threshold(float)`
  - `set_accumulation_steps(int)` / `set_broadcast_interval_ms(int)`
  - `record(layer_id, tensor)` (called by forward hook)

Constraints:

- The `record(...)` path must be fast and mostly allocation-free.
- GPU tensors: you cannot “read a raw GPU pointer” safely without coordinating CUDA context/stream; plan for a staging copy (pinned host memory) or a CPU-side tap where feasible.
- Apply thresholding and LOD selection before sending over WS.

### `wasm_parser/` (Wasm decoder)

Targets:

- Provide Wasm exports that accept a `Uint8Array`/`ArrayBuffer` (or pointer+len) and return:
  - Parsed header fields
  - For sparse messages: `Uint32Array` indices + `Float32Array` values
  - For summaries: a compact array of per-layer stats

Constraints:

- Zero-copy as much as possible: parse in-place; allocate only outputs needed by renderer.
- Strict bounds checks to avoid trapping on malformed packets.

### `frontend_dashboard/` (SvelteKit + 3D)

Targets:

- WebSocket client:
  - `binaryType = "arraybuffer"`
  - forward raw buffers to Wasm parser
- Rendering:
  - Macro view: layer blocks with intensity mapped from mean/max
  - Micro view: click to expand a layer; render neuron particles from sparse entries
- Performance:
  - Avoid per-frame allocations; reuse buffers; offload parsing to Wasm

---

## Development Checklist (for the next AI/developer)

1. Decide extension build system:
   - Pick **(A) PyBind11 + setuptools** or **(B) CPython C-API + CMake**; update `backend_extension/setup.py` and/or `backend_extension/CMakeLists.txt`.
2. Implement protocol as single source:
   - Finalize v1 in `backend_extension/include/protocol.h`.
   - Mirror parsing logic in `wasm_parser/src/protocol.rs` (parse bytes explicitly).
3. Implement backend threading skeleton:
   - Background WS thread lifecycle (start/stop/ready).
   - Safe cross-thread enqueue (ring buffer).
   - Ensure no Python C-API usage off-thread without `PyGILState_Ensure`.
4. Implement data ingestion:
   - `record(layer_id, tensor)` fast path.
   - Aggregation (`accumulation_steps` / `broadcast_interval_ms`) + threshold filter.
5. Implement uWebSockets broadcast:
   - Loop-thread send only; use `defer` or equivalent for cross-thread scheduling.
   - Validate multi-client fanout and disconnect behavior.
6. Implement Wasm parser:
   - Export parse functions; verify against golden test vectors (recorded packets).
7. Implement frontend minimal dashboard:
   - WS connect + display connection status + macro layer blocks.
   - Add micro expand-on-click rendering path.
8. Performance verification:
   - Profile Python training overhead (target <5%).
   - Measure packet sizes; tune threshold/LOD.
9. Hardening:
   - Version mismatch handling.
   - Backpressure strategy (drop-oldest vs drop-newest).
   - Clean shutdown on Python interpreter exit.
