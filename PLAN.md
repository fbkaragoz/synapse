# Project Prompt: "Neural-Flow" Real-Time LLM Visualization Engine

## Role
Senior Systems Architect & Full-Stack Engineer (C++ & WebAssembly Expert)

## Objective
Develop a high-performance, real-time 3D visualization dashboard to monitor internal neuron activations of an LLM during training without significantly impacting training throughput.

---

## 1. System Architecture & Tech Stack

### Context
The system must visualize live tensor activations from a PyTorch training loop.

### Backend (High-Performance Relay)

- **Language:** C++ (C++17 or newer)
- **Integration:** Develop a Python C-Extension (using PyBind11 or plain CPython API) acting as the "Probe"
- **Mechanism:** The Python hook passes the raw tensor pointer (GPU or CPU) to the C++ extension. The C++ side reads this memory strictly in a non-blocking thread to avoid stalling the training loop (GIL release is mandatory)
- **Transport:** Use uWebSockets (C++) for ultra-low latency broadcasting to the frontend

### Frontend (Visualization)

- **Framework:** SvelteKit (for reactive UI) + WebAssembly (Rust or C++ compiled to Wasm)
- **Rendering:** Three.js or Babylon.js (driven by Wasm logic for particle positioning)
- **Data Parsing:** Incoming WebSocket binary streams must be decoded inside the Wasm module to offload the JS main thread

---

## 2. Core Features & Logic

### A. Data Aggregation & Throttling (Backend Side)

- **No Mock Data:** Real activation values only
- **Step Aggregation:** Do not broadcast every training step. Implement a configurable `accumulation_steps` (e.g., average activations over 10 steps) or `broadcast_interval`
- **Sparsity/Compression:** Implement a threshold filter in C++. Only send neuron indices and values where `activation > threshold`. Use a binary format (e.g., FlatBuffers or a custom struct) to minimize packet size

### B. Interactive 3D UI (Frontend Side)

- **Level of Detail (LOD):** Do not render all neurons simultaneously
  - **View 1 (Macro):** Represent Layers as blocks/planes. Color intensity represents average layer activity
  - **View 2 (Micro):** User clicks a Layer Block → The block expands (animation) to reveal individual neurons (particles) in 3D space
- **Visual Metaphor:** Use a "brain synapse" aesthetic. Glowing particles for active neurons, fading trails for signal propagation

---

## 3. Deliverables & Constraints

1. **The C++ Extension:** Provide the `setup.py` and C++ source to build the module that `import neural_probe` would use
2. **The Hook Pattern:** Provide the Python snippet showing how to register this C++ probe into `model.register_forward_hook()`
3. **The Svelte Component:** A Svelte component that mounts the Wasm module and handles the Canvas rendering loop
4. **Performance Requirement:** The overhead on the Python training loop must be **< 5%**