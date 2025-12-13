# Synapse

Real-time, low-overhead visualization of LLM internals during training.
![Synapse-v0.1.0](https://github.com/user-attachments/assets/42b2e436-4dcc-4009-bfcd-2996b7d8fcfc)

Synapse streams live tensor activation statistics from a PyTorch training loop into a browser-based 3D dashboard. The design goal is “debuggability without slowing training”: keep the hot-path minimal, do aggregation/sparsification off-thread, and ship compact binary packets over WebSockets.

## What’s In This Repo Today

- `backend_extension/` — Python C++ extension (`neural_probe`) + uWebSockets broadcaster.
- `wasm_parser/` — WebAssembly packet parser (Rust) to keep parsing off the JS main thread.
- `frontend_dashboard/` — SvelteKit + Three.js dashboard (macro layers + micro sparse neurons + control panel).

## What I've Implemented So Far (status)

- Binary protocol v1 with a fixed header and typed payloads:
  - Layer macro summaries (`NF_MSG_LAYER_SUMMARY_BATCH`)
  - Sparse neuron activations (`NF_MSG_SPARSE_ACTIVATIONS`)
  - Control messages (`NF_MSG_CONTROL`) for live threshold changes
  - Topology metadata (`NF_MSG_MODEL_META`) to make layout deterministic
- Frontend:
  - Deterministic “layer grid” placement (no random scatter)
  - Auto-framing camera + orbit controls
  - Control panel (threshold + theme + bloom)
  - More robust parsing/handling when packets arrive out of order
- Tooling:
  - Llama-8B-like multi-layer activation simulator: `backend_extension/python/simulate_llama8b.py`

## Roadmap: “Transformer-Native” Visualizations

Sparse neuron grids are useful, but transformers are often best understood via:

- **Attention patterns** (token→token edges per head, per layer)
- **Residual stream norms** and per-layer contribution
- **Top-k routed activations** (MoE) and expert utilization

The next “real” step is to define an attention packet type that can be rendered as token graphs without pretending we have full neuron connectivity.


## Probable Debugs (FAQ)

- In case you have some issues with the simulator, I gathered some common ones here. Extensive FAQ will be added later.

### Why You Might Still See “One Layer”

If the frontend receives summaries before full topology is known, it can bootstrap from the first summary and later needs to rebuild as new layer summaries arrive. The dashboard now rebuilds the macro layer stack when it detects new layer IDs.

If you still see a single plane:

- Verify the simulator is running multi-layer mode (see “Run”).
- Rotate the camera: stacked planes can overlap face-on.

### Run (local)

1. Start the backend simulator (requires `neural_probe` built/installed):
   - `./backend_extension/python/run_simulate_llama8b.sh --per-block attn+mlp+residual --threshold 0.6 --fps 5`
2. Start the dashboard:
   - `npm -C frontend_dashboard install`
   - `npm -C frontend_dashboard run dev`
   - Open `http://localhost:5173`

### CMake / Build Notes (offline vs online)

The backend extension historically used CMake `FetchContent` to `git clone` dependencies. In restricted/offline environments this fails. The project now supports vendoring dependencies under `backend_extension/third_party/` and turning off FetchContent:

- See `backend_extension/third_party/README.md`

## License

This is licesened Open Source Program under CDLI Non-Commercial Open Source License - Version 2.0, 2025.
Please see LICENSE file for more details.