# Synapse

Real-time, low-overhead visualization of LLM internals during training.

![Synapse-v0.1.0](https://github.com/user-attachments/assets/42b2e436-4dcc-4009-bfcd-2996b7d8fcfc)

Synapse streams live tensor activation statistics from a PyTorch training loop into a browser-based 3D dashboard. The design goal is "debuggability without slowing training": keep the hot-path minimal, do aggregation/sparsification off-thread, and ship compact binary packets over WebSockets.

---

## Quick Start

### Prerequisites

- **Python 3.8+** for backend extension
- **Node.js 18+** and npm for frontend
- **Rust stable** and wasm-pack for WebAssembly parser
- **CMake 3.14+** for backend build
- **C++17 compatible compiler**

### One-Command Setup

```bash
# Install all dependencies
make install

# Build all components
make build

# Run tests
make test

# Start development servers (backend in one terminal, frontend in another)
make dev
```

### Manual Setup (if Makefile doesn't work)

#### 1. Backend Extension

```bash
cd backend_extension
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e . --no-build-isolation
```

#### 2. Wasm Parser

```bash
cd wasm_parser
rustup target add wasm32-unknown-unknown
wasm-pack build --target web --out-dir ../frontend_dashboard/src/lib/wasm_pkg
```

#### 3. Frontend Dashboard

```bash
cd frontend_dashboard
npm install
```

---

## Development

### Build Commands

| Command | Description |
|---------|-------------|
| `make` | Show help |
| `make build` | Build all components (wasm + backend) |
| `make wasm` | Build Wasm parser only |
| `make backend` | Build C++ extension only |
| `make frontend` | Build frontend only |
| `make test` | Run all tests |
| `make clean` | Clean all build artifacts |

### Test Commands

| Command | Description |
|---------|-------------|
| `make test-backend` | Run C++ unit tests |
| `make test-frontend` | Run frontend unit tests |
| `make test-wasm` | Run Rust unit tests |

### Code Quality Commands

| Command | Description |
|---------|-------------|
| `make lint` | Run all linters |
| `make format` | Format all code |
| `make check` | Run type checks |

### Running the Application

#### Backend Simulator

Terminal 1:
```bash
cd backend_extension
python python/simulate_llama8b.py --per-block attn+mlp+residual --threshold 0.6 --fps 5
```

Available options:
- `--host HOST` - WebSocket host (default: localhost)
- `--port PORT` - WebSocket port (default: 9000)
- `--fps N` - Packets per second (default: 10)
- `--threshold N` - Activation threshold (default: 0.5)
- `--layers N` - Number of layers (default: 32)
- `--per-block MODE` - Emissions per block (attn+mlp, attn+mlp+residual, residual)

#### Frontend Dashboard

Terminal 2:
```bash
cd frontend_dashboard
npm run dev
```

Open browser to `http://localhost:5173`

---

## Project Structure

```
synapse/
├── backend_extension/     # Python C++ extension (neural_probe)
│   ├── src/              # C++ source
│   ├── include/          # C++ headers
│   ├── python/            # Python scripts and simulator
│   └── tests/            # C++ unit tests
├── wasm_parser/           # WebAssembly packet parser (Rust)
│   ├── src/              # Rust source
│   └── tests/            # Rust unit tests
├── frontend_dashboard/    # SvelteKit dashboard (TypeScript/Svelte)
│   ├── src/              # Source code
│   │   ├── lib/          # Utilities and stores
│   │   └── routes/       # Pages and components
│   └── static/           # Static assets
└── project_documentation/ # Development documentation
```

---

## Architecture

### Data Flow

1. **Python Training Loop** → Calls `neural_probe.log_activation(layer_id, tensor)`
2. **C++ Extension** → Processes tensor, applies threshold, creates binary packets
3. **Ring Buffer** → Thread-safe queue between Python and WebSocket thread
4. **uWebSockets** → Broadcasts packets to connected clients
5. **Frontend WebSocket** → Receives binary data
6. **Wasm Parser** → Decodes binary packets off main thread
7. **Svelte Stores** → Update reactive state
8. **Three.js** → Renders 3D visualization

### Protocol

Binary protocol with 32-byte header + typed payload:

- **Layer Summary Batch**: Macro view (mean/max per layer)
- **Sparse Activations**: Micro view (indices + values above threshold)
- **Control Messages**: Frontend→backend config changes
- **Model Meta**: Topology metadata for deterministic layout

See `backend_extension/include/protocol.h` for full specification.

---

## Code Quality

### Linting

- **Frontend**: ESLint + Prettier (auto-format on save)
- **Backend**: Clang-Format (manual via `make format-cpp`)
- **Wasm**: Rustfmt (automatic via `make format-rust`)

### Type Safety

- **Frontend**: Strict TypeScript, no `any` types
- **Backend**: C++17 with modern practices
- **Wasm**: Full type safety with serde

### Testing

- **Unit tests** for all components
- Run `make test` to verify builds
- See `project_documentation/dev/TEST_PLAN.md` for coverage details

---

## Troubleshooting

### Common Issues

#### "Cannot import neural_probe"
```bash
cd backend_extension
pip install -e . --no-build-isolation
```

#### "Wasm parser not ready"
```bash
cd wasm_parser
wasm-pack build --target web --out-dir ../frontend_dashboard/src/lib/wasm_pkg
```

#### "Frontend fails to connect"
- Verify backend simulator is running on port 9000
- Check firewall settings
- Browser console shows connection status

#### "Only one layer visible"
- Verify simulator runs in multi-layer mode
- Rotate camera angle to see stacked planes

### Debugging

Enable verbose logging:

**Frontend**: Browser console shows all WebSocket events
**Backend**: C++ logs to stdout with timestamps and levels
**Wasm**: Parse errors thrown to browser console

---

## Configuration

### Environment Variables

- `CMAKE_BUILD_TYPE` - Debug or Release (default: Debug)
- `PYTHON` - Python executable (default: python3)

### Makefile Options

All standard Makefile variables supported:
- `make CMAKE_BUILD_TYPE=Release build` - Build optimized version
- `make PYTHON=python3.9` - Use specific Python version

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This is licensed Open Source Program under CDLI Non-Commercial Open Source License - Version 2.0, 2025.
See LICENSE file for details.

---

## What’s In This Repo Today

- `backend_extension/` — Python C++ extension (`neural_probe`) + uWebSockets broadcaster
- `wasm_parser/` — WebAssembly packet parser (Rust)
- `frontend_dashboard/` — SvelteKit + Three.js dashboard (macro layers + micro sparse neurons + control panel)

---

## What I've Implemented So Far (status)

- Binary protocol v1 with a fixed header and typed payloads:
  - Layer macro summaries (`NF_MSG_LAYER_SUMMARY_BATCH`)
  - Sparse neuron activations (`NF_MSG_SPARSE_ACTIVATIONS`)
  - Control messages (`NF_MSG_CONTROL`) for live threshold changes
  - Topology metadata (`NF_MSG_MODEL_META`) to make layout deterministic
- Frontend:
  - Deterministic "layer grid" placement (no random scatter)
  - Auto-framing camera + orbit controls
  - Control panel (threshold + theme + bloom)
  - More robust parsing/handling when packets arrive out of order
- Tooling:
  - Llama-8B-like multi-layer activation simulator: `backend_extension/python/simulate_llama8b.py`

---

## Roadmap: "Transformer-Native" Visualizations

Sparse neuron grids are useful, but transformers are often best understood via:

- **Attention patterns** (token→token edges per head, per layer)
- **Residual stream norms** and per-layer contribution
- **Top-k routed activations** (MoE) and expert utilization

The next "real" step is to define an attention packet type that can be rendered as token graphs without pretending we have full neuron connectivity.

---

## Probable Debugs (FAQ)

- In case you have some issues with the simulator, I gathered some common ones here. Extensive FAQ will be added later.

### Why You Might Still See "One Layer"

If the frontend receives summaries before full topology is known, it can bootstrap from the first summary and later needs to rebuild as new layer summaries arrive. The dashboard now rebuilds the macro layer stack when it detects new layer IDs.

If you still see a single plane:

- Verify the simulator is running multi-layer mode (see "Run").
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