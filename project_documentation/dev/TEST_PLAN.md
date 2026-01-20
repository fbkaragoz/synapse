# Test Coverage Plan

## Overview
This document outlines the test coverage for the Synapse project across all components: Rust Wasm parser, C++ backend, and TypeScript/Svelte frontend.

## Test Categories

### 1. Rust Wasm Parser Tests (`wasm_parser/src/protocol_tests.rs`)

#### ✅ Implemented Tests
- `test_parse_header_valid` - Validates parsing of a valid packet header
- `test_parse_header_invalid_magic` - Ensures rejection of invalid magic number
- `test_parse_header_too_short` - Verifies error on insufficient buffer size
- `test_parse_layer_summary_batch` - Tests layer summary payload parsing
- `test_parse_sparse_activations` - Tests sparse activation parsing
- `test_parse_control_packet` - Tests control packet parsing
- `test_parse_model_meta` - Tests model metadata parsing
- `test_payload_too_short` - Verifies error on insufficient payload
- `test_constants` - Validates all protocol constants

#### Running Tests
```bash
cd wasm_parser
cargo test --target wasm32-unknown-unknown
```

### 2. C++ Backend Tests (`backend_extension/tests/`)

#### ✅ Implemented Tests

##### Ring Buffer Tests (`ring_buffer_test.cpp`)
- `BasicPushPop` - Basic push and pop functionality
- `PopBlocksWhenEmpty` - Verifies blocking behavior when empty
- `DropOldestWhenFull` - Tests dropping behavior at capacity
- `MultiplePushPop` - Stress test with multiple operations
- `PopReturnsNulloptWhenStopped` - Verifies stop signal handling
- `StopUnblocksWaitingPop` - Tests stop unblocks waiting consumers
- `PushReturnsFalseAfterStop` - Verifies no push after stop
- `ConcurrentProducerConsumer` - Thread safety test
- `LargeDataPacket` - Tests large packet handling

##### Protocol Tests (`protocol_test.cpp`)
- `HeaderSize` - Validates header structure size
- `HeaderDefaults` - Tests default values
- `LayerSummarySize` - Validates summary structure
- `SparseActivationsSize` - Validates sparse structure
- `ControlPacketSize` - Validates control structure
- `LayerInfoSize` - Validates layer info structure
- `MagicNumber` - Tests magic constants
- `ControlOpcodes` - Tests control opcodes
- `Flags` - Tests flag constants
- `BuildLayerSummaryPacket` - Tests packet building
- `BuildSparseActivationsPacket` - Tests sparse packet building
- `BuildControlPacket` - Tests control packet building
- `BuildModelMetaPacket` - Tests meta packet building
- `PackedStructAlignment` - Verifies struct packing

#### Running Tests
```bash
cd backend_extension
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DPython3_ROOT=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
cmake --build build --target ring_buffer_tests
cmake --build build --target protocol_tests
./build/ring_buffer_tests
./build/protocol_tests
```

### 3. Frontend Tests (`frontend_dashboard/src/`)

#### ✅ Implemented Tests

##### WebSocket Client Tests (`lib/ws_client.test.ts`)
- Store initialization with default values
- Config threshold store updates
- Layer summaries store updates
- Sparse activations store updates
- Model meta store updates
- Connection state updates
- Packet count updates
- Wasm parser null handling
- Control sending without parser

##### Types Tests (`lib/types.test.ts`)
- PacketHeader structure validation
- LayerSummary structure validation
- SparseActivationData structure validation
- ControlPacket structure validation
- LayerInfo structure validation
- ModelMeta structure validation
- ParsedPacket structure validation
- Optional fields in ParsedPacket

#### Running Tests
```bash
cd frontend_dashboard
npm install
npm run test
npm run test:ui
```

## Test Coverage Goals

### High Priority
- [x] Protocol header parsing
- [x] Protocol payload parsing (all message types)
- [x] Ring buffer operations (push, pop, concurrency)
- [x] Frontend store management
- [x] Type definitions and interfaces

### Medium Priority
- [ ] Error handling edge cases
- [ ] Boundary value testing (max/min values)
- [ ] Performance benchmarks
- [ ] Integration tests (full pipeline)
- [ ] WebSocket connection lifecycle

### Low Priority
- [ ] Memory leak detection
- [ ] Stress testing (high packet rates)
- [ ] Network failure simulation
- [ ] Browser compatibility tests

## Continuous Integration

Test execution is integrated into the development workflow:
1. Pre-commit hooks run relevant tests
2. CI pipeline runs full test suite on PR
3. Coverage reports generated for review
4. Failed tests block merging

## Test Data Management

Test packets are generated programmatically to:
- Ensure deterministic behavior
- Avoid hardcoding test vectors
- Cover edge cases systematically
- Support future protocol versioning

## Known Test Gaps

1. **Wasm Integration** - Tests run in native Rust, not actual Wasm environment
2. **Network Layer** - Mock WebSocket, no real network tests
3. **Browser Environment** - jsdom doesn't fully emulate browser APIs
4. **Python C-Extension** - Integration with Python not tested

## Future Improvements

1. Add E2E tests with real backend server
2. Add visual regression tests for Three.js rendering
3. Add performance profiling in test suite
4. Add cross-browser testing with Playwright
5. Add memory profiling for long-running sessions
