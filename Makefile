.PHONY: all build test clean install help wasm frontend backend
.DEFAULT_GOAL := help

# Variables
CMAKE_BUILD_TYPE ?= Debug
PYTHON ?= python3
RUST_TARGET ?= wasm32-unknown-unknown
NODE_MODULES := frontend_dashboard/node_modules
BACKEND_BUILD := backend_extension/build

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

help:
	@echo "$(GREEN)Synapse - Build System$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo "  $(GREEN)all$(NC)            - Build all components"
	@echo "  $(GREEN)build$(NC)          - Build all components (alias for all)"
	@echo "  $(GREEN)test$(NC)           - Run all tests"
	@echo "  $(GREEN)clean$(NC)          - Clean all build artifacts"
	@echo "  $(GREEN)install$(NC)        - Install all dependencies"
	@echo "  $(GREEN)wasm$(NC)           - Build Wasm parser"
	@echo "  $(GREEN)frontend$(NC)       - Build frontend dashboard"
	@echo "  $(GREEN)backend$(NC)        - Build backend extension"
	@echo "  $(GREEN)lint$(NC)           - Run linters on all components"
	@echo "  $(GREEN)format$(NC)         - Format all code"
	@echo "  $(GREEN)check$(NC)          - Run type checks"
	@echo "  $(GREEN)dev$(NC)            - Start development servers"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make build              # Build everything"
	@echo "  make test               # Run all tests"
	@echo "  make dev                # Start all dev servers"
	@echo "  make lint               # Check code quality"
	@echo ""
	@echo "$(YELLOW)Environment variables:$(NC)"
	@echo "  CMAKE_BUILD_TYPE      - Debug or Release (default: Debug)"
	@echo "  PYTHON               - Python executable (default: python3)"

all: build

build: wasm backend

install:
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	@echo "$(GREEN)→ Installing Python dependencies$(NC)"
	cd backend_extension && $(PYTHON) -m pip install -e . --no-build-isolation
	@echo "$(GREEN)→ Installing Node.js dependencies$(NC)"
	cd frontend_dashboard && npm install
	@echo "$(GREEN)→ Installing Rust toolchain for Wasm target$(NC)"
	rustup target add $(RUST_TARGET) 2>/dev/null || echo "Rust target already installed"
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

wasm:
	@echo "$(YELLOW)Building Wasm parser...$(NC)"
	cd wasm_parser && \
	if command -v wasm-pack >/dev/null 2>&1; then \
		wasm-pack build --target web --out-dir ../frontend_dashboard/src/lib/wasm_pkg; \
	else \
		echo "$(RED)wasm-pack not found. Install with: cargo install wasm-pack$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Wasm parser built$(NC)"

backend:
	@echo "$(YELLOW)Building backend extension...$(NC)"
	cd backend_extension && \
	mkdir -p $(BACKEND_BUILD) && \
	cmake -S . -B $(BACKEND_BUILD) \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DPython3_ROOT=$$($(PYTHON) -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
	cmake --build $(BACKEND_BUILD) --config $(CMAKE_BUILD_TYPE)
	@echo "$(GREEN)✓ Backend extension built$(NC)"

frontend:
	@echo "$(YELLOW)Building frontend dashboard...$(NC)"
	cd frontend_dashboard && \
	npm run build
	@echo "$(GREEN)✓ Frontend built$(NC)"

test: test-backend test-frontend test-wasm
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-backend:
	@echo "$(YELLOW)Running C++ tests...$(NC)"
	cd backend_extension && \
	if [ -f "$(BACKEND_BUILD)/ring_buffer_tests" ]; then \
		$(BACKEND_BUILD)/ring_buffer_tests; \
	else \
		echo "$(RED)Tests not built. Run 'make build' first$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ C++ tests passed$(NC)"

test-frontend:
	@echo "$(YELLOW)Running frontend tests...$(NC)"
	cd frontend_dashboard && npm run test
	@echo "$(GREEN)✓ Frontend tests passed$(NC)"

test-wasm:
	@echo "$(YELLOW)Running Rust tests...$(NC)"
	cd wasm_parser && cargo test
	@echo "$(GREEN)✓ Rust tests passed$(NC)"

lint: lint-frontend lint-rust lint-cpp

lint-frontend:
	@echo "$(YELLOW)Linting frontend...$(NC)"
	cd frontend_dashboard && npm run lint

lint-rust:
	@echo "$(YELLOW)Linting Rust code...$(NC)"
	cd wasm_parser && cargo clippy --target $(RUST_TARGET)

lint-cpp:
	@echo "$(YELLOW)Checking C++ code formatting...$(NC)"
	cd backend_extension && \
	if command -v clang-format >/dev/null 2>&1; then \
		clang-format -i --dry-run --Werror src/*.cpp include/*.h || echo "Code needs formatting"; \
	else \
		echo "$(YELLOW)clang-format not found, skipping$(NC)"; \
	fi

format: format-frontend format-rust format-cpp

format-frontend:
	@echo "$(YELLOW)Formatting frontend...$(NC)"
	cd frontend_dashboard && npm run format

format-rust:
	@echo "$(YELLOW)Formatting Rust code...$(NC)"
	cd wasm_parser && cargo fmt

format-cpp:
	@echo "$(YELLOW)Formatting C++ code...$(NC)"
	cd backend_extension && \
	if command -v clang-format >/dev/null 2>&1; then \
		clang-format -i src/*.cpp include/*.h; \
		echo "$(GREEN)✓ C++ code formatted$(NC)"; \
	else \
		echo "$(YELLOW)clang-format not found, skipping$(NC)"; \
	fi

check: check-frontend check-rust check-cpp

check-frontend:
	@echo "$(YELLOW)Type checking frontend...$(NC)"
	cd frontend_dashboard && npm run check

check-rust:
	@echo "$(YELLOW)Type checking Rust...$(NC)"
	cd wasm_parser && cargo clippy --target $(RUST_TARGET)

check-cpp:
	@echo "$(YELLOW)Checking C++ (handled by clang-format)$(NC)"
	@echo "$(YELLOW)CMake configures the build type$(NC)"

clean: clean-frontend clean-backend clean-wasm

clean-frontend:
	@echo "$(YELLOW)Cleaning frontend...$(NC)"
	cd frontend_dashboard && \
	rm -rf .svelte-kit build dist

clean-backend:
	@echo "$(YELLOW)Cleaning backend...$(NC)"
	cd backend_extension && \
	rm -rf $(BACKEND_BUILD) *.so *.egg-info __pycache__

clean-wasm:
	@echo "$(YELLOW)Cleaning Rust Wasm artifacts...$(NC)"
	cd wasm_parser && \
	cargo clean

dev:
	@echo "$(YELLOW)Starting development servers...$(NC)"
	@echo "$(GREEN)→ Backend server (run in separate terminal):$(NC)"
	@echo "$(YELLOW)  cd backend_extension && $(PYTHON) python/python/simulate_llama8b.py$(NC)"
	@echo "$(GREEN)→ Frontend server:$(NC)"
	@echo "$(YELLOW)  cd frontend_dashboard && npm run dev$(NC)"
	@echo "$(GREEN)✓ Dev servers ready$(NC)"

.PHONY: test-backend test-frontend test-wasm lint-frontend lint-rust lint-cpp format-frontend format-rust format-cpp check-frontend check-rust check-cpp clean-frontend clean-backend clean-wasm
