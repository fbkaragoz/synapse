#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif [[ -x "${BACKEND_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${BACKEND_DIR}/.venv/bin/python"
else
  echo "python not found; activate venv or create one at ${BACKEND_DIR}/.venv" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/simulate_llama8b.py" "$@"

