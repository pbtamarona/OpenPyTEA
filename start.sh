#!/usr/bin/env bash
# Start backend + frontend together. Ctrl+C stops both.
# First run bootstraps .venv and installs deps; subsequent runs just activate.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

if [[ ! -d "$VENV" ]]; then
  echo "→ Creating .venv (one-time setup)"
  PY="$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10)"
  if [[ -z "$PY" ]]; then
    echo "✗ Need Python >= 3.10 (system python3 is too old). Install via Homebrew: brew install python@3.13" >&2
    exit 1
  fi
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -e "$ROOT" -r "$ROOT/backend/requirements.txt"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "→ Installing frontend deps (one-time setup)"
  (cd "$ROOT/frontend" && npm install)
fi

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "→ Starting backend on http://localhost:8000"
(
  cd "$ROOT/backend"
  PYTHONPATH="$ROOT/src" python -m uvicorn app.main:app --reload --port 8000
) &
BACKEND_PID=$!

echo "→ Starting frontend on http://localhost:5173"

# Open the browser once Vite is actually serving (poll for up to ~20s).
URL="http://localhost:5173"
if command -v open >/dev/null 2>&1; then OPENER=open
elif command -v xdg-open >/dev/null 2>&1; then OPENER=xdg-open
else OPENER=""; fi
if [[ -n "$OPENER" ]]; then
  (
    for _ in $(seq 1 40); do
      if curl -s -o /dev/null "$URL"; then
        "$OPENER" "$URL"
        exit 0
      fi
      sleep 0.5
    done
  ) &
fi

cd "$ROOT/frontend"
npm run dev
