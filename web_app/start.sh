#!/usr/bin/env bash
# ==============================================================================
# Massive MIMO AI Studio - Full Stack Launcher
# Pure React Frontend (Vite) + FastAPI Python Backend
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BACKEND_DIR="$DIR/backend"
FRONTEND_DIR="$DIR/frontend"

# Prefer conda deeplearn_arm environment if available, else system python3
PYTHON_BIN="/usr/local/Caskroom/miniforge/base/envs/deeplearn_arm/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(which python3)"
fi

echo "=============================================================="
echo " Starting Massive MIMO AI Channel Studio"
echo " Backend: Python FastAPI ($PYTHON_BIN)"
echo " Frontend: Pure JavaScript React (Vite)"
echo "=============================================================="

# 1. Start Python FastAPI backend on port 8000
echo "[1/2] Starting Backend API on http://localhost:8000..."
cd "$BACKEND_DIR"
"$PYTHON_BIN" -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Trap Ctrl+C to kill both background processes on exit
cleanup() {
    echo ""
    echo "Shutting down Massive MIMO AI Studio..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Give backend a moment to initialize
sleep 2

# 2. Start React Vite dev server on port 5173
echo "[2/2] Starting React Frontend on http://localhost:5173..."
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

echo "=============================================================="
echo " Studio is LIVE!"
echo " Web UI:     http://localhost:5173"
echo " API Docs:   http://localhost:8000/docs"
echo "=============================================================="
echo "Press Ctrl+C to stop both servers."

wait
