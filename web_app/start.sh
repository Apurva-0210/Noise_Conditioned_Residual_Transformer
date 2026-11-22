#!/usr/bin/env bash
# ==============================================================================
# Massive MIMO AI Studio - Full Stack Launcher
# Pure React Frontend (Vite) + FastAPI Python Backend
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BACKEND_DIR="$DIR/backend"
FRONTEND_DIR="$DIR/frontend"

# Robust Python discovery across platforms (conda, venv, system)
PYTHON_BIN=""
for candidate in \
    "$PYTHON" \
    "$VIRTUAL_ENV/bin/python" \
    "$CONDA_PREFIX/bin/python" \
    "/usr/local/Caskroom/miniforge/base/envs/deeplearn_arm/bin/python" \
    "$(command -v python3 2>/dev/null)" \
    "$(command -v python 2>/dev/null)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR (127): Python interpreter not found."
    echo "Please ensure Python 3.10+ is installed and available in your PATH."
    exit 127
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "ERROR (127): npm command not found."
    echo "Please install Node.js (v18+) to run the frontend."
    exit 127
fi

# Detect available port for backend
BACKEND_PORT=8000
if lsof -i :8000 >/dev/null 2>&1; then
    echo "Port 8000 is already in use; switching backend to port 8001..."
    BACKEND_PORT=8001
fi

echo "=============================================================="
echo " Starting Massive MIMO AI Channel Studio"
echo " Backend: Python FastAPI ($PYTHON_BIN) on port $BACKEND_PORT"
echo " Frontend: Pure JavaScript React (Vite) on port 5173"
echo "=============================================================="

# 1. Start Python FastAPI backend
echo "[1/2] Starting Backend API on http://localhost:$BACKEND_PORT..."
cd "$BACKEND_DIR"
"$PYTHON_BIN" -m uvicorn main:app --host 0.0.0.0 --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Trap cleanup on exit
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
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

VITE_API_BASE="http://localhost:$BACKEND_PORT" npm run dev -- --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!

echo "=============================================================="
echo " Studio is LIVE!"
echo " Web UI:     http://localhost:5173"
echo " API Docs:   http://localhost:8000/docs"
echo "=============================================================="
echo "Press Ctrl+C to stop both servers."

wait
