#!/usr/bin/env bash
# ==============================================================================
# Massive MIMO AI Studio Launcher
# Starts FastAPI backend (port 8000/8001) and Vite frontend (port 5173)
# ==============================================================================
set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "  Starting Massive MIMO AI Channel Studio"
echo "======================================================================"

# 1. Discover Python interpreter
PYTHON=""
for candidate in \
    "/usr/local/Caskroom/miniforge/base/envs/deeplearn_arm/bin/python" \
    "$VIRTUAL_ENV/bin/python" \
    "$CONDA_PREFIX/bin/python" \
    "$(command -v python3 2>/dev/null)" \
    "$(command -v python 2>/dev/null)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR (127): Python interpreter not found."
    echo "Please install Python 3.10+ or activate your conda/virtual environment."
    exit 127
fi

# 2. Check Node / npm
if ! command -v npm >/dev/null 2>&1; then
    echo "ERROR (127): npm command not found."
    echo "Please install Node.js (v18+) to run the frontend."
    exit 127
fi

# 3. Detect available backend port
PORT=8000
if lsof -i :8000 >/dev/null 2>&1; then
    echo "Port 8000 is occupied; switching backend to port 8001..."
    PORT=8001
fi

echo "==> Using Python: $PYTHON"
echo "==> Backend will run on: http://localhost:$PORT"

# 4. Launch backend in background
cd "$ROOT_DIR/web_app/backend"
"$PYTHON" -m uvicorn main:app --host 0.0.0.0 --port "$PORT" &
BACKEND_PID=$!

# Trap cleanup on exit
cleanup() {
    echo ""
    echo "Stopping background processes..."
    kill "$BACKEND_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 5. Launch frontend
cd "$ROOT_DIR/web_app/frontend"
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo "==> Launching Vite frontend on http://localhost:5173..."
VITE_API_BASE="http://localhost:$PORT" npm run dev -- --host 0.0.0.0 --port 5173
