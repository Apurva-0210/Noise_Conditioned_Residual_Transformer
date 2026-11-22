#!/usr/bin/env bash
# ==============================================================================
# NCRT & NCRN: Spatio-Temporal Massive MIMO Channel Estimation
# Master Training Pipeline — Trains all 5 base channel estimators from scratch:
#   1. CNN-LSTM   (Temporal ConvLSTM baseline)
#   2. CNN-Sparse (Spatial CNN baseline)
#   3. Sa-DLCS    (Self-Attention DLCS baseline)
#   4. NCRN       (Convolutional Residual Network)
#   5. NCRT       (Nonlinear Channel Residual Transformer - Ours)
#
# Usage:
#   bash train_all.sh                # Full training (~30-45 mins on GPU / Apple Silicon)
#   bash train_all.sh --quick        # Fast smoke-test (2 epochs/model, ~1-2 mins)
#   bash train_all.sh --skip-cnn     # Skip CNN-LSTM and CNN-Sparse baselines
#   bash train_all.sh --skip-ncrt    # Skip NCRT
#   bash train_all.sh --ncrt-only    # Train only NCRT
# ==============================================================================
set -e
cd "$(dirname "$0")"

# Quick help check
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            echo "Usage: bash train_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick, -q     Run fast smoke-test with 2 epochs per model"
            echo "  --skip-cnn      Skip CNN-LSTM and CNN-Sparse baselines"
            echo "  --skip-ncrt     Skip NCRT training"
            echo "  --ncrt-only     Train only NCRT"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
    esac
done

# Intelligent Python discovery: find python interpreter with tensorflow
if [ -z "$PYTHON" ]; then
    for candidate in \
        "/usr/local/Caskroom/miniforge/base/envs/deeplearn_arm/bin/python" \
        "$VIRTUAL_ENV/bin/python" \
        "$CONDA_PREFIX/bin/python" \
        "$(which python3 2>/dev/null)" \
        "$(which python 2>/dev/null)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ] && "$candidate" -c "import tensorflow, numpy" 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    done
    if [ -z "$PYTHON" ]; then
        if command -v python3 >/dev/null 2>&1; then
            PYTHON="python3"
        elif command -v python >/dev/null 2>&1; then
            PYTHON="python"
        else
            echo "ERROR: No Python interpreter found in PATH."
            echo "Please install Python 3.10+ or activate your virtual environment (e.g. conda activate deeplearn_arm)."
            exit 127
        fi
    fi
fi

# Sanity check dependencies
echo "==> Verifying Python environment using: $PYTHON"
"$PYTHON" -c "import tensorflow, numpy, scipy" 2>/dev/null || {
    echo "ERROR: Missing required Python packages for $PYTHON."
    echo "Please install dependencies first: pip install -r requirements.txt"
    exit 1
}

# Parse command line options
QUICK_MODE=0
SKIP_CNN=0
SKIP_NCRT=0
NCRT_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --quick|-q)
            QUICK_MODE=1
            ;;
        --skip-cnn)
            SKIP_CNN=1
            ;;
        --skip-ncrt)
            SKIP_NCRT=1
            ;;
        --ncrt-only)
            NCRT_ONLY=1
            ;;
        *)
            echo "Unknown option: $arg (run with --help for options)"
            exit 1
            ;;
    esac
done

if [ "$QUICK_MODE" -eq 1 ]; then
    echo "─── [MODE: FAST SMOKE TEST (2 epochs per model)] ───"
    export EPOCHS=2
    export STEPS_EPOCH=5
    NCRT_EPOCHS_ARG="--epochs 2 --steps 5"
    NCRN_EPOCHS_ARG="--epochs 2 --steps 5"
else
    echo "─── [MODE: FULL TRAINING] ───"
    NCRT_EPOCHS_ARG=""
    NCRN_EPOCHS_ARG=""
fi

START_TIME=$(date +%s)
echo "======================================================================"
echo "  STARTING COMPLETE TRAINING PIPELINE (64x64 Massive MIMO)"
echo "  Interpreter : $("$PYTHON" --version)"
echo "  Destination : ./models/"
echo "======================================================================"

# 1 & 2: CNN Baselines
if [ "$NCRT_ONLY" -eq 0 ] && [ "$SKIP_CNN" -eq 0 ]; then
    echo ""
    echo "─── [1/5] Training CNN-LSTM TDL (64x64) ───"
    (cd stage2_tdl/cnn_lstm && "$PYTHON" train_cnn_lstm_tdl.py)

    echo ""
    echo "─── [2/5] Training CNN-Sparse TDL (64x64) ───"
    (cd stage2_tdl/cnn_sparse && "$PYTHON" train_cnn_sparse_tdl.py)
else
    echo "─── [1/5 & 2/5] CNN-LSTM and CNN-Sparse SKIPPED ───"
fi

# 3: Sa-DLCS Baseline
if [ "$NCRT_ONLY" -eq 0 ]; then
    echo ""
    echo "─── [3/5] Training Sa-DLCS TDL (64x64) ───"
    (cd stage2_tdl/sa_dlcs && "$PYTHON" train_sa_dlcs_tdl.py)
else
    echo "─── [3/5] Sa-DLCS SKIPPED ───"
fi

# 4: NCRN
if [ "$NCRT_ONLY" -eq 0 ]; then
    echo ""
    echo "─── [4/5] Training NCRN TDL (64x64) ───"
    (cd stage2_tdl/ncrn && "$PYTHON" train_ncrn_tdl.py $NCRN_EPOCHS_ARG)
else
    echo "─── [4/5] NCRN SKIPPED ───"
fi

# 5: NCRT (Ours)
if [ "$SKIP_NCRT" -eq 0 ]; then
    echo ""
    echo "─── [5/5] Training NCRT TDL (64x64) ───"
    (cd stage2_tdl/ncrt && "$PYTHON" train_ncrt_tdl.py $NCRT_EPOCHS_ARG)
else
    echo "─── [5/5] NCRT SKIPPED ───"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MIN=$((ELAPSED / 60))
SEC=$((ELAPSED % 60))

echo ""
echo "======================================================================"
echo "  TRAINING PIPELINE COMPLETE — Total time: ${MIN}m ${SEC}s"
echo "  Checkpoints saved to: ./models/"
echo "======================================================================"
echo "Next step: Run the evaluation suite to reproduce manuscript tables & figures:"
echo "  bash eval_all.sh"
echo "======================================================================"
