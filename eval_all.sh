#!/usr/bin/env bash
# ==============================================================================
# NCRT & NCRN: Spatio-Temporal Massive MIMO Channel Estimation
# Master Evaluation Pipeline — Evaluates pretrained models & reproduces tables/figures:
#   - Table 1: Headline combined TDL-64 test set (N=500, 95% CI)
#   - Table 2: Three-seed NCRT vs. NCRN replication
#   - Table 6: Frame-order control at 9 µs spacing (Permuted / Reversed)
#   - Table 7: Link-level BER & Spectral Efficiency (16-QAM, LMMSE)
#   - Tables 9 & 10: Wang et al. (IEEE TCCN 2025) Sa-DLCS & ISTA shrinkage
#   - Table 11: Phase-aligned average control at 9 µs pilot spacing
#   - Figures 2 to 10: Publication-grade vector PDF figures
#
# Usage:
#   bash eval_all.sh                 # Run full evaluation suite
#   bash eval_all.sh --figures-only  # Regenerate only paper figures
# ==============================================================================
set -e
cd "$(dirname "$0")"

# Quick help check
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            echo "Usage: bash eval_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --figures-only   Regenerate all figures from existing CSV results"
            echo "  --help, -h       Show this help message"
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
"$PYTHON" -c "import tensorflow, numpy, scipy, matplotlib, pandas" 2>/dev/null || {
    echo "ERROR: Missing required Python packages for $PYTHON."
    echo "Please install dependencies first: pip install -r requirements.txt"
    exit 1
}

FIGURES_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --figures-only)
            FIGURES_ONLY=1
            ;;
        *)
            echo "Unknown option: $arg (run with --help for options)"
            exit 1
            ;;
    esac
done

START_TIME=$(date +%s)
echo "======================================================================"
echo "  REPRODUCING BENCHMARKS & FIGURES FOR VEHICULAR COMMUNICATIONS"
echo "  Interpreter : $("$PYTHON" --version)"
echo "  Checkpoints : ./models/"
echo "  Outputs     : ./outputs/csv/ and ./outputs/figures/"
echo "======================================================================"

if [ "$FIGURES_ONLY" -eq 0 ]; then
    echo ""
    echo "─── [1/6] Evaluating Table 1: Headline Combined TDL-64 (95% CI) ───"
    "$PYTHON" scripts/rerun_task4_ci95_ratio.py

    echo ""
    echo "─── [2/6] Evaluating Table 2: Three-Seed NCRT vs. NCRN Replication ───"
    "$PYTHON" scripts/train_eval_three_seed_ncrn_ncrt.py --eval-only

    echo ""
    echo "─── [3/6] Evaluating Tables 9 & 10: Wang et al. & ISTA Shrinkage Ablation ───"
    "$PYTHON" scripts/eval_wang_and_mu_lambda_ablations.py

    echo ""
    echo "─── [4/6] Evaluating Tables 6 & 11: High-Mobility 9 µs Controls ───"
    "$PYTHON" scripts/eval_9us_frame_shuffle_control.py
    "$PYTHON" scripts/eval_9us_phase_aligned_average.py

    echo ""
    echo "─── [5/6] Evaluating Table 7: Link-Level BER & Spectral Efficiency ───"
    "$PYTHON" scripts/link_level_ber_se.py
else
    echo "Skipping numerical evaluation (--figures-only passed)."
fi

echo ""
echo "─── [6/6] Generating Publication Figures 2 through 10 ───"
"$PYTHON" plotting/generate_clean_journal_figures.py
"$PYTHON" plotting/plot_10_tdl_scenarios.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MIN=$((ELAPSED / 60))
SEC=$((ELAPSED % 60))

echo ""
echo "======================================================================"
echo "  EVALUATION COMPLETE — Total time: ${MIN}m ${SEC}s"
echo "  Numerical CSVs : ./outputs/csv/"
echo "  Vector Figures : ./outputs/figures/"
echo "======================================================================"
