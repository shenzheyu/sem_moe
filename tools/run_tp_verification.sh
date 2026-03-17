#!/usr/bin/env bash
# Run Sem-MoE TP correctness verification.
# Usage: bash tools/run_tp_verification.sh [modes...]
# Default modes: baseline debug_fallback srs_nccl
set -euo pipefail

# Resolve paths relative to project root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
SCRIPT="$SCRIPT_DIR/verify_tp_correctness.py"
SCHEDULE_DIR="$PROJECT_ROOT/artifacts/qwen35-35b-mmlu-tp4/schedule"
OUTDIR="$PROJECT_ROOT/tp_verify_results"
MODEL="Qwen/Qwen3.5-35B-A3B"
TP=2

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

if [ "$#" -eq 0 ]; then
    MODES=(baseline debug_fallback srs_nccl)
else
    MODES=("$@")
fi

mkdir -p "$OUTDIR"

for mode in "${MODES[@]}"; do
    echo ""
    echo "================================================================"
    echo "  Running mode: $mode"
    echo "================================================================"

    # Clean env
    unset SEM_MOE SEM_MOE_TABLES SEM_MOE_MODE SEM_MOE_DEBUG_FALLBACK SEM_MOE_SRS_BACKEND 2>/dev/null || true

    case "$mode" in
        baseline)
            ;;
        debug_fallback)
            export SEM_MOE=1
            export SEM_MOE_TABLES="$SCHEDULE_DIR"
            export SEM_MOE_MODE=tp
            export SEM_MOE_DEBUG_FALLBACK=1
            ;;
        srs_nccl)
            export SEM_MOE=1
            export SEM_MOE_TABLES="$SCHEDULE_DIR"
            export SEM_MOE_MODE=tp
            export SEM_MOE_DEBUG_FALLBACK=0
            export SEM_MOE_SRS_BACKEND=nccl
            ;;
        srs_triton)
            export SEM_MOE=1
            export SEM_MOE_TABLES="$SCHEDULE_DIR"
            export SEM_MOE_MODE=tp
            export SEM_MOE_DEBUG_FALLBACK=0
            export SEM_MOE_SRS_BACKEND=triton
            ;;
        *)
            echo "Unknown mode: $mode"
            exit 1
            ;;
    esac

    $PYTHON "$SCRIPT" run \
        --run-mode "$mode" \
        --model "$MODEL" \
        --tp-size "$TP" \
        --save "$OUTDIR/${mode}.json" \
        2>&1 | tee "$OUTDIR/${mode}.log"
done

echo ""
echo "================================================================"
echo "  Comparing results"
echo "================================================================"

# Build compare args from available files
COMPARE_FILES=()
for mode in "${MODES[@]}"; do
    if [ -f "$OUTDIR/${mode}.json" ]; then
        COMPARE_FILES+=("$OUTDIR/${mode}.json")
    fi
done

if [ ${#COMPARE_FILES[@]} -ge 2 ]; then
    $PYTHON "$SCRIPT" compare --compare "${COMPARE_FILES[@]}" 2>&1 | tee "$OUTDIR/comparison.log"
else
    echo "Need at least 2 result files to compare."
fi

echo ""
echo "Done. Results in $OUTDIR/"
