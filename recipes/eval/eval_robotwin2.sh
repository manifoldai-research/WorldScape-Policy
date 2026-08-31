#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -z "${WSP_PYTHON:-}" ]]; then
    if [[ -n "${CONDA_ENV:-}" ]]; then
        WSP_PYTHON="$CONDA_ENV/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" ]]; then
        CONDA_ENV="$CONDA_PREFIX"
        WSP_PYTHON="$CONDA_PREFIX/bin/python"
    else
        WSP_PYTHON="$(command -v python || true)"
    fi
fi
if [[ -z "$WSP_PYTHON" || ! -x "$WSP_PYTHON" ]]; then
    echo "ERROR: Set CONDA_ENV or WSP_PYTHON to the RoboTwin Python environment" >&2
    exit 1
fi
if [[ -n "${CONDA_ENV:-}" ]]; then
    export PATH="$CONDA_ENV/bin:$PATH"
fi
export CONDA_ENV WSP_PYTHON

export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$ROOT/third_party/RoboTwin}"
export ROBOTWIN2_EVAL_MODEL_PATH="${ROBOTWIN2_EVAL_MODEL_PATH:-}"
export WORLDSCAPE_CHECKPOINT="$ROBOTWIN2_EVAL_MODEL_PATH"
export WSP_MODE="${WSP_MODE:-auto}"
export ROBOTWIN_EPISODES_PER_TASK="${ROBOTWIN_EPISODES_PER_TASK:-100}"
export ROBOTWIN_MEMORY_RESET_CHUNKS="${ROBOTWIN_MEMORY_RESET_CHUNKS:-0}"
# Match posttrain_robotwin2.sh: four VLM anchors at a 24-action stride.
export ROBOTWIN_VLM_HISTORY_NUM_FRAMES="${ROBOTWIN_VLM_HISTORY_NUM_FRAMES:-4}"
export ROBOTWIN_REPLAN_STEPS="${ROBOTWIN_REPLAN_STEPS:-24}"

case "$WSP_MODE" in
    interactive|auto) ;;
    *) echo "ERROR: WSP_MODE must be interactive or auto" >&2; exit 2 ;;
esac

if [[ -z "${ROBOTWIN_GPU_IDS:-}" ]]; then
    mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    if (( ${#gpu_ids[@]} == 0 )); then
        echo "ERROR: No available NVIDIA GPUs were detected" >&2
        exit 1
    fi
    gpu_csv="$(IFS=,; echo "${gpu_ids[*]}")"
    export ROBOTWIN_GPU_IDS="[$gpu_csv]"
    export CUDA_VISIBLE_DEVICES="$gpu_csv"
fi

export WORLDSCAPE_EVAL_OUTPUT="${WORLDSCAPE_EVAL_OUTPUT:-./evaluate_results/robotwin/$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$WSP_PYTHON" "$ROOT/experiments/robotwin/run_robotwin_manager.py" \
    MULTIRUN.eval_phases="${ROBOTWIN_EVAL_PHASE:-both}" \
    MULTIRUN.gpu_ids="$ROBOTWIN_GPU_IDS" \
    MULTIRUN.resume="${ROBOTWIN_RESUME:-false}" \
    "$@"
