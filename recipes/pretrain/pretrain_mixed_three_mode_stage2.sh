#!/usr/bin/env bash
set -euo pipefail

# Stage2 mixed three-mode pretraining (Auto + event memory + prompt schedule).
# PRETRAINED_MODEL_PATH must be a complete native Stage1 portable policy artifact.
# Otherwise checkpoint resolution may auto-resume OUTPUT_DIR or use components.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Set this to a Stage1 checkpoint for component-first Stage2 overlay.
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-none}"

for required in T2VA_DATA_ROOT GOAL_IMAGE_DATA_ROOT VIDEO_DATA_ROOT; do
    if [[ -z "${!required:-}" ]]; then
        echo "ERROR: $required is required for Stage2" >&2
        exit 2
    fi
done

export WSP_MODE=auto
export WSP_INITIALIZATION=checkpoint_overlay
if [[ -z "${NUM_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${GPU_ID:-}" ]]; then
    export NUM_GPUS=8
fi

# shellcheck source=recipes/common/launcher_common.sh
source "$ROOT/recipes/common/launcher_common.sh"
wsp_distributed_train_launch "$ROOT" "$ROOT/configs/pretrain/mixed_three_mode_wan22_stage2.yaml" "$@"
