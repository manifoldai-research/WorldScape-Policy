#!/usr/bin/env bash
set -euo pipefail

# Stage1 mixed three-mode pretraining (Interactive-only, no event memory).
# OmegaConf dot-list overrides may be appended.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-none}"

for required in T2VA_DATA_ROOT GOAL_IMAGE_DATA_ROOT VIDEO_DATA_ROOT; do
    if [[ -z "${!required:-}" ]]; then
        echo "ERROR: $required is required" >&2
        exit 2
    fi
done

export WSP_MODE=interactive
if [[ -z "${NUM_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${GPU_ID:-}" ]]; then
    export NUM_GPUS=8
fi

# shellcheck source=recipes/common/launcher_common.sh
source "$ROOT/recipes/common/launcher_common.sh"
wsp_distributed_train_launch "$ROOT" "$ROOT/configs/pretrain/mixed_three_mode_wan22.yaml" "$@"
