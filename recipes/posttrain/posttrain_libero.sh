#!/usr/bin/env bash
set -euo pipefail

# LIBERO generalist post-training (multi-task LeRobot mixture). The selected
# dataset must match VISUAL_PROMPT; OmegaConf dot-list overrides may be appended.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-none}"
export LEARNING_RATE="${LEARNING_RATE:-6e-5}"
export LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
if [[ -z "${NUM_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${GPU_ID:-}" ]]; then
    export NUM_GPUS=8
fi
export QWEN_VARIANT="${QWEN_VARIANT:-qwen3-4b}"

export WSP_TASK="${WSP_TASK:-libero}"
export RUN_NAME="${RUN_NAME:-baseline}"
export WSP_MODE="${WSP_MODE:-interactive}"
export VISUAL_PROMPT="${VISUAL_PROMPT:-demo}"
export NATIVE_DATASET_NAME="${NATIVE_DATASET_NAME:-worldscape_lerobot_demo}"
: "${DATA_ROOT:?Set DATA_ROOT to the LIBERO LeRobot dataset}"
export DATA_ROOT

# shellcheck source=recipes/common/launcher_common.sh
source "$ROOT/recipes/common/launcher_common.sh"
wsp_distributed_train_launch "$ROOT" "$ROOT/configs/posttrain/libero.yaml" "$@"
