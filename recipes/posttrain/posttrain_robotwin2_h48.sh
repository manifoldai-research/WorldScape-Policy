#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${PRETRAINED_MODEL_PATH:?Set PRETRAINED_MODEL_PATH to a native initialization checkpoint}"
: "${DATA_ROOT:?Set DATA_ROOT to the RoboTwin LeRobot dataset root}"
: "${ZSCORE_STATS_PATH:?Set ZSCORE_STATS_PATH to dataset_stats.json}"

export WSP_POSTTRAIN_CONFIG="$ROOT/configs/posttrain/robotwin2_h48_auto.yaml"
export WSP_INITIALIZATION="${WSP_INITIALIZATION:-checkpoint_overlay}"
export ACTION_HORIZON=48
export WAM_MAX_CHUNK_SIZE=1
export WAM_NUM_FRAMES=9
export VLM_HISTORY_NUM_FRAMES=4
export VLM_HISTORY_STRIDE=48
export VLM_HISTORY_WINDOW=192
export RUN_NAME="${RUN_NAME:-robotwin2_mosaic_T_h48_auto}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"

# An existing OUTPUT_DIR may resume a complete checkpoint, as in the base recipe.
exec bash "$SCRIPT_DIR/posttrain_robotwin2.sh" "$@"
