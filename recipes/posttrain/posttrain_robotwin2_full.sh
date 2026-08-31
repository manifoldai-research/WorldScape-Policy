#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SINGLE_HIGH_LEVEL_INSTRUCTION_PER_EPISODE=true
export ALIGN_LOSS_WEIGHT=0.001
export LEARNING_RATE="${LEARNING_RATE:-3e-5}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
export MAX_STEPS="${MAX_STEPS:-50000}"
export RUN_NAME="${RUN_NAME:-stage1_posttrain_full}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"

exec "$SCRIPT_DIR/posttrain_robotwin2.sh" "$@"
