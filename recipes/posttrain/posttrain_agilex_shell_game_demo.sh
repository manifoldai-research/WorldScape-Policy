#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the Shell Game dataset}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [[ ! -f "$DATA_ROOT/meta/episodes.jsonl" ]]; then
    echo "ERROR: Expected merged dataset meta at $DATA_ROOT/meta/episodes.jsonl"
    exit 1
fi

export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-none}"
export LEARNING_RATE="${LEARNING_RATE:-6e-5}"
export LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
export MAX_STEPS="${MAX_STEPS:-100000}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
if [[ -z "${NUM_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${GPU_ID:-}" ]]; then
    export NUM_GPUS=8
fi
export QWEN_VARIANT="${QWEN_VARIANT:-qwen3-4b}"

export WSP_TASK="shell-game-demo"
export RUN_NAME="${RUN_NAME:-baseline}"
export WSP_MODE="${WSP_MODE:-interactive}"
export VISUAL_PROMPT="demo"
export NATIVE_DATASET_NAME="worldscape_hdf5_demo"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/${WSP_TASK}/${WSP_MODE}-${VISUAL_PROMPT}/${RUN_NAME}}"
export WSP_AUTO_DOWNLOAD="${WSP_AUTO_DOWNLOAD:-true}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-world-scape-policy}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"
export HISTORY_NUM_FRAMES="${HISTORY_NUM_FRAMES:-2}"
export HISTORY_WINDOW="${HISTORY_WINDOW:-48}"
export SHARD_SIZE="${SHARD_SIZE:-10000}"
export SHARD_SAMPLING_RATE="${SHARD_SAMPLING_RATE:-0.1}"
export NUM_SHARDS_TO_SAMPLE="${NUM_SHARDS_TO_SAMPLE:-1048576}"
export EXEC_EARLY_SAMPLING_ENABLED="${EXEC_EARLY_SAMPLING_ENABLED:-true}"
export EXEC_EARLY_RATIO="${EXEC_EARLY_RATIO:-0.4}"
export EXEC_EARLY_WEIGHT="${EXEC_EARLY_WEIGHT:-3.0}"

exec "$ROOT/recipes/common/posttrain_agilex.sh" \
    "data_loader.dataset_kwargs.dataset_roots=[\"${DATA_ROOT}\"]" \
    "data_loader.dataset_kwargs.source_names=[SHELL_GAME]" \
    "data_loader.dataset_kwargs.mixture_weights=[1.0]" \
    "data_loader.dataset_kwargs.shard_size=${SHARD_SIZE}" \
    "data_loader.dataset_kwargs.shard_sampling_rate=${SHARD_SAMPLING_RATE}" \
    "data_loader.dataset_kwargs.num_shards_to_sample=${NUM_SHARDS_TO_SAMPLE}" \
    "data_loader.dataset_kwargs.exec_early_sampling_enabled=${EXEC_EARLY_SAMPLING_ENABLED}" \
    "data_loader.dataset_kwargs.exec_early_ratio=${EXEC_EARLY_RATIO}" \
    "data_loader.dataset_kwargs.exec_early_weight=${EXEC_EARLY_WEIGHT}" \
    "data_loader.action_dim_mask=[1,1,1,1,1,1,1,1,1,1,5,5,5,5,5,5,5,5,5,10]" \
    "data_loader.shuffle=false" \
    "$@"
