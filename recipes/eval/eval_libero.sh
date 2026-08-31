#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export WSP_TASK="${WSP_TASK:-libero}"
export RUN_NAME="${RUN_NAME:-baseline}"
export WSP_MODE="${WSP_MODE:-interactive}"
export VISUAL_PROMPT="${VISUAL_PROMPT:-demo}"
# shellcheck source=recipes/common/eval_checkpoint.sh
source "$ROOT/recipes/common/eval_checkpoint.sh"
export WORLDSCAPE_CHECKPOINT="$(
    wsp_resolve_eval_checkpoint \
        "$ROOT/checkpoints/${WSP_TASK}/${WSP_MODE}-${VISUAL_PROMPT}/${RUN_NAME}" \
        "${LIBERO_EVAL_MODEL_PATH:-}"
)"

case "$WSP_MODE" in
    interactive|auto) ;;
    *) echo "ERROR: WSP_MODE must be interactive or auto" >&2; exit 2 ;;
esac

exec "${WSP_EVAL:-wsp-eval}" --config "$ROOT/configs/eval/libero.yaml" "$@"
