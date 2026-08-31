# Evaluation

`wsp-eval` is the common entry point for offline HDF5 replay and guarded native
AgileX evaluation. RoboTwin uses its native manager launcher.
Optional backend packages are imported only after their backend is selected.

## Install

```bash
pip install -e .                         # policy runtime and common CLI
pip install -e ".[robotwin2]"            # RoboTwin 2 adapter dependencies
pip install -e ".[agilex]"               # HDF5/real-robot dependencies
```

RoboTwin source is vendored at `third_party/RoboTwin`, but its assets, cuRobo checkout, and
simulator-specific binary dependencies must be installed separately by
following that directory's README.

## RoboTwin 2

RoboTwin uses its native `script/eval_policy.py` loop. A manager dynamically
assigns task/phase jobs to long-lived GPU workers, each of which loads one WSP2
model and reuses it across jobs.

```bash
export ROBOTWIN2_EVAL_MODEL_PATH=/checkpoint_path
export ROBOTWIN_ROOT="$(pwd)/third_party/RoboTwin"
export ROBOTWIN_GPU_IDS='[0,1,2,3]'
WSP_MODE=auto ./recipes/eval/eval_robotwin2.sh
```

Use `EVALUATION.task_name=adjust_bottle` for a single task. Seed filtering,
expert checks, instruction generation, video capture, and success accounting
remain owned by RoboTwin.

## HDF5 replay

```bash
export WORLDSCAPE_CHECKPOINT=/checkpoints/worldscape
export WORLDSCAPE_HDF5_EPISODE=/data/episode.hdf5
wsp-eval --config configs/eval/agilex.yaml
./recipes/common/eval_agilex.sh
```

Replay is read-only and reports no task success unless the source environment
provides it. CI generates a small HDF5 episode and replays it through the same
backend, so no repository fixture or external data download is needed.
The common AgileX evaluator uses this path when `AGILEX_TRANSPORT=hdf5`; it does
not actuate a robot.

## AgileX

The common evaluator uses the WSP-owned robot contract and writes the common
artifacts plus `action_previews.jsonl`. Use an explicit HDF5 episode for safe
replay:

```bash
WORLDSCAPE_CHECKPOINT=/checkpoints/worldscape \
WORLDSCAPE_HDF5_EPISODE=/data/episode.hdf5 \
AGILEX_TRANSPORT=hdf5 \
./recipes/common/eval_agilex.sh
```

Evaluation reads image range conversion, ordered state/action fields,
normalization statistics, relative-action semantics, per-horizon statistics,
and embodiment IDs from the checksum-validated `transform_bundle.json`.
Evaluation fails closed when that artifact is absent or invalid. Public runtime
entry points accept native WorldScape checkpoint bundles only.

Live hardware requires `backend_config.transport: manifold` and the explicit
`--live-hardware` flag. The standalone wheel packages WorldScape-owned HDF5 and
Manifold transports. The Manifold deployment must additionally provide
`manifold_msg`. Task launchers add `--live-hardware` automatically when
`AGILEX_TRANSPORT=manifold`.

### Task-specific AgileX recipes

All AgileX task launchers resolve the common `configs/eval/agilex.yaml` through
strict mode and visual-prompt profiles:

```bash
AGILEX_TRANSPORT=hdf5 WORLDSCAPE_HDF5_EPISODE=/data/fold.hdf5 \
  ./recipes/eval/eval_agilex_fold_shirt_text.sh

AGILEX_TRANSPORT=hdf5 WORLDSCAPE_HDF5_EPISODE=/data/build-block.hdf5 \
WSP_GOAL_IMAGE=/data/goal.png \
  ./recipes/eval/eval_agilex_build_block_goal.sh

AGILEX_TRANSPORT=hdf5 WORLDSCAPE_HDF5_EPISODE=/data/build-block.hdf5 \
  ./recipes/eval/eval_agilex_build_block_demo.sh

AGILEX_TRANSPORT=hdf5 WORLDSCAPE_HDF5_EPISODE=/data/shell-game.hdf5 \
  ./recipes/eval/eval_agilex_shell_game_demo.sh
```

The Fold Shirt recipe selects `VISUAL_PROMPT=none` and never sends a visual
prompt. The goal recipe sends
exactly one head-camera frame, loaded from an explicit path by default; Python
callers may instead select `source: hdf5` or `source: upload`, and
`source: first_observation` is accepted only with the explicit
`goal_from_first_observation: true` opt-in. The demo recipe uniformly samples
exactly 50 frames. `ctx_head_only: true` produces `V=1`; setting it to false
produces high/left/right `V=3`.

For live demo upload, use `source: transport` with `poll_transport: true`.
Every completed Manifold upload starts a new policy session and clears visual,
WAM, and event memory before the 50-frame prompt is sent once. HDF5 context is
preloaded through the WSP-owned replay transport. The task recipes require the
checksum-validated `agilex` EEF transform fields in exact
high/left/right and left/right state/action order.

## Artifacts

Every backend writes the same versioned directory:

- `config.yaml` or `config.json`: resolved run recipe;
- `summary.json`: aggregate success, subgoal, latency, and per-task metrics;
- `episodes.jsonl`: trial records with suite/task metadata, seed, horizon,
control frequency, success, subgoal metrics, and step latency;
- `per_task.csv`: task success and mean subgoal completion;
- `videos/<episode-id>.mp4`: present when frame capture is enabled.

Recipes fix trial count and seed. Override them only when intentionally
creating a separately named evaluation protocol.
