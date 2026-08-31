# Vendored RoboTwin

This directory vendors code from the upstream RoboTwin repository:

- Upstream project: https://github.com/RoboTwin-Platform/RoboTwin
- Upstream commit: `bf44be51cf5717a5595ce59447f2cf5263d2aa95`
- Upstream license: MIT License

License compliance notes:

- The original upstream license is preserved in [`LICENSE`](./LICENSE).
- Files copied from RoboTwin remain subject to the MIT License in this directory.
- The locally maintained WorldScape policy implementation lives outside this
  directory at `experiments/robotwin/wsp2_policy` and is linked into
  `policy/wsp2_policy` at runtime.
- If code is later copied from any upstream subdirectory with an additional license notice, the corresponding license file and attribution must also be preserved.

Local modifications:

- RoboTwin is vendored under `third_party/RoboTwin` for easier integration with this project.
- Upstream policy implementations, simulator assets, cuRobo, checkpoints, and
  generated outputs are intentionally not redistributed.
- `task_config/` contains real files rather than the machine-local symlink used
  by the reference evaluation checkout.
- `script/eval_policy.py` supports persistent model reuse, chunk-managed
  observations, bounded seed search, and timing callbacks.
- `envs/_base_task.py` provides the evaluation camera composition used by the
  manager, and `envs/robot/planner.py` includes the Warp/cuRobo compatibility
  shim used by the reference environment.
