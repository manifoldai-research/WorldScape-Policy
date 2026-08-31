from __future__ import annotations

import copy
import json
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evals.robotwin2.adapter import RoboTwin2Adapter
from evals.robotwin2.checkpoint import load_robotwin2_checkpoint_transform
from worldscape_policy.native_builder import (
    build_wan22_policy_from_checkpoint,
    checkpoint_mode,
    checkpoint_supports_mode,
)
from worldscape_policy.rollout.session import PolicyRuntime
from worldscape_policy.types import InteractionMode, WorldActionOutput

ACTION_HORIZON = 24
JOINT_DIM = 14
DEFAULT_OBSERVATION_INTERVAL = 3
OBSERVATION_HISTORY_FRAMES = (
    ACTION_HORIZON // DEFAULT_OBSERVATION_INTERVAL + 1
)
DEFAULT_VLM_COT_PROMPT = (
    "You are a robot planner. Instructions: {task}. Given the current high-level "
    "task instruction and current head-view observation, predict the next atomic "
    "action subtask for the next second."
)
DEFAULT_T5_PROMPT_TEMPLATE = (
    "A video shows that a robot {instruction} The robot {instruction}"
)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


class WSP2RoboTwinPolicy:
    """RoboTwin ``eval_policy.py`` bridge backed by one persistent WSP2 model."""

    def __init__(self, usr_args: dict[str, Any]) -> None:
        checkpoint = Path(str(usr_args["ckpt_setting"])).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"WSP2 checkpoint not found: {checkpoint}")
        self.device = str(usr_args.get("device", "cuda"))
        primary_mode = checkpoint_mode(checkpoint, validate_artifacts=False)
        self.mode = InteractionMode.parse(usr_args.get("mode", primary_mode.value))
        if not checkpoint_supports_mode(primary_mode, self.mode):
            raise ValueError(
                f"checkpoint mode is {primary_mode.value!r}, "
                f"not requested {self.mode.value!r}"
            )
        action_horizon = int(usr_args.get("action_horizon", ACTION_HORIZON))
        self.replan_steps = int(usr_args.get("replan_steps", ACTION_HORIZON))
        if action_horizon != ACTION_HORIZON:
            raise ValueError(
                "WSP2 RoboTwin policy requires action_horizon=24"
            )
        if not 1 <= self.replan_steps <= ACTION_HORIZON:
            raise ValueError("replan_steps must be between 1 and 24")
        if str(usr_args.get("action_type", "qpos")) != "qpos":
            raise ValueError("WSP2 RoboTwin policy supports qpos actions only")
        self.observation_interval = int(
            usr_args.get("observation_interval", DEFAULT_OBSERVATION_INTERVAL)
        )
        if self.observation_interval != DEFAULT_OBSERVATION_INTERVAL:
            raise ValueError("WSP2 RoboTwin policy requires observation_interval=3")
        if self.replan_steps % self.observation_interval:
            raise ValueError(
                "replan_steps must be divisible by observation_interval"
            )
        self.memory_reset_chunks = int(usr_args.get("memory_reset_chunks", 2))
        if self.memory_reset_chunks < 0:
            raise ValueError("memory_reset_chunks cannot be negative")
        self.frames_per_replan = OBSERVATION_HISTORY_FRAMES
        self.skip_get_obs_within_replan = _parse_bool(
            usr_args.get("skip_get_obs_within_replan", True)
        )
        self.manages_action_chunks = True

        transform = load_robotwin2_checkpoint_transform(checkpoint)
        self.vlm_history_num_frames = int(
            usr_args.get("vlm_history_num_frames", 8)
        )
        if self.vlm_history_num_frames < 1:
            raise ValueError("vlm_history_num_frames must be positive")
        self.vlm_cot_prompt = str(
            usr_args.get("vlm_cot_prompt", DEFAULT_VLM_COT_PROMPT)
        )
        self.t5_prompt_template = str(
            usr_args.get("t5_prompt_template", DEFAULT_T5_PROMPT_TEMPLATE)
        )
        try:
            self.vlm_cot_prompt.format(task="test")
            self.t5_prompt_template.format(instruction="test")
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "RoboTwin prompt templates must contain valid {task} and "
                "{instruction} placeholders"
            ) from exc
        policy = build_wan22_policy_from_checkpoint(
            checkpoint,
            visual_input_range="zero_one",
            device=self.device,
            expected_mode=self.mode,
            vlm_cot_prompt=self.vlm_cot_prompt,
            validate_checkpoint_artifacts=_parse_bool(
                usr_args.get("validate_checkpoint_artifacts", False)
            ),
        )
        if self.mode is InteractionMode.AUTO:
            kernel = getattr(getattr(policy, "wam", None), "_numerical_kernel", None)
            kernel_config = getattr(kernel, "config", None)
            if kernel_config is None:
                raise RuntimeError("Auto mode could not locate the WAM kernel config")
            kernel.config = replace(kernel_config, cfg_scale=1.0)
        self.runtime = PolicyRuntime(policy)
        self.adapter = RoboTwin2Adapter(
            checkpoint_transform=transform,
            vlm_history_num_frames=self.vlm_history_num_frames,
        )
        self.generator = torch.Generator(device=torch.device(self.device)).manual_seed(
            int(usr_args.get("seed", 0))
        )
        self.log_inference = _parse_bool(usr_args.get("log_inference", True))
        self.pending_actions: deque[np.ndarray] = deque()
        self.pending_output: WorldActionOutput | None = None
        self.executed_in_chunk = 0
        self.chunk_frames: tuple[deque[np.ndarray], ...] | None = None
        self.inference_seconds: list[float] = []
        self.simulation_seconds = 0.0
        self.episode_started = time.perf_counter()
        self.reset()
        print(
            "[wsp2-config] "
            + json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "mode": self.mode.value,
                    "device": self.device,
                    "action_horizon": ACTION_HORIZON,
                    "replan_steps": self.replan_steps,
                    "memory_reset_chunks": self.memory_reset_chunks,
                },
                sort_keys=True,
            )
        )

    @staticmethod
    def _camera_frames(
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = observation["observation"]
        frames = tuple(
            np.asarray(data[name]["rgb"], dtype=np.uint8)
            for name in ("head_camera", "left_camera", "right_camera")
        )
        if any(frame.ndim != 3 or frame.shape[-1] != 3 for frame in frames):
            raise ValueError(
                "RoboTwin camera observations must be individual HWC RGB frames"
            )
        return frames  # type: ignore[return-value]

    def _policy_observation(
        self,
        current: dict[str, Any],
        *,
        append_current: bool = True,
    ) -> dict[str, Any]:
        if self.chunk_frames is None:
            result = copy.deepcopy(current)
            cameras = result["observation"]
            for name, frame in zip(
                ("head_camera", "left_camera", "right_camera"),
                self._camera_frames(current),
            ):
                cameras[name]["rgb"] = np.repeat(
                    frame[None, ...],
                    self.frames_per_replan,
                    axis=0,
                )
            return result
        if append_current:
            self._append_history_observation(current)
        counts = tuple(len(history) for history in self.chunk_frames)
        if len(set(counts)) != 1 or counts[0] > self.frames_per_replan:
            raise RuntimeError(
                "Invalid rolling observation history counts: "
                f"{counts}"
            )
        result = copy.deepcopy(current)
        cameras = result["observation"]
        for name, history in zip(
            ("head_camera", "left_camera", "right_camera"),
            self.chunk_frames,
        ):
            values = list(history)
            padded = values + [values[-1]] * (
                self.frames_per_replan - len(values)
            )
            cameras[name]["rgb"] = np.stack(padded, axis=0)
        return result

    def _append_history_observation(self, observation: dict[str, Any]) -> None:
        if self.chunk_frames is None:
            raise RuntimeError("Missing observation buffer for active chunk")
        for history, frame in zip(
            self.chunk_frames,
            self._camera_frames(observation),
        ):
            history.append(frame.copy())

    def _fill_action_queue(
        self,
        task_env: Any,
        observation: dict[str, Any],
        *,
        append_current: bool = True,
    ) -> None:
        native_observation = self._policy_observation(
            observation,
            append_current=append_current,
        )
        policy_observation = self.adapter.observation(
            native_observation,
            device=self.device,
        )
        instruction = str(task_env.get_instruction())
        if self.mode is InteractionMode.INTERACTIVE:
            instruction = self.t5_prompt_template.format(
                instruction=instruction.lower()
            )
        prompts = self.adapter.prompt(instruction, mode=self.mode)
        started = time.perf_counter()
        output = self.runtime.predict(
            observation=policy_observation,
            prompts=prompts,
            generator=self.generator,
        )
        inference_s = time.perf_counter() - started
        actions = self.adapter.action(output)
        if actions.shape != (ACTION_HORIZON, JOINT_DIM):
            self.runtime.discard()
            raise ValueError(
                f"WSP2 policy produced {actions.shape}, expected (24, 14)"
            )
        executed_actions = actions[: self.replan_steps]
        self.pending_output = output
        self.pending_actions.extend(executed_actions)
        self.executed_in_chunk = 0
        # Keep a rolling, training-aligned nine-frame window. With a 12-step
        # replan interval, adjacent windows overlap by five sampled frames.
        if self.chunk_frames is None:
            self.chunk_frames = tuple(
                deque([frame.copy()], maxlen=self.frames_per_replan)
                for frame in self._camera_frames(observation)
            )
        self.inference_seconds.append(inference_s)
        if self.log_inference:
            print(
                f"[wsp2-infer] count={len(self.inference_seconds)} "
                f"infer_s={inference_s:.3f}"
            )

    def step(self, task_env: Any, observation: dict[str, Any] | None) -> None:
        if self.pending_actions or self.pending_output is not None:
            raise RuntimeError("A policy step must start at an action-chunk boundary")

        append_current = True
        if observation is None:
            if self._latest_observation is None:
                observation = task_env.get_obs()
            else:
                observation = self._latest_observation
                append_current = not self._latest_observation_buffered
        self._latest_observation = None
        self._latest_observation_buffered = False
        self._fill_action_queue(
            task_env,
            observation,
            append_current=append_current,
        )

        while self.pending_actions:
            action = self.pending_actions.popleft()
            started = time.perf_counter()
            task_env.take_action(action, action_type="qpos")
            self.simulation_seconds += time.perf_counter() - started
            self.executed_in_chunk += 1

            if self.executed_in_chunk % self.observation_interval == 0:
                sampled = task_env.get_obs()
                self._append_history_observation(sampled)
                self._latest_observation = sampled
                self._latest_observation_buffered = True
            elif not self.skip_get_obs_within_replan:
                # Preserve the legacy option to refresh simulator observations
                # after every action. Only stride-aligned frames enter the
                # model's training-aligned nine-frame history.
                task_env.get_obs()

        step_limit = getattr(task_env, "step_lim", None)
        episode_finished = bool(getattr(task_env, "eval_success", False))
        if step_limit is not None:
            episode_finished = episode_finished or int(
                getattr(task_env, "take_action_cnt", 0)
            ) >= int(step_limit)
        if episode_finished:
            self.pending_actions.clear()
            if self.pending_output is not None:
                self.runtime.discard()
                self.pending_output = None
            return

        if self.pending_output is None:
            raise RuntimeError("Missing transactional output at chunk boundary")
        # The WAM candidate memory does not store an unexecuted action suffix;
        # the next call prefills it from the real rolling observation window.
        self.runtime.commit(self.pending_output)
        self.pending_output = None
        self.memory_chunks_since_reset += 1
        if (
            self.memory_reset_chunks
            and self.memory_chunks_since_reset >= self.memory_reset_chunks
        ):
            self.runtime.reset(self.mode.value)
            self.chunk_frames = None
            self._latest_observation_buffered = False
            self.memory_chunks_since_reset = 0

    def reset(self) -> None:
        if getattr(self, "runtime", None) is not None:
            if self.runtime.has_pending_prediction:
                self.runtime.discard()
            self.runtime.reset(self.mode.value)
        adapter_reset = getattr(getattr(self, "adapter", None), "reset", None)
        if callable(adapter_reset):
            adapter_reset()
        self.pending_actions.clear()
        self.pending_output = None
        self.executed_in_chunk = 0
        self.chunk_frames = None
        self._latest_observation: dict[str, Any] | None = None
        self._latest_observation_buffered = False
        self.memory_chunks_since_reset = 0
        self.inference_seconds = []
        self.simulation_seconds = 0.0
        self.episode_started = time.perf_counter()

    def record_episode_result(self, **record: Any) -> dict[str, Any]:
        count = len(self.inference_seconds)
        timing = {
            "infer_count": count,
            "infer_s": sum(self.inference_seconds),
            "infer_s_mean": (
                sum(self.inference_seconds) / count if count else 0.0
            ),
            "sim_s": self.simulation_seconds,
            "episode_s": time.perf_counter() - self.episode_started,
        }
        print("[wsp2-timing] " + json.dumps(timing, sort_keys=True))
        return {**record, "timing": timing}

    def prepare_evaluation_job(self) -> None:
        self.reset()


def get_model(usr_args: dict[str, Any]) -> WSP2RoboTwinPolicy:
    return WSP2RoboTwinPolicy(usr_args)


def eval(
    task_env: Any,
    model: WSP2RoboTwinPolicy,
    observation: dict[str, Any] | None,
) -> None:
    model.step(task_env, observation)


def reset_model(model: WSP2RoboTwinPolicy) -> None:
    model.reset()


def prepare_model_for_evaluation_job(
    model: WSP2RoboTwinPolicy,
) -> None:
    """Reset job-scoped state without reloading checkpoint weights."""

    model.prepare_evaluation_job()


__all__ = [
    "WSP2RoboTwinPolicy",
    "eval",
    "get_model",
    "prepare_model_for_evaluation_job",
    "reset_model",
]
