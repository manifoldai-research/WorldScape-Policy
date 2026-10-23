from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch

from evals.common.simulator import ObservationMapping, SimulatorAdapter
from worldscape_policy.checkpoint.transforms import NativeCheckpointTransform
from worldscape_policy.types import ObservationBatch, WorldActionOutput
from worldscape_policy.data.video_preprocessing import (
    preprocess_vlm_video,
    validate_vlm_preprocessing,
)

POLICY_VIDEO_NATIVE_WIDTH = 320
POLICY_VIDEO_NATIVE_HEIGHT = 160
DEFAULT_VLM_HISTORY_NUM_FRAMES = 8


class RoboTwin2Adapter(SimulatorAdapter):
    """Map RoboTwin 2 observations and actions to the WorldScape API."""

    def __init__(
        self,
        *,
        camera_keys: tuple[str, ...] = (
            "observation.head_camera.rgb",
            "observation.left_camera.rgb",
            "observation.right_camera.rgb",
        ),
        state_keys: tuple[str, ...] = ("joint_action.vector",),
        head_camera_key: str | None = "observation.head_camera.rgb",
        embodiment_id: int = 0,
        action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        checkpoint_transform: NativeCheckpointTransform | None = None,
        image_width: int = POLICY_VIDEO_NATIVE_WIDTH,
        image_height: int = POLICY_VIDEO_NATIVE_HEIGHT,
        vlm_history_num_frames: int = DEFAULT_VLM_HISTORY_NUM_FRAMES,
        preserve_camera_resolution: bool = False,
        action_horizon: int = 24,
        observation_interval: int | None = None,
        vlm_history_stride: int | None = None,
        vlm_preprocessing: dict[str, Any] | None = None,
    ) -> None:
        if vlm_history_num_frames < 1:
            raise ValueError("vlm_history_num_frames must be positive")
        if action_horizon not in {24, 48}:
            raise ValueError("RoboTwin2 action_horizon must be 24 or 48")
        self.action_horizon = int(action_horizon)
        self.observation_interval = (
            action_horizon // 8 if observation_interval is None else int(observation_interval)
        )
        self.vlm_history_stride = (
            action_horizon if vlm_history_stride is None else int(vlm_history_stride)
        )
        if self.observation_interval != action_horizon // 8:
            raise ValueError("observation_interval must produce nine frames per action block")
        if self.vlm_history_stride < 1 or self.vlm_history_stride % self.observation_interval:
            raise ValueError("vlm_history_stride must be a positive multiple of observation_interval")
        self.vlm_preprocessing = validate_vlm_preprocessing(vlm_preprocessing)
        super().__init__(
            ObservationMapping(
                camera_keys=camera_keys,
                state_keys=state_keys,
                head_camera_key=head_camera_key,
                embodiment_id=embodiment_id,
            ),
            action_transform=action_transform,
            image_size=(image_height, image_width),
            image_resize_interpolation="area",
            preserve_camera_resolution=preserve_camera_resolution,
            head_video_transform=(
                partial(
                    preprocess_vlm_video,
                    config=self.vlm_preprocessing or {"mode": "raw"},
                )
                if preserve_camera_resolution or self.vlm_preprocessing is not None
                else None
            ),
        )
        self.checkpoint_transform = checkpoint_transform
        self.vlm_history_num_frames = int(vlm_history_num_frames)
        self._vlm_frames: dict[int, torch.Tensor] = {}
        self._first_vlm_frame: torch.Tensor | None = None
        self._observation_step: int | None = None

    def set_checkpoint_transform(
        self, transform: NativeCheckpointTransform
    ) -> None:
        self.checkpoint_transform = transform

    def observation(
        self,
        value,
        *,
        device: torch.device | str = "cpu",
        observation_step: int | None = None,
    ) -> ObservationBatch:
        batch = super().observation(value, device=device)
        if batch.proprioception.shape[-1] != 14:
            raise ValueError(
                "RoboTwin2 joint state must have width 14, got "
                f"{batch.proprioception.shape[-1]}"
            )
        if batch.images.shape[2] != 3:
            raise ValueError("RoboTwin2 requires exactly three camera views")
        if batch.images.shape[0] != 1:
            raise ValueError("RoboTwin2 evaluation supports batch size 1 only")
        frame_count = int(batch.images.shape[1])
        if frame_count not in {1, 9}:
            raise ValueError(
                "RoboTwin2 observation history must contain 1 or 9 frames, "
                f"got {frame_count}"
            )
        if observation_step is None:
            observation_step = (
                0 if self._observation_step is None
                else self._observation_step + self.action_horizon
            )
        if observation_step < 0 or (
            self._observation_step is not None and observation_step < self._observation_step
        ):
            raise ValueError("observation_step must be nonnegative and chronological")
        self._observation_step = observation_step
        heads = self._last_head_video.detach().cpu()
        if self._first_vlm_frame is None:
            self._first_vlm_frame = heads[-1].clone()
        # Record actual sampled frames, so a partial replan does not change the
        # configured history stride. Negative times are initial left padding.
        for index, frame in enumerate(heads):
            step = observation_step - (len(heads) - 1 - index) * self.observation_interval
            if step >= 0:
                self._vlm_frames.setdefault(step, frame.clone())
        self._vlm_frames[observation_step] = heads[-1].clone()
        targets = [
            observation_step - index * self.vlm_history_stride
            for index in reversed(range(self.vlm_history_num_frames))
        ]
        anchors = []
        for target in targets:
            if target < 0:
                anchors.append(self._first_vlm_frame)
            elif target in self._vlm_frames:
                anchors.append(self._vlm_frames[target])
            else:
                raise ValueError(
                    f"Missing VLM history frame at simulator step {target}; "
                    "provide every stride-aligned observation history"
                )
        history = torch.stack(anchors, dim=0).unsqueeze(0).to(device=batch.images.device)
        oldest = max(0, targets[0])
        self._vlm_frames = {
            step: frame for step, frame in self._vlm_frames.items() if step >= oldest
        }
        batch.vlm_history_images = history
        batch.vlm_history_mask = torch.ones(
            history.shape[:2], dtype=torch.bool, device=history.device
        )
        if self.checkpoint_transform is not None:
            state = self.checkpoint_transform.apply_state(
                {"state.vector": batch.proprioception}
            )
            max_state_dim = self.checkpoint_transform.embodiment.max_state_dim
            if state.shape[-1] > max_state_dim:
                raise ValueError(
                    f"Normalized RoboTwin2 state width {state.shape[-1]} exceeds "
                    f"checkpoint maximum {max_state_dim}"
                )
            batch.proprioception = torch.nn.functional.pad(
                state,
                (0, max_state_dim - state.shape[-1]),
            )
            batch.validate()
        return batch

    def reset(self) -> None:
        """Clear episode-scoped VLM anchor history."""

        self._vlm_frames.clear()
        self._first_vlm_frame = None
        self._observation_step = None

    def action(self, output: WorldActionOutput) -> np.ndarray:
        action = output.require_action().detach().to(device="cpu", dtype=torch.float32)
        if (
            action.ndim != 3
            or action.shape[0] != 1
            or action.shape[1] != self.action_horizon
        ):
            raise ValueError(
                f"RoboTwin2 policy output must have shape [1,{self.action_horizon},D], got "
                f"{tuple(action.shape)}"
            )
        if self.checkpoint_transform is not None:
            action = self.checkpoint_transform.unapply({"action": action[0]})[
                "action.vector"
            ]
            result = np.asarray(action.numpy(), dtype=np.float32)
        else:
            if action.shape[-1] != 14:
                raise ValueError(
                    "RoboTwin2 action must have width 14 without a checkpoint transform"
                )
            result = np.asarray(action[0].numpy(), dtype=np.float32)
        if result.shape != (self.action_horizon, 14):
            raise ValueError(
                f"RoboTwin2 absolute qpos chunk must be [{self.action_horizon},14], got {result.shape}"
            )
        if self._action_transform is not None:
            result = np.asarray(self._action_transform(result))
        return result

__all__ = ["RoboTwin2Adapter"]
