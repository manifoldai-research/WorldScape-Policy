from collections.abc import Callable
from collections import deque

import numpy as np
import torch

from evals.common.simulator import ObservationMapping, SimulatorAdapter
from worldscape_policy.checkpoint.transforms import NativeCheckpointTransform
from worldscape_policy.types import ObservationBatch, WorldActionOutput

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
    ) -> None:
        if vlm_history_num_frames < 1:
            raise ValueError("vlm_history_num_frames must be positive")
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
        )
        self.checkpoint_transform = checkpoint_transform
        self.vlm_history_num_frames = int(vlm_history_num_frames)
        self._vlm_anchor_history: deque[torch.Tensor] = deque(
            maxlen=self.vlm_history_num_frames
        )

    def set_checkpoint_transform(
        self, transform: NativeCheckpointTransform
    ) -> None:
        self.checkpoint_transform = transform

    def observation(
        self,
        value,
        *,
        device: torch.device | str = "cpu",
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
        head_frames = batch.images[0, :, 0]
        vlm_head_frames = head_frames
        if head_frames.shape[-2:] != batch.head_view.shape[-2:]:
            vlm_head_frames = torch.nn.functional.interpolate(
                head_frames,
                size=batch.head_view.shape[-2:],
                mode="area",
            )
        frame_count = int(head_frames.shape[0])
        if frame_count == 1:
            self._vlm_anchor_history.append(
                vlm_head_frames[-1].detach().cpu().clone()
            )
        elif frame_count != 9:
            raise ValueError(
                "RoboTwin2 observation history must contain 1 or 9 frames, "
                f"got {frame_count}"
            )
        else:
            if not self._vlm_anchor_history:
                self._vlm_anchor_history.append(
                    vlm_head_frames[0].detach().cpu().clone()
                )
            self._vlm_anchor_history.append(
                vlm_head_frames[-1].detach().cpu().clone()
            )
        anchors = list(self._vlm_anchor_history)
        anchors = [anchors[0]] * (self.vlm_history_num_frames - len(anchors)) + anchors
        history = torch.stack(anchors[-self.vlm_history_num_frames :], dim=0)
        history = history.unsqueeze(0).to(device=batch.images.device)
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

        self._vlm_anchor_history.clear()

    def action(self, output: WorldActionOutput) -> np.ndarray:
        action = output.require_action().detach().to(device="cpu", dtype=torch.float32)
        if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != 24:
            raise ValueError(
                "RoboTwin2 policy output must have shape [1,24,D], got "
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
        if result.shape != (24, 14):
            raise ValueError(
                f"RoboTwin2 absolute qpos chunk must be [24,14], got {result.shape}"
            )
        if self._action_transform is not None:
            result = np.asarray(self._action_transform(result))
        return result

__all__ = ["RoboTwin2Adapter"]
