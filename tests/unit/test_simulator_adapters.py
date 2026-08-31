import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from evals.libero.adapter import LiberoAdapter
from evals.libero.task_suite import (
    LiberoTaskSuiteEnvironment,
    make_libero_environment,
)
from evals.robotwin2.adapter import RoboTwin2Adapter
from evals.common.simulator import (
    OptionalSimulatorDependencyError,
    SimulatorAdapterProtocol,
    SimulatorEnvironment,
)
from evals.common.suite import EvaluationTask
from worldscape_policy.types import WorldActionOutput


class FakeEnvironment:
    def __init__(self, observation):
        self._observation = observation
        self.actions = []

    def reset(self, **kwargs):
        return self._observation, {"seed": kwargs.get("seed")}

    def step(self, action):
        self.actions.append(action)
        return self._observation, 1.0, False, False, {}


def test_libero_adapter_maps_fake_environment_through_public_schemas():
    fake = FakeEnvironment(
        {
            "agentview_image": np.full((8, 6, 3), 255, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((3, 8, 6), dtype=np.uint8),
            "robot0_joint_pos": np.arange(7, dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.25, 0.75], dtype=np.float32),
        }
    )
    adapter = LiberoAdapter(embodiment_id=9)

    observation = adapter.observation(fake.reset(seed=3))
    prompt = adapter.prompt("open the drawer", mode="interactive")
    action = adapter.action(
        WorldActionOutput(action=torch.arange(14).reshape(1, 2, 7).float())
    )

    assert isinstance(fake, SimulatorEnvironment)
    assert isinstance(adapter, SimulatorAdapterProtocol)
    assert observation.images.shape == (1, 1, 2, 3, 8, 6)
    assert observation.head_view.shape == (1, 1, 3, 8, 6)
    assert observation.proprioception.shape == (1, 1, 9)
    assert observation.embodiment_id.item() == 9
    assert prompt.language_instruction == ["open the drawer"]
    assert prompt.negative_language_instruction == [""]
    assert action.shape == (2, 7)


def test_robotwin2_adapter_supports_nested_fake_observations_and_history():
    video = np.zeros((9, 4, 5, 3), dtype=np.uint8)
    fake = FakeEnvironment(
        {
            "observation": {
                "head_camera": {"rgb": video},
                "left_camera": {"rgb": video + 1},
                "right_camera": {"rgb": video + 2},
            },
            "joint_action": {
                "vector": np.concatenate((np.zeros(7), np.ones(7)))
            },
        }
    )
    adapter = RoboTwin2Adapter()

    observation = adapter.observation(fake.reset())
    prompt = adapter.prompt("stack the blocks", mode="auto")

    assert observation.images.shape == (1, 9, 3, 3, 160, 320)
    assert observation.proprioception.shape == (1, 1, 14)
    assert observation.vlm_history_images.shape == (1, 8, 3, 160, 320)
    assert prompt.vlm_planning_text == ["stack the blocks"]
    assert prompt.negative_vlm_text is None


def test_robotwin2_adapter_resizes_images_with_training_area_interpolation():
    import cv2

    image = np.random.default_rng(0).integers(
        0, 256, size=(240, 320, 3), dtype=np.uint8
    )
    value = {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image},
            "right_camera": {"rgb": image},
        },
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
    }

    observation = RoboTwin2Adapter().observation(value)
    expected = torch.from_numpy(
        cv2.resize(image, (320, 160), interpolation=cv2.INTER_AREA)
    ).permute(2, 0, 1).float().div(255.0)

    torch.testing.assert_close(observation.images[0, 0, 0], expected)


def test_robotwin2_adapter_pads_normalized_state_to_checkpoint_width():
    class FakeCheckpointTransform:
        embodiment = SimpleNamespace(max_state_dim=64)

        @staticmethod
        def apply_state(data):
            return torch.as_tensor(data["state.vector"]).float().add(1.0)

    image = np.zeros((4, 5, 3), dtype=np.uint8)
    value = {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image},
            "right_camera": {"rgb": image},
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }

    observation = RoboTwin2Adapter(
        checkpoint_transform=FakeCheckpointTransform()
    ).observation(value)

    assert observation.proprioception.shape == (1, 1, 64)
    torch.testing.assert_close(
        observation.proprioception[..., :14],
        torch.arange(1, 15, dtype=torch.float32).view(1, 1, 14),
    )
    torch.testing.assert_close(
        observation.proprioception[..., 14:],
        torch.zeros(1, 1, 50),
    )


def test_robotwin2_vlm_history_uses_one_anchor_per_completed_chunk():
    adapter = RoboTwin2Adapter()

    def value(head):
        return {
            "observation": {
                "head_camera": {"rgb": head},
                "left_camera": {"rgb": head},
                "right_camera": {"rgb": head},
            },
            "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
        }

    initial = adapter.observation(
        value(np.full((4, 5, 3), 10, dtype=np.uint8))
    )
    first_chunk = adapter.observation(
        value(
            np.stack(
                [np.full((4, 5, 3), item, dtype=np.uint8) for item in range(20, 29)]
            )
        )
    )
    second_chunk = adapter.observation(
        value(
            np.stack(
                [np.full((4, 5, 3), item, dtype=np.uint8) for item in range(30, 39)]
            )
        )
    )

    torch.testing.assert_close(
        initial.vlm_history_images,
        torch.full_like(initial.vlm_history_images, 10 / 255),
    )
    first_values = first_chunk.vlm_history_images.mean(dim=(0, 2, 3, 4))
    torch.testing.assert_close(
        first_values,
        torch.tensor([10 / 255] * 7 + [28 / 255]),
    )
    torch.testing.assert_close(first_chunk.head_view.mean(), torch.tensor(20 / 255))
    history_values = second_chunk.vlm_history_images.mean(dim=(0, 2, 3, 4))
    torch.testing.assert_close(
        history_values,
        torch.tensor([10 / 255] * 6 + [28 / 255, 38 / 255]),
    )


def test_robotwin2_vlm_history_length_is_configurable():
    adapter = RoboTwin2Adapter(vlm_history_num_frames=4)
    image = np.full((4, 5, 3), 10, dtype=np.uint8)
    value = {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image},
            "right_camera": {"rgb": image},
        },
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
    }

    observation = adapter.observation(value)

    assert observation.vlm_history_images.shape[1] == 4
    assert observation.vlm_history_mask.shape[1] == 4


def test_robotwin2_single_frame_replan_preserves_vlm_history_until_reset():
    adapter = RoboTwin2Adapter(vlm_history_num_frames=4)

    def value(pixel):
        image = np.full((4, 5, 3), pixel, dtype=np.uint8)
        return {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            },
            "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
        }

    adapter.observation(value(10))
    preserved = adapter.observation(value(20))
    preserved_values = preserved.vlm_history_images.mean(dim=(0, 2, 3, 4))
    torch.testing.assert_close(
        preserved_values,
        torch.tensor([10 / 255] * 2 + [10 / 255, 20 / 255]),
    )

    adapter.reset()
    cleared = adapter.observation(value(30))
    torch.testing.assert_close(
        cleared.vlm_history_images,
        torch.full_like(cleared.vlm_history_images, 30 / 255),
    )


def test_robotwin2_action_is_not_modified_after_denormalization():
    class FakeCheckpointTransform:
        embodiment = SimpleNamespace(max_state_dim=64)

        @staticmethod
        def apply_state(data):
            return torch.as_tensor(data["state.vector"]).float()

        @staticmethod
        def unapply(data):
            return {"action.vector": torch.as_tensor(data["action"])[..., :14]}

    image = np.zeros((4, 5, 3), dtype=np.uint8)
    value = {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image},
            "right_camera": {"rgb": image},
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }
    adapter = RoboTwin2Adapter(checkpoint_transform=FakeCheckpointTransform())
    adapter.observation(value)

    predicted = torch.arange(24 * 32, dtype=torch.float32).reshape(1, 24, 32)
    action = adapter.action(WorldActionOutput(action=predicted))

    np.testing.assert_allclose(action, predicted[0, :, :14].numpy())


def test_optional_simulator_factory_is_loaded_only_when_requested(monkeypatch):
    fake_module = ModuleType("_fake_libero_for_worldscape_test")
    calls = []

    def make_env(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeEnvironment({})

    fake_module.build = make_env
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    environment = make_libero_environment(
        "task",
        module_name=fake_module.__name__,
        factory_name="build",
        camera_height=128,
    )

    assert isinstance(environment, FakeEnvironment)
    assert calls == [(("task",), {"camera_height": 128})]


def test_missing_optional_dependency_has_actionable_error():
    with pytest.raises(OptionalSimulatorDependencyError, match="LIBERO.*optional module"):
        make_libero_environment(module_name="_worldscape_missing_libero_dependency")


def test_libero_suite_wrapper_constructs_each_trial_from_task_metadata(monkeypatch):
    fake_module = ModuleType("_fake_libero_suite")
    calls = []

    def make_env(**kwargs):
        calls.append(kwargs)
        return FakeEnvironment(
            {
                "agentview_image": np.zeros((4, 5, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((4, 5, 3), dtype=np.uint8),
                "robot0_joint_pos": np.zeros(7, dtype=np.float32),
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            }
        )

    fake_module.make_env = make_env
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    environment = LiberoTaskSuiteEnvironment(
        module_name=fake_module.__name__,
        factory_name="make_env",
        environment_kwargs={"headless": True},
    )
    task = EvaluationTask(
        "suite-3",
        "open the drawer",
        metadata={"bddl_file_name": "task.bddl"},
    )

    environment.reset(task, seed=17)
    environment.close()

    assert calls == [{"headless": True, "bddl_file_name": "task.bddl"}]
