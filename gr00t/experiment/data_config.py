# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
    StateActionPerturbationVRH31,
    StateActionDropout,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCropCenter,
    VideoCropTopLeft,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
    VideoRandomlyRandomAffine,
    VideoRandomPerspective,
    VideoRandomPosterize,
    VideoRandomlyGaussianNoise,
)
from gr00t.model.transforms import GR00TTransform


@dataclass
class BaseDataConfig(ABC):
    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


#####################################################################################
# helper functions
#####################################################################################


def import_external_data_config(data_config_str: str) -> Optional[BaseDataConfig]:
    """
    Import and instantiate an external data configuration class.

    Format: "module_path:ClassName" (e.g., "my_configs:RobotConfig")
    Supports nested modules like "package.submodule:ClassName"
    """
    if ":" not in data_config_str:
        return None

    import importlib
    import os
    import sys
    from pathlib import Path

    # Add current working directory to Python path
    current_dir = str(Path(os.getcwd()).absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        module_path, class_name = data_config_str.split(":", 1)
        if not module_path or not class_name:
            raise ValueError(f"Invalid format: '{data_config_str}'. Use 'module:ClassName'")

        print(f"Loading external config: {module_path}.{class_name}")

        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            available = [
                n
                for n in dir(module)
                if not n.startswith("_") and isinstance(getattr(module, n), type)
            ]
            raise AttributeError(
                f"Class '{class_name}' not found in '{module_path}'. Available: {available}"
            )

        # assert if the class has 'transform' and 'modality_config' methods
        if not hasattr(getattr(module, class_name), "transform"):
            raise AttributeError(f"Class '{class_name}' does not have a 'transform' method")
        if not hasattr(getattr(module, class_name), "modality_config"):
            raise AttributeError(f"Class '{class_name}' does not have a 'modality_config' method")

        return getattr(module, class_name)()

    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        print(f"Config loading failed: {e}")
        print("Example: my_configs:MyConfig, package.submodule:ClassName")
        raise


def load_data_config(data_config_str: str) -> BaseDataConfig:
    """
    Get a data config class from a string.
    >>> load_data_config("so100")
    >>> get_data_config("dir.subdir.my_configs:RobotConfig")
    """
    if data_config_str in DATA_CONFIG_MAP:
        return DATA_CONFIG_MAP[data_config_str]
    data_config_cls = import_external_data_config(data_config_str)
    if data_config_cls is not None:
        return data_config_cls
    # Yellow warning color
    yellow = "\033[93m"
    reset = "\033[0m"
    raise ValueError(
        f"{yellow}Invalid data_config '{data_config_str}'. "
        f"Available options: {list(DATA_CONFIG_MAP.keys())}, "
        f"or use 'module:ClassName' for external configs{reset}"
    )


###########################################################################################


class FourierGr1ArmsOnlyDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class So100DataConfig(BaseDataConfig):
    video_keys = ["video.webcam"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class So100DualCamDataConfig(So100DataConfig):
    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))


###########################################################################################


class UnitreeG1DataConfig(BaseDataConfig):
    video_keys = ["video.rs_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class UnitreeG1FullBodyDataConfig(UnitreeG1DataConfig):
    video_keys = ["video.rs_view"]
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))


###########################################################################################


class FourierGr1FullUpperBodyDataConfig(BaseDataConfig):
    video_keys = ["video.front_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
        "state.neck",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
        "action.neck",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaGripperDataConfig(BaseDataConfig):
    video_keys = [
        "video.right_wrist_view",
        "video.left_wrist_view",
        "video.front_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_gripper_qpos",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_gripper_qpos",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_gripper_close",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.right_arm_eef_pos": "min_max",
        "state.right_gripper_qpos": "min_max",
        "state.left_arm_eef_pos": "min_max",
        "state.left_gripper_qpos": "min_max",
    }
    state_target_rotations = {
        "state.right_arm_eef_quat": "rotation_6d",
        "state.left_arm_eef_quat": "rotation_6d",
    }
    action_normalization_modes = {
        "action.right_gripper_close": "binary",
        "action.left_gripper_close": "binary",
    }

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaHandDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.right_wrist_view",
        "video.left_wrist_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_hand",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_hand",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_hand",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.right_arm_eef_pos": "min_max",
        "state.right_hand": "min_max",
        "state.left_arm_eef_pos": "min_max",
        "state.left_hand": "min_max",
    }
    action_normalization_modes = {
        "action.right_hand": "min_max",
        "action.left_hand": "min_max",
    }
    state_target_rotations = {
        "state.right_arm_eef_quat": "rotation_6d",
        "state.left_arm_eef_quat": "rotation_6d",
    }


###########################################################################################


class SinglePandaGripperDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.left_view",
        "video.right_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "min_max",
        "action.end_effector_rotation": "min_max",
        "action.gripper_close": "binary",
        "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }


###########################################################################################


class FourierGr1ArmsWaistDataConfig(FourierGr1ArmsOnlyDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        return super().transform()


###########################################################################################


class OxeDroidDataConfig(BaseDataConfig):
    video_keys = [
        "video.exterior_image_1",
        "video.exterior_image_2",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": "min_max",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_position": "binary",
                },
                target_rotations={"action.eef_rotation_delta": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class AgibotGenie1DataConfig(BaseDataConfig):
    video_keys = [
        "video.top_head",
        "video.hand_left",
        "video.hand_right",
    ]
    state_keys = [
        "state.left_arm_joint_position",
        "state.right_arm_joint_position",
        "state.left_effector_position",
        "state.right_effector_position",
        "state.head_position",
        "state.waist_position",
    ]
    action_keys = [
        "action.left_arm_joint_position",
        "action.right_arm_joint_position",
        "action.left_effector_position",
        "action.right_effector_position",
        "action.head_position",
        "action.waist_position",
        "action.robot_velocity",
    ]
    language_keys = ["annotation.language.action_text"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


############################################################################################

class G1VRLeftRightHandOnlyConfig(BaseDataConfig):
    video_keys = ["video.cam_top"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
        }
        return modality_configs

    def transform(self):
        transforms=[
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95, backend="torchvision"),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear", backend="torchvision" ),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
            VideoToNumpy(apply_to=self.video_keys),

            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={
                "state.left_arm": "min_max",
                "state.left_hand": "min_max",
                "state.right_arm": "min_max",
                "state.right_hand": "min_max",
            }),

            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={
                "action.left_arm": "min_max",
                "action.left_hand": "min_max",
                "action.right_arm": "min_max",
                "action.right_hand": "min_max",
            }),

            # ConcatTransform
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################

class AGITwotHandConfig(BaseDataConfig):
    # video_keys = ["video.cam_left", "video.cam_right"]
    # video_keys = ["video.cam_front", "video.cam_overview"]
    video_keys = ["video.cam_front"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality
        }
        return modality_configs

    def transform(self):
        transforms=[
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95, backend="torchvision"),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear", backend="torchvision" ),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
            VideoToNumpy(apply_to=self.video_keys),

            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={
                "state.left_arm": "min_max",
                "state.left_hand": "min_max",
                "state.right_arm": "min_max",
                "state.right_hand": "min_max",
            }),

            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={
                "action.left_arm": "min_max",
                "action.left_hand": "min_max",
                "action.right_arm": "min_max",
                "action.right_hand": "min_max",
            }),

            # ConcatTransform
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


############################################################################################

class AGITwotHand3CamConfig(BaseDataConfig):
    # video_keys = ["video.cam_left", "video.cam_right"]
    video_keys = ["video.cam_front", "video.cam_right", "video.cam_left"]
    # video_keys = ["video.cam_front"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality
        }
        return modality_configs

    def transform(self):
        transforms=[
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95, backend="torchvision"),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear", backend="torchvision" ),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
            VideoToNumpy(apply_to=self.video_keys),

            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={
                "state.left_arm": "min_max",
                "state.left_hand": "min_max",
                "state.right_arm": "min_max",
                "state.right_hand": "min_max",
            }),

            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={
                "action.left_arm": "min_max",
                "action.left_hand": "min_max",
                "action.right_arm": "min_max",
                "action.right_hand": "min_max",
            }),

            # ConcatTransform
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    

############################################################################################

class LiberoConfig(BaseDataConfig):
    video_keys = ["video.image", "video.wrist_image"]
    state_keys = [
        "state.state",
    ]
    action_keys = [
        "action.actions",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH2TwotHand2CamConfig(BaseDataConfig):
    video_keys = ["video.cam_front", "video.cam_waist"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH2TwotHand2CamVelEffConfig(BaseDataConfig):
    video_keys = ["video.cam_front", "video.cam_waist"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.velocity_left_arm",
        "state.velocity_right_arm",
        "state.velocity_left_hand",
        "state.velocity_right_hand",
        "state.effort_left_arm",
        "state.effort_right_arm",
        "state.effort_left_hand",
        "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH3TwotHandConfig(BaseDataConfig):
    video_keys = ["video.cam_head"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.velocity_left_arm",
        "state.velocity_right_arm",
        "state.velocity_left_hand",
        "state.velocity_right_hand",
        "state.effort_left_arm",
        "state.effort_right_arm",
        "state.effort_left_hand",
        "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################

class VRH3FullBodyConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.effort_left_arm",
        "state.effort_right_arm",
        "state.effort_left_hand",
        "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
###########################################################################################
class VRH3TwotHandTaskCompletionConfig(BaseDataConfig):
    """
    VRH3 Two Hand config with task completion prediction enabled.
    This config loads observation.tasks.done from the dataset and trains the model
    to predict task completion alongside action prediction.
    
    Note: This config is designed for datasets with task completion labels.
    The state keys should match your dataset's modality.json configuration.
    
    Task completion key format:
        task_completion_keys = ["<parquet_column_name>"]
        e.g., ["observation.tasks.done"] reads directly from the 
        "observation.tasks.done" column in the parquet file.
    """
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    # Task completion key - directly the parquet column name
    task_completion_keys = ["observation.tasks.done"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        
        # Add task completion modality if enabled
        if self.task_completion_keys:
            task_completion_modality = ModalityConfig(
                delta_indices=[0],  # Same indices as action to get future completion status
                modality_keys=self.task_completion_keys,
            )
            modality_configs["task_completion"] = task_completion_modality
        
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # task completion transforms (binary normalization)
            StateActionToTensor(apply_to=self.task_completion_keys),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                use_task_completion=self.use_task_completion,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class AlohaRightArmConfig(BaseDataConfig):
    video_keys = ["video.cam_high", "video.cam_wrist_right"]
    state_keys = [
        "state.right_arm",
        # "state.velocity_right_arm",
        "state.effort_right_arm",
    ]
    action_keys = [
        "action.right_arm",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################

class AlohaRightArm30Config(BaseDataConfig):
    video_keys = ["video.cam_high", "video.cam_wrist_right"]
    state_keys = [
        "state.right_arm",
        # "state.velocity_right_arm",
        "state.effort_right_arm",
    ]
    action_keys = [
        "action.right_arm",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(30))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH3TwotHandConfig_1H_to_4TDBU_5V(BaseDataConfig):
    video_keys = ["video.cam_virtual_head", "video.cam_virtual_leftTD", "video.cam_virtual_rightTD", "video.cam_virtual_leftBU", "video.cam_virtual_rightBU"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.velocity_left_arm",
        "state.velocity_right_arm",
        "state.velocity_left_hand",
        "state.velocity_right_hand",
        "state.effort_left_arm",
        "state.effort_right_arm",
        "state.effort_left_hand",
        "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH3TwotHandConfig_3HLR_to_4TDBU_7V(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right", "video.cam_virtual_leftBU", "video.cam_virtual_rightBU"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.velocity_left_arm",
        "state.velocity_right_arm",
        "state.velocity_left_hand",
        "state.velocity_right_hand",
        "state.effort_left_arm",
        "state.effort_right_arm",
        "state.effort_left_hand",
        "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH3TwotHandExtraVirtualConfig(BaseDataConfig):
    video_keys = [
        "video.cam_head", 
        "video.cam_left", 
        "video.cam_right", 
        "video.cam_virtual_leftBU",
        "video.cam_virtual_rightBU"
    ]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.velocity_left_arm",
        "state.velocity_right_arm",
        "state.velocity_left_hand",
        "state.velocity_right_hand",
        "state.effort_left_arm",
        "state.effort_right_arm",
        "state.effort_left_hand",
        "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH3TwotHandExtraOnlyVirtualConfig(BaseDataConfig):
    video_keys = [
        "video.cam_virtual_head", 
        "video.cam_virtual_left", 
        "video.cam_virtual_right", 
        "video.cam_virtual_leftBU",
        "video.cam_virtual_rightBU"
    ]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.velocity_left_arm",
        "state.velocity_right_arm",
        "state.velocity_left_hand",
        "state.velocity_right_hand",
        "state.effort_left_arm",
        "state.effort_right_arm",
        "state.effort_left_hand",
        "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH3TwotHandDoubleExtraVirtualConfig(BaseDataConfig):
    video_keys = [
        "video.cam_head", 
        "video.cam_left", 
        "video.cam_right", 
        "video.cam_virtual_leftBU",
        "video.cam_virtual_rightBU",
        "video.cam_virtual_leftTD",
        "video.cam_virtual_rightTD"
    ]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.velocity_left_arm",
        "state.velocity_right_arm",
        "state.velocity_left_hand",
        "state.velocity_right_hand",
        "state.effort_left_arm",
        "state.effort_right_arm",
        "state.effort_left_hand",
        "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH3TwotHandDoubleExtraOnlyVirtualConfig(BaseDataConfig):
    video_keys = [
        "video.cam_virtual_head", 
        "video.cam_virtual_left", 
        "video.cam_virtual_right", 
        "video.cam_virtual_leftBU",
        "video.cam_virtual_rightBU",
        "video.cam_virtual_leftTD",
        "video.cam_virtual_rightTD",

    ]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.velocity_left_arm",
        "state.velocity_right_arm",
        "state.velocity_left_hand",
        "state.velocity_right_hand",
        "state.effort_left_arm",
        "state.effort_right_arm",
        "state.effort_left_hand",
        "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################
    
class VRH31TwoArmsExcludeEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.velocity_left_arm",
        "state.velocity_right_arm",
        "state.velocity_left_hand",
        "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH31TwoArmsQuant2ExcludeEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.velocity_left_arm",
        "state.velocity_right_arm",
        "state.velocity_left_hand",
        "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH31TwoArmsQuant3ExcludeEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.velocity_left_arm",
        "state.velocity_right_arm",
        "state.velocity_left_hand",
        "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 48, 3))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################

class VRH31TwoArmsIncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        # "state.velocity_left_arm",
        # "state.velocity_right_arm",
        # "state.velocity_left_hand",
        # "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH31TwoArmsQuant2IncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        # "state.velocity_left_arm",
        # "state.velocity_right_arm",
        # "state.velocity_left_hand",
        # "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################

class VRH31TwoArmsHor50IncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        # "state.velocity_left_arm",
        # "state.velocity_right_arm",
        # "state.velocity_left_hand",
        # "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(51))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    

###########################################################################################

class VRH31TwoArmsHor50Quant2IncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        # "state.velocity_left_arm",
        # "state.velocity_right_arm",
        # "state.velocity_left_hand",
        # "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 102, 2))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH31TwoArmsHor50CropIncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        # "state.velocity_left_arm",
        # "state.velocity_right_arm",
        # "state.velocity_left_hand",
        # "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(51))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCropCenter(apply_to=["video.cam_head"], height=int(336 * 1.05), width=int(224 * 1.05), scale=0.95),
            VideoCropTopLeft(apply_to=["video.cam_left"], height=int(336 * 1.05), width=int(224 * 1.05), scale=0.95),
            VideoCrop(apply_to=["video.cam_right"], scale=0.95),
            VideoResize(apply_to=["video.cam_head", "video.cam_left"], height=224, width=224, interpolation="linear"),
            VideoResize(apply_to=["video.cam_right"], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH31TwoArmsHor50Quant2CropIncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        # "state.velocity_left_arm",
        # "state.velocity_right_arm",
        # "state.velocity_left_hand",
        # "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 102, 2))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCropCenter(apply_to=["video.cam_head"], height=int(336 * 1.05), width=int(224 * 1.05), scale=0.95),
            VideoCropTopLeft(apply_to=["video.cam_left"], height=int(336 * 1.05), width=int(224 * 1.05), scale=0.95),
            VideoCrop(apply_to=["video.cam_right"], scale=0.95),
            VideoResize(apply_to=["video.cam_head", "video.cam_left"], height=224, width=224, interpolation="linear"),
            VideoResize(apply_to=["video.cam_right"], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################

class VRH31LeftArmIncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        # "state.right_arm",
        "state.left_hand",
        # "state.right_hand",
        # "state.velocity_left_arm",
        # "state.velocity_right_arm",
        # "state.velocity_left_hand",
        # "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        # "action.right_arm",
        "action.left_hand",
        # "action.right_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################

class VRH31LeftArmExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        # "state.right_arm",
        "state.left_hand",
        # "state.right_hand",
        # "state.velocity_left_arm",
        # "state.velocity_right_arm",
        # "state.velocity_left_hand",
        # "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        "action.left_arm",
        # "action.right_arm",
        "action.left_hand",
        # "action.right_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################

class VRH311LeftArmStereoQuant2IncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_head_right"]
    state_keys = [
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.left_hand"
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################
    
class VRH311GripperOpenQuant2IncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.gripper_open"
    ]
    action_keys = [
        "action.left_arm",
        "action.gripper_open",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################

class VRH311GripperOpenStereoQuant2IncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_head_right"]
    state_keys = [
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.gripper_open"
    ]
    action_keys = [
        "action.left_arm",
        "action.gripper_open",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################

class VRH311LeftArmRandQuant2IncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.left_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoRandomlyRandomAffine(
                apply_to=self.video_keys[:1],
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.85, 1.05),
                shear=[-10, 10, -10, 10],
                height=600,
                width=960,
                p=0.5
            ),
            VideoRandomPerspective(
                apply_to=self.video_keys[:1],
                distortion_scale=0.15,
                p=0.5
            ),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),

            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoRandomlyRandomAffine(
                apply_to=self.video_keys[1:],
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.85, 1.05),
                shear=[-10, 10, -10, 10],
                height=480,
                width=640,
                p=0.5
            ),
            VideoRandomPerspective(
                apply_to=self.video_keys[1:],
                distortion_scale=0.15,
                p=0.5
            ),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),

            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoRandomPosterize(apply_to=self.video_keys, bits=4, p=0.3),
            VideoRandomlyGaussianNoise(apply_to=self.video_keys, mean=0.0, sigma=0.05, p=0.7),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionPerturbationVRH31(
                apply_to=self.state_keys,
                perturbation_prob=0.7,
                min_noises={
                    "state.left_shoulder": [-0.075, -0.075, -0.075],
                    "state.left_elbow": [-0.075],
                    "state.left_wrist": [-0.075, -0.075, -0.075],
                    "state.left_hand": [-0.075, -0.075, -0.075, -0.075, -0.075, -0.075],
                },
                max_noises={
                    "state.left_shoulder": [0.075, 0.075, 0.075],
                    "state.left_elbow": [0.075],
                    "state.left_wrist": [0.075, 0.075, 0.075],
                    "state.left_hand": [0.075, 0.075, 0.075, 0.075, 0.075, 0.075],
                },
                min_state_limits={
                    "state.left_shoulder": [-3.14, 0.0, -2.967],
                    "state.left_elbow": [-0.524],
                    "state.left_wrist": [-2.967, -1.134, -0.96],
                    "state.left_hand": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                max_state_limits={
                    "state.left_shoulder": [1.047, 2.356, 2.967],
                    "state.left_elbow": [1.571],
                    "state.left_wrist": [2.967, 0.698, 0.96],
                    "state.left_hand": [1.658, 0.62, 1.4381, 1.4381, 1.4381, 1.4381],
                },
            ),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH311LeftArmRandDropoutQuant2IncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.left_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoRandomlyRandomAffine(
                apply_to=self.video_keys[:1],
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.85, 1.05),
                shear=[-10, 10, -10, 10],
                height=600,
                width=960,
                p=0.5
            ),
            VideoRandomPerspective(
                apply_to=self.video_keys[:1],
                distortion_scale=0.15,
                p=0.5
            ),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),

            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoRandomlyRandomAffine(
                apply_to=self.video_keys[1:],
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.85, 1.05),
                shear=[-10, 10, -10, 10],
                height=480,
                width=640,
                p=0.5
            ),
            VideoRandomPerspective(
                apply_to=self.video_keys[1:],
                distortion_scale=0.15,
                p=0.5
            ),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),

            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoRandomPosterize(apply_to=self.video_keys, bits=4, p=0.3),
            VideoRandomlyGaussianNoise(apply_to=self.video_keys, mean=0.0, sigma=0.05, p=0.7),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionPerturbationVRH31(
                apply_to=self.state_keys,
                perturbation_prob=0.7,
                min_noises={
                    "state.left_shoulder": [-0.05, -0.05, -0.05],
                    "state.left_elbow": [-0.05],
                    "state.left_wrist": [-0.05, -0.05, -0.05],
                    "state.left_hand": [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05],
                },
                max_noises={
                    "state.left_shoulder": [0.05, 0.05, 0.05],
                    "state.left_elbow": [0.05],
                    "state.left_wrist": [0.05, 0.05, 0.05],
                    "state.left_hand": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                },
                min_state_limits={
                    "state.left_shoulder": [-3.14, 0.0, -2.967],
                    "state.left_elbow": [-0.524],
                    "state.left_wrist": [-2.967, -1.134, -0.96],
                    "state.left_hand": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                max_state_limits={
                    "state.left_shoulder": [1.047, 2.356, 2.967],
                    "state.left_elbow": [1.571],
                    "state.left_wrist": [2.967, 0.698, 0.96],
                    "state.left_hand": [1.658, 0.62, 1.4381, 1.4381, 1.4381, 1.4381],
                },
            ),
            StateActionDropout(apply_to=self.state_keys, dropout_prob=0.3),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH311LeftArmQuant2IncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.left_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH311LeftArmQuant2ExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.left_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand"
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH311LeftArmQuant2IncludeTaskProgressConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.left_hand",
        "state.velocity_left_shoulder",
        "state.velocity_left_elbow",
        "state.velocity_left_wrist",
        "state.velocity_left_hand",
        "state.effort_left_shoulder",
        "state.effort_left_elbow",
        "state.effort_left_wrist",
        "state.effort_left_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH31TwoHand3CamConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    task_completion_keys = ["observation.tasks.done"]
    
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=[-20, -15, -10, -5, 0],
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
                # Add task completion modality if enabled
        if self.task_completion_keys:
            task_completion_modality = ModalityConfig(
                delta_indices=[0],  # Same indices as action to get future completion status
                modality_keys=self.task_completion_keys,
            )
            modality_configs["task_completion"] = task_completion_modality
        
        return modality_configs
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            StateActionToTensor(apply_to=self.task_completion_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                use_task_completion=True,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################

class VRH31GripperAction(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        # "state.left_arm",
        # "state.right_arm",
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.left_hand",
        # "state.velocity_left_arm",
        # "state.velocity_right_arm",
        # "state.velocity_left_hand",
        # "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        # "action.left_arm",
        # "action.right_arm",
        "action.left_arm",
        "action.gripper_open",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = [x * 2 for x in range(16)]

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
 
###########################################################################################

class VRH31GripperBoth(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        # "state.left_arm",
        # "state.right_arm",
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.gripper_open",
        # "state.velocity_left_arm",
        # "state.velocity_right_arm",
        # "state.velocity_left_hand",
        # "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        # "action.left_arm",
        # "action.right_arm",
        "action.left_arm",
        "action.gripper_open",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = [x * 2 for x in range(16)]

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    


###########################################################################################

class VRH31LeftArm2Cam(BaseDataConfig):
    video_keys = ["video.cam_front", "video.cam_left"]
    state_keys = [
        "state.left_arm",
        "state.left_hand"
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.task_progress"
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = action_indices = [x * 2 for x in range(16)]

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class VRH31OneHand3CamConfig(BaseDataConfig):
    video_keys = ["video.cam_front", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.arm",
        "state.left_hand",
    ]
    action_keys = [
        "action.arm",
        "action.left_hand",
        "action.task_progress"
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    
###########################################################################################

class VRH311LeftArmRVTQuant2IncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_virtual_view3", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.left_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:2]),
            VideoCrop(apply_to=self.video_keys[:2], scale=0.95),
            VideoResize(apply_to=self.video_keys[:2], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[2:]),
            VideoCrop(apply_to=self.video_keys[2:], scale=0.95),
            VideoResize(apply_to=self.video_keys[2:], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

DATA_CONFIG_MAP = {
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "fourier_gr1_arms_only": FourierGr1ArmsOnlyDataConfig(),
    "fourier_gr1_full_upper_body": FourierGr1FullUpperBodyDataConfig(),
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig(),
    "bimanual_panda_hand": BimanualPandaHandDataConfig(),
    "single_panda_gripper": SinglePandaGripperDataConfig(),
    "so100": So100DataConfig(),
    "so100_dualcam": So100DualCamDataConfig(),
    "unitree_g1": UnitreeG1DataConfig(),
    "unitree_g1_full_body": UnitreeG1FullBodyDataConfig(),
    "oxe_droid": OxeDroidDataConfig(),
    "agibot_genie1": AgibotGenie1DataConfig(),
    "vr_left_right_g1": G1VRLeftRightHandOnlyConfig(),
    "agi_two_hand": AGITwotHandConfig(),
    "agi_two_hand_3_cam": AGITwotHand3CamConfig(),
    "libero": LiberoConfig(),
    "vrh2_two_hand_2_cam": VRH2TwotHand2CamConfig(),
    "vrh2_two_hand_2_cam_vel_eff": VRH2TwotHand2CamVelEffConfig(), 
    "vrh3_two_hand": VRH3TwotHandConfig(),
    "vrh3_two_hand_1h_to_4tdbu_5v": VRH3TwotHandConfig_1H_to_4TDBU_5V(),
    "vrh3_two_hand_3hlr_to_4tdbu_7v": VRH3TwotHandConfig_3HLR_to_4TDBU_7V(),
    "vrh3_two_hand_extra_virtual": VRH3TwotHandExtraVirtualConfig(),
    "vrh3_two_hand_extra_only_virtual": VRH3TwotHandExtraOnlyVirtualConfig(),
    "vrh3_two_hand_double_extra_virtual": VRH3TwotHandDoubleExtraVirtualConfig(),
    "vrh3_two_hand_double_extra_only_virtual": VRH3TwotHandDoubleExtraOnlyVirtualConfig(),
    "vr_h31_two_arms_exclude_effort": VRH31TwoArmsExcludeEffortConfig(),
    "vr_h31_two_arms_quant2_exclude_effort": VRH31TwoArmsQuant2ExcludeEffortConfig(),
    "vr_h31_two_arms_quant3_exclude_effort": VRH31TwoArmsQuant3ExcludeEffortConfig(),
    "vr_h31_two_arms_include_task_progress_exclude_velocity_effort": VRH31TwoArmsIncludeTaskProgressExcludeVelocityEffortConfig(),
    "vr_h31_two_arms_quant2_include_task_progress_exclude_velocity_effort": VRH31TwoArmsQuant2IncludeTaskProgressExcludeVelocityEffortConfig(),
    "vr_h31_two_arms_hor50_include_task_progress_exclude_velocity_effort": VRH31TwoArmsHor50IncludeTaskProgressExcludeVelocityEffortConfig(),
    "vr_h31_two_arms_hor50_quant2_include_task_progress_exclude_velocity_effort": VRH31TwoArmsHor50Quant2IncludeTaskProgressExcludeVelocityEffortConfig(),
    "vr_h31_two_arms_hor50_crop_include_task_progress_exclude_velocity_effort": VRH31TwoArmsHor50CropIncludeTaskProgressExcludeVelocityEffortConfig(),
    "vr_h31_two_arms_hor50_quant2_crop_include_task_progress_exclude_velocity_effort": VRH31TwoArmsHor50Quant2CropIncludeTaskProgressExcludeVelocityEffortConfig(),
    "vr_h31_left_arm_include_task_progress_exclude_velocity_effort": VRH31LeftArmIncludeTaskProgressExcludeVelocityEffortConfig(),

    "vr_h311_left_arm_stereo_quant2_include_task_progress_exclude_velocity_effort": VRH311LeftArmStereoQuant2IncludeTaskProgressExcludeVelocityEffortConfig(),
    "vr_h311_gripper_open_quant2_include_task_progress_exclude_velocity_effort": VRH311GripperOpenQuant2IncludeTaskProgressExcludeVelocityEffortConfig(),
    "vr_h311_gripper_open_stereo_quant2_include_task_progress_exclude_velocity_effort": VRH311GripperOpenStereoQuant2IncludeTaskProgressExcludeVelocityEffortConfig(),

    "vr_h311_left_arm_quant2_include_task_progress_exclude_velocity_effort": VRH311LeftArmQuant2IncludeTaskProgressExcludeVelocityEffortConfig(),
    "vr_h311_left_arm_quant2_include_task_progress": VRH311LeftArmQuant2IncludeTaskProgressConfig(),
    "vr_h311_left_arm_quant2_exclude_velocity_effort": VRH311LeftArmQuant2ExcludeVelocityEffortConfig(),
    "vr_h311_left_arm_rand_quant2_include_task_progress_exclude_velocity_effort": VRH311LeftArmRandQuant2IncludeTaskProgressExcludeVelocityEffortConfig(),
    "vr_h311_left_arm_rand_dropout_quant2_include_task_progress_exclude_velocity_effort": VRH311LeftArmRandDropoutQuant2IncludeTaskProgressExcludeVelocityEffortConfig(),
    "vr_h311_left_arm_rvt_quant2_include_task_progress_exclude_velocity_effort": VRH311LeftArmRVTQuant2IncludeTaskProgressExcludeVelocityEffortConfig(),

    "vr_h31_two_hand_3cam": VRH31TwoHand3CamConfig(),

    "vr_h31_left_arm_gripper_action": VRH31GripperAction(),
    "vr_h31_left_arm_gripper_both": VRH31GripperBoth(), 
    "vr_h31_left_arm_2cam": VRH31LeftArm2Cam(),
    "vr_h31_one_hand_3cam": VRH31OneHand3CamConfig(),
    
    "vr_h31_left_arm_exclude_velocity_effort": VRH31LeftArmExcludeVelocityEffortConfig(),
    "vrh3_two_hand_task_completion": VRH3TwotHandTaskCompletionConfig(),
    "vrh3_effort": VRH3FullBodyConfig(),
    "aloha_right_arm_only": AlohaRightArmConfig(),
    "aloha_right_arm_30_only": AlohaRightArm30Config()
}
