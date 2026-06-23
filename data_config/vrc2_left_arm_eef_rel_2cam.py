from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat

vrc2_left_arm_eef_rel_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_head", "cam_left"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_eef",
            "left_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list([i * 2 for i in range(16)]),
        modality_keys=[
            "left_eef",
            "left_gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
    "reward": ModalityConfig(
        delta_indices=[0],
        modality_keys=["reward", "reward.current_frame_idx", "reward.episode_lengths"],
    ),
}

register_modality_config(vrc2_left_arm_eef_rel_config)