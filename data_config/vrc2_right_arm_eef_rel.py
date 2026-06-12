from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat

vrc2_right_arm_eef_rel_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_head", "cam_left", "cam_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_eef",
            "right_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list([i * 2 for i in range(16)]),
        modality_keys=[
            "right_eef",
            "right_gripper",
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
}

register_modality_config(vrc2_right_arm_eef_rel_config)