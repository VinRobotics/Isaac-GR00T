from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat

vrh3_two_hands_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_head", "cam_left", "cam_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "gripper_open",
        ],
        sin_cos_embedding_keys=[
            "left_shoulder",
            "left_elbow",
            "left_wrist",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list([i * 2 for i in range(16)]),
        modality_keys=[
            "left_arm",
            "gripper_open",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
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

register_modality_config(vrh3_two_hands_config)