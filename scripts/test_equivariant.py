from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.model.action_head.flow_matching_action_head import RotationTransformer
import numpy as np

from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup 
import einops
import torch
from copy import deepcopy


data_config = load_data_config("vrh3_two_hand_1_cam_equi")
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

dataset = LeRobotSingleDataset(
    "/home/locht1/Documents/locht1/code_convert/output/20251126_VR_H3_pickpart_equi_speedup1",
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
    modality_configs=modality_config,
    transforms=modality_transform
)

client = RobotInferenceClient(
    
)
state = dataset[0]["state"]

obs = {
    "video.cam_front": np.ones((1, 480, 640, 3), dtype=np.uint8),
    "state.left_hand": state[0, 14:20][None, :],
    "state.right_hand": state[0, 20:26][None, :],
    "state.l_ee_pos": state[0, :3][None, :],
    "state.l_ee_quat": state[0, 3:7][None, :],
    "state.r_ee_pos": state[0, 7:10][None, :],
    "state.r_ee_quat": state[0, 10:14][None, :],
    "annotation.human.task_description": ["Pick up the metal part and hold it using two hands."],
}

def getJointFieldType(group, is_action):
    max_dim = 26 if is_action else 64
    return enn.FieldType(
        group,
        2 * 4 * [group.irrep(1)] # pos xy, rot 6, left and right
        + (max_dim - 14 + 2) * [group.trivial_repr], # gripper 1, z from both ee is 2
    )


def get6DRotation(quat):
    # data is in xyzw, but rotation transformer takes wxyz
    quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')
    return quaternion_to_sixd.forward(quat[:, :, [3, 0, 1, 2]]) 

    
def getQuaternionFrom6D(rot_6d):
    quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')
    
    quat = quaternion_to_sixd.inverse(rot_6d)
    return quat[:, :, [1, 2, 3, 0]]  # xyzw

def getJointGeometricTensor(group, state, is_action):
    def getJointGeometricTensorEachHand(ee_state):
        ee_pos = ee_state[:, :, :3] # (bs, t, 3)
        ee_quat = ee_state[:, :, 3:7] # (bs, t, 4)
        ee_rot = get6DRotation(ee_quat)
        pos_xy = ee_pos[:, :, 0:2] # 2
        pos_z = ee_pos[:, :, 2:3] # 1
        joint_features = torch.cat(
            [
                pos_xy,
                ee_rot[:, :, 0:1], # 1
                ee_rot[:, :, 3:4], # 1
                ee_rot[:, :, 1:2], # 1
                ee_rot[:, :, 4:5], # 1
                ee_rot[:, :, 2:3], # 1
                ee_rot[:, :, 5:6], # 1
            ],
            dim=-1
        )
        return joint_features, pos_z
    
    l_ee_state = state[:, :, :7] # bs, t, 7 
    r_ee_state = state[:, :, 7:14] # bs, t, 7  
    hand_state = state[:, :, 14:]
    
    l_tf, l_pos_z = getJointGeometricTensorEachHand(l_ee_state)
    r_tf, r_pos_z = getJointGeometricTensorEachHand(r_ee_state)

    state_features = torch.cat([l_tf, r_tf, l_pos_z, r_pos_z, hand_state], dim=-1)
    state_features = einops.rearrange(state_features, 'b t c -> (b t) c')

    return enn.GeometricTensor(state_features, getJointFieldType(group, is_action))

def getActionOutput(pred):
        def getActionOutputEachHand(ee_pred):
            ee_xy = ee_pred[:, :, 0:2] # (bs, t, 3)
            ee_cos1 = ee_pred[:, :, 2:3]
            ee_sin1 = ee_pred[:, :, 3:4]
            ee_cos2 = ee_pred[:, :, 4:5]
            ee_sin2 = ee_pred[:, :, 5:6]
            ee_cos3 = ee_pred[:, :, 6:7]
            ee_sin3 = ee_pred[:, :, 7:8]

            rot_6d = torch.cat([ee_cos1, ee_cos2, ee_cos3, ee_sin1, ee_sin2, ee_sin3], dim=-1)
            quat = getQuaternionFrom6D(rot_6d)
            return ee_xy, quat

        l_xy, l_quat = getActionOutputEachHand(pred[:, :, :8]) # bs, t, 8
        r_xy, r_quat = getActionOutputEachHand(pred[:, :, 8:16]) # bs, t, 8
        
        l_z, r_z = pred[:, :, 16:17], pred[:, :, 17:18] # bs, t, 1
        hand_state = pred[:, :, 18:] # bs, t, rest
        
        action_output = torch.cat(
            [l_xy, l_z, l_quat, r_xy, r_z, r_quat, hand_state],
            dim=-1
        )
        
        return action_output

n_group = 8
group = gspaces.no_base_space(CyclicGroup(n_group))
state_in_type = getJointFieldType(group, is_action=False)

origin = client.get_action(obs)
print(origin)

action = np.concatenate(
    [
        origin["action.l_ee_pos"],
        origin["action.l_ee_quat"],
        origin["action.r_ee_pos"],
        origin["action.r_ee_quat"],
        origin["action.left_hand"],
        origin["action.right_hand"],
    ],
    axis=-1
)


for element in group.testing_elements:
    rotated_obs = deepcopy(obs)

    # Rotate state
    rotated_state_tensor = getJointGeometricTensor(
        group, torch.tensor(deepcopy(state)).unsqueeze(1), is_action=False
    )
    rotated_state_tensor = rotated_state_tensor.transform(element)
    rotated_state_tensor = einops.rearrange(
        rotated_state_tensor.tensor, '(b t) c -> b t c', b=1
    )
    
    rotated_state = getActionOutput(rotated_state_tensor).squeeze(0)
    rotated_state = rotated_state.numpy()

    print("element:", element)
    print("state", state)
    print("rotated state", rotated_state)
    
    # NOTE: video.cam_front is NOT rotated - this is intentional if 
    # testing state-only equivariance, but the model must handle this correctly
    
    rotated_obs["state.left_hand"] = rotated_state[0, 14:20][None, :]
    rotated_obs["state.right_hand"] = rotated_state[0, 20:26][None, :]
    rotated_obs["state.l_ee_pos"] = rotated_state[0, :3][None, :]
    rotated_obs["state.l_ee_quat"] = rotated_state[0, 3:7][None, :]
    rotated_obs["state.r_ee_pos"] = rotated_state[0, 7:10][None, :]
    rotated_obs["state.r_ee_quat"] = rotated_state[0, 10:14][None, :]
    output = client.get_action(rotated_obs)

    output_action = torch.tensor(np.concatenate(
        [
            output["action.l_ee_pos"],
            output["action.l_ee_quat"],
            output["action.r_ee_pos"],
            output["action.r_ee_quat"],
            output["action.left_hand"],
            output["action.right_hand"],
        ],
        axis=-1
    ))
    rotated_origin_action = getJointGeometricTensor(
        group, torch.tensor(deepcopy(action)).unsqueeze(1), is_action=True
    )
    rotated_origin_action = rotated_origin_action.transform(element)
    rotated_origin_action = einops.rearrange(
        rotated_origin_action.tensor, '(b t) c -> b t c', b=1
    )
    
    rotated_origin_action = getActionOutput(rotated_origin_action).squeeze(0)
    print(rotated_origin_action.shape, output_action.shape)
    err = (rotated_origin_action - output_action).abs().mean()
    print(torch.allclose(output_action, rotated_origin_action, atol=1e-4), element, err)
