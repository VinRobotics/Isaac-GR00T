"""
Test script to verify EquivariantFeedForward implementation
"""
import einops
import escnn
import torch
import escnn.nn as enn
from escnn import gspaces
from gr00t.model.action_head.flow_matching_action_head import RotationTransformer
import torch

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

            rot_6d = torch.cat([ee_cos1,ee_cos2, ee_cos3, ee_sin1, ee_sin2, ee_sin3], dim=-1)
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

from gr00t.model.action_head.flow_matching_action_head import MultiEmbodimentActionEncoder

# Setup: Create a simple SO(2) group space
from escnn.group import CyclicGroup

G = CyclicGroup(8)
gspace = escnn.gspaces.no_base_space(G)

# Define field types
input_embedding_dim = 32
action_type = getJointFieldType(gspace, is_action=True)
action_hidden_type = enn.FieldType(gspace, int(2 * input_embedding_dim / 8) * [gspace.regular_repr])
action_out_type = enn.FieldType(gspace, int(input_embedding_dim / 8) * [gspace.regular_repr])

action_encoder = MultiEmbodimentActionEncoder(
    in_type=action_type,
    hidden_type=action_hidden_type,
    out_type=action_out_type,
    num_embodiments=2
)
    

# Create test input: (batch=2, sequence=10, channels=input_embedding_dim)
B, T = 2, 10
x = torch.randn(B, T, action_type.size)
timestep = torch.tensor([5, 10])  # Two different timesteps for batch
x = einops.rearrange(x, 'b t c -> (b t) c')

noisy_trajectory = enn.GeometricTensor(x, action_type)

action_encoder_embodiment_id = torch.tensor([1, 1], dtype=torch.int64).repeat((10))
action_features = action_encoder(noisy_trajectory, timestep, action_encoder_embodiment_id)
origin_action_features = action_features

# print(origin_action_features)
for element in gspace.testing_elements:
    # Rotate input
    rotated_noisy_trajectory = noisy_trajectory.transform(element)

    # Forward pass with rotated input
    rotated_output = action_encoder(rotated_noisy_trajectory, timestep, action_encoder_embodiment_id).tensor

    # rotate original output
    rotated_origin_output = origin_action_features.transform(element).tensor
        
    err = (rotated_origin_output - rotated_output).abs().mean()
    print(torch.allclose(rotated_output, rotated_origin_output, atol=1e-4), element, err)

