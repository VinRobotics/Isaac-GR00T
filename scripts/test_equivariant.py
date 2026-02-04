from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.model.action_head.flow_matching_action_head import RotationTransformer
import numpy as np
import math
import torch.nn.functional as F
from PIL import Image

from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup 
import einops
import torch
from copy import deepcopy


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by angle (in radians) using grid_sample.
    
    Args:
        img: [H, W, C] or [B, H, W, C] numpy array
        angle: rotation angle in radians
        
    Returns:
        Rotated image numpy array
    """
    if img.ndim == 3:
        img = img[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False
    
    B, H, W, C = img.shape
    
    # Convert to torch tensor [B, C, H, W]
    img_tensor = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    
    # Create rotation matrix (negative angle for clockwise rotation)
    cos_val = math.cos(-angle)
    sin_val = math.sin(-angle)
    
    rotation_matrix = torch.tensor([
        [cos_val, -sin_val, 0],
        [sin_val, cos_val, 0]
    ], dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    
    # Generate sampling grid
    grid = F.affine_grid(rotation_matrix, list([B, C, H, W]), align_corners=True)
    
    # Apply rotation
    rotated = F.grid_sample(img_tensor, grid, align_corners=True, padding_mode='zeros')
    
    # Convert back to numpy [B, H, W, C]
    rotated_np = rotated.permute(0, 2, 3, 1).numpy().astype(img.dtype)
    
    if squeeze:
        rotated_np = rotated_np[0]
    
    return rotated_np


def save_image(img: np.ndarray, save_path: str):
    """Save image as PNG."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)
    print(f"Saved image to: {save_path}")


def create_structured_image(height: int = 480, width: int = 640) -> np.ndarray:
    """
    Create a structured (non-symmetric) test image with gradient and diagonal patterns.
    This ensures rotation produces visually different images.
    
    Returns:
        [H, W, C] numpy array with uint8 values (0-255)
    """
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    center_y, center_x = height // 2, width // 2
    
    for i in range(height):
        for j in range(width):
            # Horizontal gradient in red channel
            img[i, j, 0] = j / width
            
            # Vertical gradient in green channel  
            img[i, j, 1] = i / height
            
            # Diagonal pattern in blue channel
            if i > j * (height / width):
                img[i, j, 2] = 0.8
            elif i < (j - width // 4) * (height / width):
                img[i, j, 2] = 0.3
    
    # Add a circle in top-right as orientation marker
    circle_radius = min(height, width) // 8
    cx, cy = width * 3 // 4, height // 4
    for i in range(height):
        for j in range(width):
            dist = math.sqrt((i - cy)**2 + (j - cx)**2)
            if dist < circle_radius:
                img[i, j, 0] = 1.0  # Bright red circle
                img[i, j, 1] = 0.2
                img[i, j, 2] = 0.2
    
    # Add corner marker in top-left
    corner_size = min(height, width) // 6
    for i in range(corner_size):
        for j in range(corner_size):
            img[i, j, 1] = 1.0  # Bright green corner
            img[i, j, 0] = 0.2
            img[i, j, 2] = 0.2
    
    # Convert to uint8
    img = (img * 255).clip(0, 255).astype(np.uint8)
    
    return img


data_config = load_data_config("vrh3_two_hand_1_cam_equi")
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

dataset = LeRobotSingleDataset(
    "/home/locht1/vr_data/20251205_VR_H3_placepart_equi_speedup1",
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
    modality_configs=modality_config,
    transforms=modality_transform
)

client = RobotInferenceClient(
    
)
state = dataset[0]["state"]

# Create structured image for better equivariance testing
structured_image = create_structured_image(height=480, width=640)

obs = {
    "video.cam_front": structured_image[np.newaxis, ...],  # Add batch dimension [1, H, W, C]
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

# Save original image for visualization
original_image = obs["video.cam_front"][0]  # [H, W, C]
save_image(original_image, "test_original_image.png")

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

print("\n" + "=" * 70)
print("Testing Equivariance: f(g*x) = g*f(x)")
print("  - Rotating both IMAGE and STATE")
print("  - Checking if output action transforms correctly")
print("  - NOTE: Vision FA should make features INVARIANT to image rotation")
print("  - NOTE: State/Action should be EQUIVARIANT")
print("=" * 70)

# First, test with ONLY state rotation (no image rotation) to isolate state equivariance
print("\n[Test 1] State-only equivariance (same image, rotated state):")
for element in group.testing_elements:
    if element.value == 0:
        continue  # Skip identity
    
    rotated_obs = deepcopy(obs)
    angle = element.value * 2 * math.pi / n_group
    angle_deg = math.degrees(angle)
    
    # Keep image the SAME (no rotation)
    # Only rotate state
    rotated_state_tensor = getJointGeometricTensor(
        group, torch.tensor(deepcopy(state)).unsqueeze(1), is_action=False
    )
    rotated_state_tensor = rotated_state_tensor.transform(element)
    rotated_state_tensor = einops.rearrange(
        rotated_state_tensor.tensor, '(b t) c -> b t c', b=1
    )
    rotated_state = getActionOutput(rotated_state_tensor).squeeze(0).numpy()
    
    rotated_obs["state.left_hand"] = rotated_state[0, 14:20][None, :]
    rotated_obs["state.right_hand"] = rotated_state[0, 20:26][None, :]
    rotated_obs["state.l_ee_pos"] = rotated_state[0, :3][None, :]
    rotated_obs["state.l_ee_quat"] = rotated_state[0, 3:7][None, :]
    rotated_obs["state.r_ee_pos"] = rotated_state[0, 7:10][None, :]
    rotated_obs["state.r_ee_quat"] = rotated_state[0, 10:14][None, :]
    
    output = client.get_action(rotated_obs)
    output_action = torch.tensor(np.concatenate([
        output["action.l_ee_pos"], output["action.l_ee_quat"],
        output["action.r_ee_pos"], output["action.r_ee_quat"],
        output["action.left_hand"], output["action.right_hand"],
    ], axis=-1))
    
    rotated_origin_action = getJointGeometricTensor(
        group, torch.tensor(deepcopy(action)).unsqueeze(1), is_action=True
    ).transform(element)
    rotated_origin_action = einops.rearrange(rotated_origin_action.tensor, '(b t) c -> b t c', b=1)
    rotated_origin_action = getActionOutput(rotated_origin_action).squeeze(0)
    
    err = (rotated_origin_action - output_action).abs().mean()
    print(f"  g_{element.value} ({angle_deg:.0f}°): mean_err={err:.6f}")

# Second, test with ONLY image rotation (no state rotation) to check FA invariance
print("\n[Test 2] Image-only rotation (rotated image, same state):")
print("  If FA works, action should be SAME regardless of image rotation")
for element in group.testing_elements:
    if element.value == 0:
        continue
    
    rotated_obs = deepcopy(obs)
    angle = element.value * 2 * math.pi / n_group
    angle_deg = math.degrees(angle)
    
    # Rotate image only
    rotated_image = rotate_image(obs["video.cam_front"], angle)
    rotated_obs["video.cam_front"] = rotated_image
    # Keep state the SAME
    
    output = client.get_action(rotated_obs)
    output_action = torch.tensor(np.concatenate([
        output["action.l_ee_pos"], output["action.l_ee_quat"],
        output["action.r_ee_pos"], output["action.r_ee_quat"],
        output["action.left_hand"], output["action.right_hand"],
    ], axis=-1))
    
    # Compare to original action (should be same if FA makes features invariant)
    err = (torch.tensor(action) - output_action).abs().mean()
    print(f"  g_{element.value} ({angle_deg:.0f}°): diff from original={err:.6f}")

# Third, full equivariance test (rotate both)
print("\n[Test 3] Full equivariance (rotated image AND state):")
for element in group.testing_elements:
    rotated_obs = deepcopy(obs)
    
    # Compute rotation angle for this group element
    angle = element.value * 2 * math.pi / n_group
    angle_deg = math.degrees(angle)

    # Rotate IMAGE
    rotated_image = rotate_image(obs["video.cam_front"], angle)
    rotated_obs["video.cam_front"] = rotated_image
    
    # Save rotated image for visualization
    save_image(rotated_image[0], f"test_rotated_image_g{element.value}.png")

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

    print(f"\n--- Group element: g_{element.value} ({angle_deg:.1f}°) ---")
    
    rotated_obs["state.left_hand"] = rotated_state[0, 14:20][None, :]
    rotated_obs["state.right_hand"] = rotated_state[0, 20:26][None, :]
    rotated_obs["state.l_ee_pos"] = rotated_state[0, :3][None, :]
    rotated_obs["state.l_ee_quat"] = rotated_state[0, 3:7][None, :]
    rotated_obs["state.r_ee_pos"] = rotated_state[0, 7:10][None, :]
    rotated_obs["state.r_ee_quat"] = rotated_state[0, 10:14][None, :]
    
    # Get action from rotated input: f(g*x)
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
    
    # Transform original action by group element: g*f(x)
    rotated_origin_action = getJointGeometricTensor(
        group, torch.tensor(deepcopy(action)).unsqueeze(1), is_action=True
    )
    rotated_origin_action = rotated_origin_action.transform(element)
    rotated_origin_action = einops.rearrange(
        rotated_origin_action.tensor, '(b t) c -> b t c', b=1
    )
    
    rotated_origin_action = getActionOutput(rotated_origin_action).squeeze(0)
    
    # Compute error: f(g*x) vs g*f(x)
    err = (rotated_origin_action - output_action).abs().mean()
    max_err = (rotated_origin_action - output_action).abs().max()
    is_close = torch.allclose(output_action, rotated_origin_action, atol=1e-2)  # Relaxed tolerance
    
    status = "✓ PASS" if is_close else "✗ FAIL"
    print(f"  f(g*x) vs g*f(x): mean_err={err:.6f}, max_err={max_err:.6f} [{status}]")
    
    # Print detailed comparison for all group elements
    print(f"    f(g*x) l_ee_pos:  {output_action[0, :3].numpy()}")
    print(f"    g*f(x) l_ee_pos:  {rotated_origin_action[0, :3].numpy()}")
    print(f"    f(g*x) l_ee_quat: {output_action[0, 3:7].numpy()}")
    print(f"    g*f(x) l_ee_quat: {rotated_origin_action[0, 3:7].numpy()}")
    
    # Print detailed comparison for debugging
    if element.value == 0:
        print(f"\n  [Debug g_0] This should be identity - checking for non-determinism:")
        print(f"    Original action (first 6):   {action[0, :6]}")
        print(f"    f(g_0*x) action (first 6):   {output_action[0, :6].numpy()}")
        print(f"    g_0*f(x) action (first 6):   {rotated_origin_action[0, :6].numpy()}")
        
        # Also test calling get_action again on same input
        output2 = client.get_action(obs)
        action2 = np.concatenate([
            output2["action.l_ee_pos"],
            output2["action.l_ee_quat"],
            output2["action.r_ee_pos"],
            output2["action.r_ee_quat"],
            output2["action.left_hand"],
            output2["action.right_hand"],
        ], axis=-1)
        repeat_err = np.abs(action - action2).mean()
        print(f"    Repeat call error (determinism check): {repeat_err:.6f}")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
