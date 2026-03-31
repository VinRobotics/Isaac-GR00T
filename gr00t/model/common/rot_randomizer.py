import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotation_transformer import RotationTransformer


class RotRandomizer(nn.Module):
    """
    Randomly applies a Z-axis rotation augmentation during training.
    Rotates equivariant camera images, EE positions, and EE orientations
    by the same random angle per batch element.
    Does nothing during evaluation.
    """

    def __init__(
        self,
        rot_type: str,
        num_hand: int = 1,
        ee_dim: int = 7,
        rotate_image_indices: list[int] | None = None,
        num_images_per_sample: int = 1,
    ):
        """
        Args:
            rot_type: Rotation representation in state/action ("quaternion" or "euler_angles").
            num_hand: Number of end-effectors (1 or 2).
            ee_dim: Dimension per EE entry (7 for quaternion, 6 for euler_angles).
            rotate_image_indices: Which image slots (within each sample) to rotate.
                                  None means rotate all images.
            num_images_per_sample: Total number of images per sample in eagle_pixel_values.
        """
        super().__init__()
        self.rot_type = rot_type
        self.num_hand = num_hand
        self.ee_dim = ee_dim
        self.num_images_per_sample = num_images_per_sample
        self.rotate_image_indices = (
            list(range(num_images_per_sample))
            if rotate_image_indices is None
            else rotate_image_indices
        )
        self.tf = RotationTransformer(rot_type, "matrix")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_rotation_matrices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random Z-axis rotation matrices [B, 3, 3]."""
        angles = torch.rand(batch_size, device=device) * 2 * math.pi - math.pi
        rot = torch.zeros(batch_size, 3, 3, device=device)
        rot[:, 2, 2] = 1.0
        rot[:, 0, 0] = torch.cos(angles)
        rot[:, 0, 1] = -torch.sin(angles)
        rot[:, 1, 0] = torch.sin(angles)
        rot[:, 1, 1] = torch.cos(angles)
        return rot

    def _rotate_ee_tensor(self, tensor: torch.Tensor, rot_mat: torch.Tensor) -> torch.Tensor:
        """
        Rotate EE positions and orientations in a state or action tensor.

        Args:
            tensor:  [B, T, D] – state or action
            rot_mat: [B, 3, 3] – rotation matrix per batch element

        Returns:
            [B, T, D] rotated tensor
        """
        result = tensor.clone()
        for h in range(self.num_hand):
            offset = h * self.ee_dim

            # Rotate xyz position: [B, T, 3] -> apply R
            pos = result[:, :, offset : offset + 3]
            result[:, :, offset : offset + 3] = (
                rot_mat.to(pos.dtype) @ pos.permute(0, 2, 1)
            ).permute(0, 2, 1)

            # Rotate orientation
            rot_slice = result[:, :, offset + 3 : offset + self.ee_dim]
            if self.rot_type == "quaternion":
                # Stored as (x,y,z,w); RotationTransformer expects (w,x,y,z)
                quat_wxyz = rot_slice[:, :, [3, 0, 1, 2]]
                rot_mats = self.tf.forward(quat_wxyz)  # [B, T, 3, 3]
                composed = rot_mat.unsqueeze(1).to(rot_mats.dtype) @ rot_mats
                new_wxyz = self.tf.inverse(composed)  # [B, T, 4]
                result[:, :, offset + 3 : offset + self.ee_dim] = new_wxyz[:, :, [1, 2, 3, 0]]  # → (x,y,z,w)
            elif self.rot_type == "euler_angles":
                rot_mats = self.tf.forward(rot_slice)  # [B, T, 3, 3]
                composed = rot_mat.unsqueeze(1).to(rot_mats.dtype) @ rot_mats
                result[:, :, offset + 3 : offset + self.ee_dim] = self.tf.inverse(composed)

        return result

    def _rotate_images(
        self,
        backbone_inputs: dict,
        rot_mat: torch.Tensor,
        batch_size: int,
    ) -> dict:
        """
        Apply the same Z-axis rotation to equivariant images in eagle_pixel_values.
        Only images at self.rotate_image_indices are rotated; others are left unchanged.

        Args:
            backbone_inputs: dict containing "eagle_pixel_values" [B*num_imgs_per_sample, C, H, W]
            rot_mat: [B, 3, 3]
            batch_size: B
        """
        pixel_values = backbone_inputs["eagle_pixel_values"]  # [B*num_imgs_per_sample, C, H, W]
        _, C, H, W = pixel_values.shape

        # Reshape to [B, num_images_per_sample, C, H, W] for index-based access
        img_batch = pixel_values.reshape(batch_size, self.num_images_per_sample, C, H, W)

        rot_2d = rot_mat[:, :2, :2]  # [B, 2, 2]
        n_equi = len(self.rotate_image_indices)

        # Gather equivariant images: [B, n_equi, C, H, W] -> [B*n_equi, C, H, W]
        equi_imgs = torch.stack(
            [img_batch[:, idx] for idx in self.rotate_image_indices], dim=1
        ).reshape(batch_size * n_equi, C, H, W)

        # Build affine theta: [B*n_equi, 2, 3]
        rot_2d_exp = rot_2d.unsqueeze(1).expand(-1, n_equi, -1, -1).reshape(batch_size * n_equi, 2, 2)
        theta = torch.cat(
            [rot_2d_exp, torch.zeros(batch_size * n_equi, 2, 1, device=pixel_values.device)], dim=-1
        )

        grid = F.affine_grid(theta.to(equi_imgs.dtype), equi_imgs.shape, align_corners=True)
        rotated_equi = F.grid_sample(
            equi_imgs, grid, align_corners=True, mode="bilinear", padding_mode="border"
        )  # [B*n_equi, C, H, W]

        # Write rotated images back into their slots
        rotated_equi = rotated_equi.reshape(batch_size, n_equi, C, H, W)
        for slot, idx in enumerate(self.rotate_image_indices):
            img_batch[:, idx] = rotated_equi[:, slot]

        backbone_inputs["eagle_pixel_values"] = img_batch.reshape(
            batch_size * self.num_images_per_sample, C, H, W
        )
        return backbone_inputs

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, backbone_inputs, action_input):
        """
        Randomly rotate images and state/action by the same Z-axis angle.

        Args:
            backbone_inputs: BatchFeature with "eagle_pixel_values" [B*num_imgs, C, H, W]
            action_input:    BatchFeature with .state [B, T, D] and .action [B, T, D]

        Returns:
            (backbone_inputs, action_input) – both updated in-place.
        """
        if not self.training:
            return backbone_inputs, action_input

        B = action_input.state.shape[0]
        device = action_input.state.device

        # Retry until rotated xy positions remain inside the normalized [-1, 1] cube.
        for _ in range(1000):
            rot_mat = self._build_rotation_matrices(B, device)
            state_rot = self._rotate_ee_tensor(action_input.state, rot_mat)
            action_rot = self._rotate_ee_tensor(action_input.action, rot_mat)

            xy_ok = (
                state_rot[:, :, 0:2].abs().max() <= 1.0
                and action_rot[:, :, 0:2].abs().max() <= 1.0
            )
            if xy_ok:
                break
        else:
            # Could not find a valid rotation – skip augmentation for this batch
            return backbone_inputs, action_input

        backbone_inputs = self._rotate_images(backbone_inputs, rot_mat, B)
        action_input["state"] = state_rot
        action_input["action"] = action_rot

        return backbone_inputs, action_input

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rot_type={self.rot_type}, num_hand={self.num_hand}, ee_dim={self.ee_dim}, "
            f"rotate_image_indices={self.rotate_image_indices})"
        )
