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

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup 
import einops
import numpy as np


from typing import Union
import pytorch3d.transforms as pt
import torch
import numpy as np
import functools


from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer
from .equivariant_cross_attention_dit import EDiT


def get_prefix_weights(start: int, end: int, total: int, schedule: str) -> torch.Tensor:
    """
    With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    assert schedule in ["ones", "zeros", "linear", "exp"], f"Invalid schedule: {schedule}"
    start = min(start, end)
    idx = torch.arange(total, dtype=torch.float32)
    if schedule == "ones":
        w = torch.ones(total, dtype=torch.float32)
    elif schedule == "zeros":
        w = (idx < start).float()
    elif schedule == "linear" or schedule == "exp":
        w = torch.clamp((start - 1 - idx) / (end - start + 1) + 1, min=0, max=1)
        if schedule == "exp":
            # torch.expm1(x) = exp(x) - 1, torch.e = math.e
            w = w * torch.expm1(w) / (torch.tensor(torch.e) - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    w = torch.where(idx >= end, torch.tensor(0.0, dtype=w.dtype), w)
    return w


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)
### equi


class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self, 
            from_rep='axis_angle', 
            to_rep='rotation_6d', 
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(pt, f'{from_rep}_to_matrix'),
                getattr(pt, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(pt, f'matrix_to_{to_rep}'),
                getattr(pt, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])
        
        inverse_funcs = inverse_funcs[::-1]
        
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y
        
    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)
    
    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


class EquiCategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, in_type, hidden_type, out_type):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = EquiCategorySpecificLinear(num_categories, in_type, hidden_type)
        self.layer2 = EquiCategorySpecificLinear(num_categories, hidden_type, out_type)
        self.activation = enn.ReLU(hidden_type)

    def forward(self, x, cat_ids):
        hidden = self.activation(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)

class EquiCategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, in_type, out_type):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type

        self.layers = nn.ModuleList([
            enn.Linear(in_type, out_type, initialize=True)
            for _ in range(num_categories)
        ])

    def _group_indices(self, cat_ids):
        unique = torch.unique(cat_ids)
        groups = {}
        for c in unique:
            idx = (cat_ids == c).nonzero(as_tuple=False).view(-1)
            groups[int(c)] = idx
        return groups

    def forward(self, x: enn.GeometricTensor, cat_ids):
        B = x.tensor.shape[0]
        device = x.tensor.device
        
        # Collect all outputs and their original indices
        all_outputs = []
        all_indices = []
        groups = self._group_indices(cat_ids)

        for cat, idx in groups.items():
            xb = x.tensor[idx]
            geom_xb = enn.GeometricTensor(xb, self.in_type)
            layer = self.layers[cat]
            geom_out = layer(geom_xb)  # [num_in_group, dim]

            all_outputs.append(geom_out.tensor)
            all_indices.append(idx)

        # Concatenate all outputs: [B, dim]
        all_outputs = torch.cat(all_outputs, dim=0)
        all_indices = torch.cat(all_indices, dim=0)

        # Create inverse permutation to restore original order
        inverse_perm = torch.argsort(all_indices)
        out = all_outputs[inverse_perm]  # [B, dim]

        return enn.GeometricTensor(out, self.out_type)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, in_type, out_type, num_embodiments):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.num_embodiments = num_embodiments
        
        # Get gspace for creating field types
        gspace = out_type.gspace
        
        # Timestep embedding type - TRIVIAL because time doesn't rotate
        self.time_type = enn.FieldType(gspace, [gspace.trivial_repr] * out_type.size)
        
        # Hidden type = action features (regular) + timestep features (trivial)
        self.hidden_type = out_type + self.time_type

        # W1: action_dim -> out_dim (equivariant)
        self.W1 = EquiCategorySpecificLinear(num_embodiments, self.in_type, self.out_type)
        # W2: (out_dim + time_dim) -> out_dim (mixed input: regular + trivial)
        self.W2 = EquiCategorySpecificLinear(num_embodiments, self.hidden_type, self.out_type)
        # W3: out_dim -> out_dim (equivariant)
        self.W3 = EquiCategorySpecificLinear(num_embodiments, self.out_type, self.out_type)
        
        # Sinusoidal encoding for timestep (outputs trivial features)
        self.pos_encoding = SinusoidalPositionalEncoding(self.out_type.size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   GeometricTensor with shape (B*T, action_dim)
        timesteps: shape (B,) -- diffusion timestep per batch item
        cat_ids:   shape (B*T,)
        returns:   GeometricTensor with shape (B*T, out_type.size)
        """
        B = actions.tensor.shape[0]

        # Expand timesteps to match batch size if needed
        timesteps = timesteps.repeat((B // timesteps.shape[0]))

        # 1) Embed actions: (B*T, action_dim) -> (B*T, out_dim)
        a_emb = self.W1(actions, cat_ids)

        # 2) Get sinusoidal timestep encoding (B*T, out_dim) - TRIVIAL
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.tensor.dtype)
        
        # 3) Concatenate: [action_features (regular), timestep_features (trivial)]
        # This creates a tensor with hidden_type = out_type + time_type
        x = torch.cat([a_emb.tensor, tau_emb], dim=-1)
        x = enn.GeometricTensor(x, self.hidden_type)
        
        # 4) Project and apply swish: (B*T, out_dim + time_dim) -> (B*T, out_dim)
        x = swish(self.W2(x, cat_ids).tensor)

        # 5) Final projection: (B*T, out_dim) -> (B*T, out_dim)
        x = enn.GeometricTensor(x, self.out_type)
        x = self.W3(x, cat_ids)
        return x

@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "n_group": 8,
            "attention_head_dim": 48,
            "cross_attention_dim": 2048,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_attention_heads": 32,
            "num_layers": 16,
            "output_dim": 1024,
        }, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.config = config
        self.n_group = 8
        
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # self.model = DiT(**config.diffusion_model_cfg)
        self.model = EDiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # equi state
        self.group = gspaces.no_base_space(CyclicGroup(self.n_group))
        self.state_in_type = self.getJointFieldType(is_action=False)
        self.state_hidden_type = enn.FieldType(self.group, int(config.hidden_size / self.n_group) * [self.group.regular_repr])
        self.state_out_type = enn.FieldType(self.group, int(config.input_embedding_dim / self.n_group) * [self.group.regular_repr])
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')

        self.state_encoder = EquiCategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            in_type=self.state_in_type,
            hidden_type=self.state_hidden_type,
            out_type=self.state_out_type,
        )
        
        self.action_type = self.getJointFieldType(is_action=True)
        self.action_out_type = enn.FieldType(self.group, int(self.input_embedding_dim / self.n_group) * [self.group.regular_repr])
        
        self.action_encoder = MultiEmbodimentActionEncoder(
            in_type=self.action_type,
            out_type=self.action_out_type,
            num_embodiments=config.max_num_embodiments,
        )
        
        self.action_decoder = EquiCategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            in_type=self.state_hidden_type,
            hidden_type=self.state_hidden_type,
            out_type=self.action_type,
        )
        
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )
        

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)
    
    def getJointFieldType(self, is_action):
        max_dim = self.config.max_action_dim if is_action else self.config.max_state_dim
        return enn.FieldType(
            self.group,
            2 * 4 * [self.group.irrep(1)] # pos xy, rot 6, left and right
            + (max_dim - 14 + 2) * [self.group.trivial_repr], # gripper 1, z from both ee is 2
        )
        
    def getJointGeometricTensor(self, state, is_action):
        def getJointGeometricTensorEachHand(ee_state):
            ee_pos = ee_state[:, :, :3] # (bs, t, 3)
            ee_quat = ee_state[:, :, 3:7] # (bs, t, 4)
            ee_rot = self.get6DRotation(ee_quat)
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

        return enn.GeometricTensor(state_features, self.getJointFieldType(is_action))
    
    def getActionGT(self, action):
        def getActionGTEachHand(ee_state):
            ee_pos = ee_state[:, :, :3] # (bs, t, 3)
            ee_quat = ee_state[:, :, 3:7] # (bs, t, 4)
            ee_rot = self.get6DRotation(ee_quat)
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
        
        l_ee_state = action[:, :, :7] # bs, t, 7 
        r_ee_state = action[:, :, 7:14] # bs, t, 7  
        hand_state = action[:, :, 14:]
        
        l_tf, l_pos_z = getActionGTEachHand(l_ee_state)
        r_tf, r_pos_z = getActionGTEachHand(r_ee_state)

        state_features = torch.cat([l_tf, r_tf, l_pos_z, r_pos_z, hand_state], dim=-1)
        return state_features
    
    
    def getActionOutput(self, pred):
        def getActionOutputEachHand(ee_pred):
            ee_xy = ee_pred[:, :, 0:2] # (bs, t, 3)
            ee_cos1 = ee_pred[:, :, 2:3]
            ee_sin1 = ee_pred[:, :, 3:4]
            ee_cos2 = ee_pred[:, :, 4:5]
            ee_sin2 = ee_pred[:, :, 5:6]
            ee_cos3 = ee_pred[:, :, 6:7]
            ee_sin3 = ee_pred[:, :, 7:8]

            rot_6d = torch.cat([ee_cos1, ee_cos2, ee_cos3, ee_sin1, ee_sin2, ee_sin3], dim=-1)
            quat = self.getQuaternionFrom6D(rot_6d)
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

    def get6DRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        return self.quaternion_to_sixd.forward(quat[:, :, [3, 0, 1, 2]]) 
    
    def getQuaternionFrom6D(self, rot_6d):
        quat = self.quaternion_to_sixd.inverse(rot_6d)
        return quat[:, :, [1, 2, 3, 0]]  # xyzw

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            print("set requires_grad false for tune projector")
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            self.vl_equi_proj.requires_grad_(False)
            self.future_tokens_equi_proj.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            print("set requires_grad false for tune diffusion model")
            
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Custom load_state_dict that handles the transition from old state_encoder (simple linear)
        to new state_encoder (ESCNN equivariant).
        
        Old checkpoint has: action_head.state_encoder.layer1.W, action_head.state_encoder.layer1.b
        New model has: action_head.state_encoder.layer1.layers.*.* (ESCNN linear layers)
        
        This method filters out incompatible state_encoder keys to prevent weight mismatches.
        """
        # Filter out old state_encoder weights that don't match the new ESCNN architecture
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # Skip old-style state_encoder weights
            if 'state_encoder.layer' in key or 'action_encoder' in key or 'action_decoder' in key or "model" in key or "future_tokens_equi_proj" in key or "vl_equi_proj" in key:
                print(f"Skipping incompatible state_encoder weight: {key}")
                continue
            filtered_state_dict[key] = value
        # print(filtered_state_dict)
        # Call parent's load_state_dict with filtered state
        return super().load_state_dict(filtered_state_dict, strict=False, assign=assign)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                print("not tune projector")
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                print("not tune diffusion model")
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        B, T, _ = action_input.state.shape
        state_input = self.getJointGeometricTensor(action_input.state, is_action=False)
        state_features = self.state_encoder(state_input, embodiment_id)
        state_features = state_features.tensor
        state_features = einops.rearrange(state_features, '(b t) c -> b t c', b=B, t=T)
        
        # Embed noised action trajectory.
        actions = action_input.action
        
        actions_gt = self.getActionGT(actions)
        noise = torch.randn(actions_gt.shape, device=actions_gt.device, dtype=actions_gt.dtype)
        t = self.sample_time(actions_gt.shape[0], device=actions_gt.device, dtype=actions_gt.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions_gt
        velocity = actions_gt - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        
        noisy_trajectory = einops.rearrange(
            noisy_trajectory, "b t c -> (b t) c"
        )
        
        noisy_trajectory = enn.GeometricTensor(noisy_trajectory, self.getJointFieldType(True))

        action_encoder_embodiment_id = embodiment_id.repeat((actions_gt.shape[1]))
        action_features = self.action_encoder(noisy_trajectory, t_discretized, action_encoder_embodiment_id)
        action_features = action_features.tensor
        action_features = einops.rearrange(
            action_features,
            '(b t) c -> b t c',
            b=actions_gt.shape[0],
            t=actions_gt.shape[1]
        )

        # # Maybe add position embedding.
        # if self.config.add_pos_embed:
        #     pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        #     pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        #     action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        # future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        
        # future_tokens = einops.rearrange(
        #     future_tokens, "b t d -> (b t) d"
        # )
        # future_tokens = enn.GeometricTensor(future_tokens, self.future_tokens_in_type)
        # future_tokens = self.future_tokens_equi_proj(future_tokens)
        # future_tokens = einops.rearrange(
        #     future_tokens.tensor, "(b t) d -> b t d", b = B, t = self.config.num_target_vision_tokens
        # )
        
        sa_embs = torch.cat((state_features, action_features), dim=1)
        B, T, C = sa_embs.shape
        B, Tv, Cv = vl_embs.shape
        # sa embs: B, T, G*D
        # project vl embs to 2d space
        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        
        action_decoder_embodiment_id = embodiment_id.repeat((model_output.shape[1]))

        model_output = einops.rearrange(
            model_output,
            'b t c -> (b t) c',
        )
        model_output = enn.GeometricTensor(model_output, self.state_hidden_type)
        pred = self.action_decoder(model_output, action_decoder_embodiment_id)

        pred = einops.rearrange(
            pred.tensor,
            '(b t) c -> b t c',
            b=sa_embs.shape[0],
            t=sa_embs.shape[1]
        )

        pred_actions = pred[:, -actions_gt.shape[1] :, :]

        # pred_actions = self.getActionOutput(pred_actions)
        # Slice out only the action portion of pred and target.
        with torch.no_grad():
            B, T, original_action_dim = action_input.action_mask.shape
            velocity_dim = velocity.shape[-1]
            
            # Create action mask that matches the velocity dimension
            action_mask = torch.zeros((B, T, velocity_dim), device=velocity.device)
            
            # Copy original mask values
            action_mask[:, :, :original_action_dim] = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        self.model.eval()
        self.action_decoder.eval()
        self.state_encoder.eval()
        self.action_encoder.eval()
        
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        
        embodiment_id = action_input.embodiment_id

        # Embed state.
        B, T, _ = action_input.state.shape
        state_input = self.getJointGeometricTensor(action_input.state, is_action=False)
        state_features = self.state_encoder(state_input, embodiment_id)
        state_features = state_features.tensor
        state_features = einops.rearrange(state_features, '(b t) c -> b t c', b=B, t=T)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.tensor(
            [[[9.8907e-01, 8.9235e-01, 2.2414e-01, 4.7390e-03, 8.8978e-02,
            9.5681e-01, 8.1972e-01, 8.6800e-01, 6.4302e-01, 5.5405e-01,
            1.6448e-01, 7.8000e-01, 4.5793e-01, 8.8432e-01, 7.0771e-02,
            2.2561e-01, 3.0732e-01, 1.0589e-01, 5.5346e-01, 2.8520e-01,
            5.5303e-01, 4.1317e-01, 4.4095e-01, 8.3718e-01, 1.6206e-01,
            1.8077e-01, 3.5778e-01, 7.9089e-01, 4.2683e-01, 3.1968e-01,
            2.7926e-01, 4.6051e-01, 2.3046e-01, 3.7925e-03, 6.0060e-01,
            4.4165e-01],
            [2.8513e-01, 3.3400e-01, 7.9122e-01, 9.7216e-01, 5.2621e-01,
            7.0183e-01, 8.6866e-01, 7.2411e-01, 4.0245e-01, 9.4150e-01,
            2.3868e-01, 2.1945e-01, 3.4592e-01, 6.7118e-01, 2.4564e-02,
            7.9741e-01, 6.3326e-01, 2.8713e-01, 1.4226e-02, 1.9164e-01,
            8.3274e-01, 8.0732e-01, 9.4645e-01, 9.9499e-02, 5.5563e-01,
            5.6521e-01, 2.9681e-01, 6.6982e-02, 4.3161e-01, 9.0992e-01,
            1.1349e-02, 6.4707e-01, 6.4069e-01, 1.0034e-01, 4.3677e-01,
            3.0956e-01],
            [2.1589e-01, 7.8849e-01, 8.9463e-01, 4.1971e-01, 5.8723e-01,
            7.6195e-01, 1.3945e-01, 2.9714e-01, 9.9551e-01, 7.3140e-01,
            8.2838e-01, 3.0297e-01, 1.2700e-02, 3.0283e-01, 7.4363e-01,
            7.2849e-01, 2.7161e-01, 8.5186e-01, 7.8261e-01, 4.4626e-01,
            7.5512e-01, 1.5149e-01, 2.8515e-01, 8.9218e-01, 5.2147e-01,
            6.7073e-01, 6.4412e-01, 5.9634e-01, 2.4536e-01, 9.9374e-01,
            8.0660e-01, 3.2658e-01, 3.5005e-01, 2.5861e-01, 6.8739e-01,
            2.5821e-01],
            [1.9654e-02, 8.8153e-01, 4.7600e-01, 1.0869e-01, 4.8524e-01,
            3.4270e-01, 1.5668e-01, 1.5009e-01, 8.3253e-01, 8.1213e-01,
            4.6524e-01, 3.4344e-01, 4.6745e-01, 9.9040e-01, 1.7940e-01,
            2.1683e-01, 5.9956e-01, 7.9347e-01, 7.9091e-01, 2.4836e-01,
            4.3231e-01, 2.1375e-01, 1.7844e-01, 9.3271e-01, 8.6684e-02,
            5.8755e-01, 4.7764e-02, 1.7581e-01, 6.2742e-01, 5.3021e-01,
            1.7433e-01, 7.5125e-01, 7.5527e-01, 6.2374e-01, 4.0680e-01,
            9.3000e-02],
            [5.8635e-01, 2.4494e-01, 7.0762e-01, 8.4780e-02, 7.3087e-01,
            4.2170e-02, 6.3070e-02, 6.4160e-01, 3.4711e-01, 5.6312e-01,
            2.3958e-01, 4.6903e-01, 5.1101e-01, 3.9880e-01, 9.4331e-01,
            3.5758e-01, 5.1914e-01, 4.2688e-02, 7.4113e-01, 9.2126e-01,
            6.8851e-01, 1.3341e-01, 1.3172e-01, 3.0636e-01, 7.4142e-01,
            9.3512e-01, 1.8293e-01, 1.4279e-01, 3.6770e-01, 8.0039e-02,
            2.5757e-01, 8.6610e-01, 1.1392e-01, 5.0768e-01, 5.4617e-01,
            1.1914e-01],
            [2.1788e-01, 9.6061e-02, 7.3506e-01, 3.2786e-01, 9.9928e-01,
            6.7003e-02, 7.2869e-01, 9.3967e-01, 5.0180e-01, 8.3392e-01,
            9.6212e-01, 9.3472e-01, 9.0528e-01, 1.0035e-01, 3.0143e-01,
            8.0891e-01, 2.7840e-01, 1.4839e-01, 4.5029e-01, 1.6506e-01,
            8.7121e-01, 9.0394e-01, 2.0959e-01, 2.7498e-01, 3.1311e-01,
            4.1314e-01, 1.8723e-01, 1.2990e-03, 8.7132e-01, 6.9606e-01,
            8.7222e-02, 5.1975e-01, 5.3345e-01, 6.3867e-01, 6.9103e-01,
            7.5153e-01],
            [3.3729e-01, 8.7868e-01, 3.4206e-02, 6.8306e-01, 6.9636e-01,
            5.0403e-01, 6.3598e-03, 7.4601e-01, 3.2566e-01, 3.0089e-01,
            7.3883e-01, 9.1219e-01, 8.9675e-01, 4.8840e-01, 3.7745e-01,
            7.8829e-01, 8.2394e-01, 9.8085e-01, 4.4697e-01, 1.9450e-01,
            4.3534e-01, 3.4490e-01, 8.1664e-01, 8.3259e-01, 1.2913e-01,
            5.9998e-01, 2.2435e-01, 6.8950e-01, 8.3761e-01, 7.6445e-01,
            9.9988e-01, 9.6651e-01, 1.5804e-01, 2.9779e-01, 9.2610e-01,
            8.5797e-01],
            [7.4398e-01, 4.5306e-01, 5.7967e-01, 5.9724e-01, 2.2655e-01,
            8.0820e-01, 3.7991e-01, 9.7291e-02, 3.4921e-01, 2.0167e-01,
            4.4333e-01, 7.0007e-01, 7.8807e-01, 1.1505e-01, 6.1875e-01,
            8.9602e-01, 1.2650e-01, 5.8252e-01, 7.1503e-02, 9.6537e-01,
            4.6619e-01, 8.0964e-01, 9.6920e-01, 3.1073e-01, 2.7848e-01,
            7.9352e-01, 7.9384e-01, 2.0867e-01, 3.9298e-01, 9.9342e-01,
            3.0566e-01, 5.7617e-01, 2.0294e-01, 1.4986e-01, 2.8628e-01,
            5.2915e-01],
            [8.3482e-01, 6.6598e-01, 5.0660e-01, 3.7039e-03, 4.6411e-01,
            5.2080e-01, 9.7806e-01, 9.2093e-01, 6.3366e-01, 3.8087e-01,
            2.5400e-01, 1.7846e-01, 7.4497e-01, 5.7329e-01, 6.2490e-01,
            4.1944e-01, 4.2706e-01, 6.7995e-02, 5.5677e-01, 4.6686e-01,
            5.6131e-02, 9.2367e-01, 1.4353e-01, 2.0277e-01, 7.8544e-01,
            3.8158e-01, 5.7946e-01, 4.4055e-01, 7.8545e-01, 1.5391e-01,
            7.0071e-01, 7.1391e-01, 2.8456e-01, 8.7790e-01, 3.0585e-01,
            3.6887e-01],
            [5.1375e-01, 6.9643e-01, 9.8314e-01, 9.8055e-01, 7.9865e-01,
            1.9615e-01, 5.2861e-01, 9.7249e-01, 3.8719e-01, 9.3055e-01,
            7.9849e-02, 4.2973e-01, 3.9248e-01, 3.6235e-01, 2.3094e-01,
            1.1177e-01, 4.8303e-01, 1.5214e-01, 7.7484e-01, 8.5285e-02,
            1.7147e-01, 9.6112e-01, 7.8893e-01, 4.4498e-01, 5.7421e-01,
            4.2003e-01, 9.5980e-01, 1.1326e-01, 1.3167e-01, 3.9133e-01,
            1.8681e-01, 4.5531e-01, 7.5378e-01, 7.0437e-01, 9.9888e-01,
            3.2985e-01],
            [2.2387e-01, 1.6721e-01, 7.1274e-01, 9.9917e-01, 8.1893e-01,
            1.6414e-01, 5.8091e-01, 2.9513e-01, 9.2879e-01, 5.8158e-01,
            7.6002e-01, 1.0084e-01, 5.1427e-01, 7.3032e-01, 3.3027e-01,
            3.9236e-01, 2.9495e-01, 1.3506e-01, 3.9833e-01, 8.5263e-01,
            5.5124e-02, 2.0411e-02, 9.6893e-01, 1.5180e-01, 5.3176e-01,
            6.6951e-02, 5.1185e-01, 5.7006e-02, 4.1550e-01, 7.1453e-02,
            2.4479e-02, 4.2714e-02, 3.9453e-01, 5.1047e-01, 9.4019e-01,
            7.0791e-01],
            [8.9799e-01, 7.8527e-01, 2.7943e-01, 1.7642e-01, 6.8461e-01,
            4.2754e-01, 7.9616e-01, 2.5773e-01, 1.7838e-01, 7.4846e-01,
            2.5653e-01, 3.5337e-01, 4.5636e-01, 6.3733e-01, 6.8289e-01,
            2.8067e-01, 8.3056e-01, 3.0822e-01, 7.5869e-01, 7.8437e-01,
            1.8107e-01, 6.5200e-01, 4.3651e-01, 3.4693e-01, 4.7691e-01,
            3.9588e-01, 6.1873e-01, 9.7135e-01, 2.5828e-01, 9.6926e-01,
            9.4727e-01, 4.0805e-01, 2.0499e-01, 9.8314e-01, 5.4119e-01,
            4.0049e-01],
            [4.2851e-01, 3.9804e-01, 7.3489e-01, 2.2130e-01, 3.1172e-01,
            2.4455e-01, 9.4005e-02, 5.8548e-01, 9.8813e-01, 6.8956e-01,
            6.9947e-01, 7.7407e-01, 6.8738e-01, 7.1465e-01, 2.0684e-01,
            4.9391e-01, 1.8838e-01, 9.1970e-01, 4.0157e-03, 7.0033e-01,
            7.0611e-01, 1.9578e-01, 5.9608e-01, 4.8023e-01, 8.6803e-02,
            5.3307e-01, 2.3984e-02, 7.6525e-01, 2.5520e-01, 6.8776e-01,
            5.0157e-01, 1.1211e-01, 7.4110e-01, 2.1815e-01, 8.4678e-01,
            4.6412e-01],
            [3.7567e-01, 9.4264e-02, 4.4201e-02, 7.8077e-01, 3.9002e-01,
            8.4992e-01, 9.4099e-01, 9.0149e-01, 5.5527e-01, 5.6511e-01,
            2.5279e-01, 6.6812e-01, 3.5094e-01, 8.7894e-01, 9.1029e-01,
            3.4258e-01, 2.2299e-01, 7.4189e-01, 1.6391e-01, 3.8211e-01,
            3.3700e-01, 9.1559e-01, 2.5457e-01, 7.4586e-01, 4.5064e-01,
            2.0146e-01, 7.2413e-01, 4.6876e-01, 3.6133e-01, 5.1551e-01,
            2.7166e-01, 4.9087e-01, 7.9115e-01, 8.5925e-01, 5.2100e-01,
            1.9977e-01],
            [2.9858e-01, 9.1536e-01, 8.5281e-01, 3.1588e-01, 5.9320e-01,
            9.9388e-01, 2.7641e-02, 6.8108e-01, 2.8789e-01, 1.2685e-01,
            5.9330e-01, 7.3170e-01, 5.5751e-01, 1.1626e-01, 6.9055e-01,
            7.7664e-01, 7.0234e-01, 8.5447e-01, 1.0833e-01, 7.6150e-01,
            7.6125e-01, 4.9015e-01, 9.0106e-01, 2.8828e-01, 8.6388e-01,
            2.8713e-01, 8.1961e-01, 1.1609e-01, 8.7049e-01, 6.0052e-01,
            8.3579e-01, 1.4917e-01, 4.0301e-01, 5.7908e-01, 2.4192e-01,
            4.9658e-01],
            [4.1645e-01, 9.3053e-01, 7.1018e-01, 6.6776e-01, 3.5112e-01,
            5.1364e-02, 3.1296e-01, 1.7182e-01, 2.7783e-01, 6.0082e-01,
            5.5890e-01, 6.6957e-01, 5.4132e-01, 1.1313e-04, 5.2363e-01,
            2.5444e-01, 6.1992e-01, 2.0678e-01, 1.1442e-01, 6.3573e-01,
            5.0373e-02, 3.8620e-01, 2.1184e-01, 6.0320e-01, 6.4616e-01,
            7.4350e-01, 8.3359e-01, 6.2860e-02, 5.7594e-01, 5.3040e-01,
            5.0834e-01, 8.5241e-01, 1.7429e-01, 1.2514e-01, 3.1657e-01,
            4.9281e-01]]],
                    dtype=vl_embs.dtype,
                    device=device,
                )
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_encoder_embodiment_id = embodiment_id.repeat((actions.shape[1]))
            noisy_trajectory = einops.rearrange(
                actions, "b t c -> (b t) c"
            )
            noisy_trajectory = enn.GeometricTensor(noisy_trajectory, self.getJointFieldType(True))
            action_features = self.action_encoder(noisy_trajectory, timesteps_tensor, action_encoder_embodiment_id)
            action_features = action_features.tensor
            action_features = einops.rearrange(
                action_features,
                '(b t) c -> b t c',
                b=actions.shape[0],
                t=actions.shape[1]
            )

            sa_embs = torch.cat((state_features, action_features), dim=1)
            
            B, T, C = sa_embs.shape
            B, Tv, Cv = vl_embs.shape
            # sa embs: B, T, G*D

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            
            action_decoder_embodiment_id = embodiment_id.repeat((model_output.shape[1]))

            model_output = einops.rearrange(
                model_output,
                'b t c -> (b t) c',
            )
            model_output = enn.GeometricTensor(model_output, self.state_hidden_type)
            pred = self.action_decoder(model_output, action_decoder_embodiment_id)

            pred = einops.rearrange(
                pred.tensor,
                '(b t) c -> b t c',
                b=sa_embs.shape[0],
                t=sa_embs.shape[1]
            )

            pred_velocity = pred[:, -self.action_horizon :]
            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
            
        actions = self.getActionOutput(actions)
        return BatchFeature(data={"action_pred": actions})
    
    @torch.enable_grad()
    def get_realtime_action(
        self,
        action_input: BatchFeature,
        backbone_output:BatchFeature,
        prev_action_chunk: torch.Tensor,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str,
        max_guidance_weight: float,
    )  -> BatchFeature:
        torch.set_grad_enabled(True)
        num_steps = self.num_inference_timesteps
        self.sigma_d_o = 1.0
        dt = 1.0 / num_steps
        prev_action_chunk = torch.as_tensor(prev_action_chunk, device=self.device, dtype=self.dtype)

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds      = backbone_output.backbone_features          # [B, T_vl, D]
        embodiment_id  = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        x_t = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        for t in range(num_steps):
            # weights: [horizon]
            weights = get_prefix_weights(
                inference_delay, prefix_attention_horizon, self.config.action_horizon, prefix_attention_schedule
            )
            weights = weights.to(device)

            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            
            def denoiser(x_t_):
                t_discretized = int(t_cont * self.num_timestep_buckets)

                # Embed noised action trajectory.
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(x_t_, timesteps_tensor, embodiment_id)
                # Maybe add position embedding.
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                vl_embs = vl_embeds

                # Join vision, language, state and action embedding along sequence dimension.
                sa_embs = torch.cat((state_features, action_features), dim=1)

                # Run model forward.
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
                pred = self.action_decoder(model_output, embodiment_id)

                pred_velocity = pred[:, -self.action_horizon :]
                return x_t_ + pred_velocity * (1 - t_cont), pred_velocity
            
            (outputs, vjp_func) = torch.func.vjp(denoiser, x_t)
            (x_1_i_vjp, v_t_i_vjp) = outputs
            error = (prev_action_chunk - x_1_i_vjp) * weights[:, None]
            
            pinv_correction = vjp_func((error, torch.zeros_like(x_t)))[0]
            if pinv_correction is None:
                pinv_correction = torch.zeros_like(x_1_i_vjp)
            inv_r2 = (self.sigma_d_o**2 * t_cont**2 + (1 - t_cont)**2) / (self.sigma_d_o**2 * (1 - t_cont)**2)
            # inv_r2 = (t_cont**2 + (1 - t_cont) ** 2) / ((1 - t_cont) ** 2)
            c = torch.nan_to_num(torch.tensor((1 - t_cont) / max(t_cont, 1e-12), device=self.device, dtype=self.dtype),  # Avoid division by zero
                                 nan=0.0, posinf=max_guidance_weight)
            
            guidance_weight = torch.minimum(c * inv_r2, torch.tensor(max_guidance_weight, device=device))
            v_t_corr = v_t_i_vjp + guidance_weight * pinv_correction

            x_t = x_t + dt * v_t_corr

        assert x_t.shape == (batch_size, self.config.action_horizon, self.config.action_dim), x_t.shape
        x_t = x_t.clone().detach()
        return BatchFeature(data={"action_pred": x_t})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
