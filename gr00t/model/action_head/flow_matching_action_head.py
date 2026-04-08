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
from gr00t.model.common import RotationTransformer

from .equivariant_cross_attention_dit import EDiT, EquiVisionTransformer


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

    n_group: int = field(
        default=8, metadata={"help": "Number of groups for equivariant operations (cyclic group order)."}
    )
    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "attention_head_dim": 48,
            "cross_attention_dim": 2048,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_attention_heads": 32,
            "num_layers": 16,
            "output_dim": 1024,
            "use_relative_position_bias": False,  # disabled: equivariant absolute pos embeds handle ordering
        }, metadata={"help": "Diffusion model configuration (n_group is injected from top-level)."}
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

    vl_self_attention_cfg: dict = field(
        default_factory=lambda: {
        "attention_head_dim": 64,
        "dropout": 0.2,
        "final_dropout": True,
        "num_attention_heads": 32,
        "num_layers": 4,
        "positional_embeddings": None
        }, metadata={"help": "VL self-attention configuration (n_group is injected from top-level)."}
    )
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )
    num_hand: int = field(
        default=2, metadata={"help": "Whether to use two hands (True) or one hand (False)."}
    )
    
    rot_type: str = field(
        default="quaternion", metadata={"help": "Define rot type: quaternion, euler_angles"}
    )
    
    rel_action: bool = field(
        default=False, metadata={"help": "Whether to predict actions in relative (velocity) space instead of absolute state space."}
    )
    
    backbone_language_embedding_dim: int = field(
        default=2048, metadata={"help": "Language feature dim from backbone LLM (D_llm)."}
    )
    def __init__(self, **kwargs):
        import dataclasses
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Ensure all dataclass fields with defaults become instance attributes
        # so PretrainedConfig.to_dict() (which uses self.__dict__) saves them.
        for f in dataclasses.fields(self):
            if f.name not in self.__dict__:
                if f.default is not dataclasses.MISSING:
                    setattr(self, f.name, f.default)
                elif f.default_factory is not dataclasses.MISSING:
                    setattr(self, f.name, f.default_factory())


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.config = config
        self.n_group = config.n_group

        self.num_hand = self.config.num_hand

        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Inject n_group into diffusion model config
        diffusion_cfg = {
            **config.diffusion_model_cfg,
            "n_group": self.n_group,
        }
        self.model = EDiT(**diffusion_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # equi state
        self.rel_action = self.config.rel_action
        self.ee_dim = 7 if self.config.rot_type == "quaternion" else 6
        self.real_dim = 12 if self.rel_action else 9
        self.group = gspaces.no_base_space(CyclicGroup(self.n_group))
        self.state_in_type = self.getJointFieldType(is_action=False)
        self.state_hidden_type = enn.FieldType(self.group, int(config.hidden_size / self.n_group) * [self.group.regular_repr])
        self.state_out_type = enn.FieldType(self.group, int(config.input_embedding_dim / self.n_group) * [self.group.regular_repr])
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')
        self.axisangle_to_sixd = RotationTransformer('axis_angle', 'rotation_6d', from_convention="ZYX")
        
        self.quaternion_to_matrix = RotationTransformer('quaternion', 'matrix')
        self.axisangle_to_matrix = RotationTransformer('axis_angle', 'matrix', from_convention="ZYX")
        
        self.state_encoder = EquiCategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            in_type=self.state_in_type,
            hidden_type=self.state_hidden_type,
            out_type=self.state_out_type,
        )
        
        self.action_type = self.getJointFieldType(is_action=True) if not self.rel_action else self.getActionRelFieldType(is_action=True)
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

        if config.add_pos_embed:
            # Equivariant temporal position embeddings for state and action tokens.
            # Embeddings are in trivial (invariant) repr, then projected to regular repr via
            # enn.Linear — this is equivariant because trivial → regular is always equivariant.
            _D = self.model.in_type.size
            self.temporal_pos_trivial_type = enn.FieldType(
                self.group, [self.group.trivial_repr] * _D
            )
            self.temporal_pos_embed = nn.Embedding(config.max_seq_len, _D)
            self.temporal_pos_proj = enn.Linear(self.temporal_pos_trivial_type, self.model.in_type)
            nn.init.normal_(self.temporal_pos_embed.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        
        self.p = torch.tensor([
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 1, 0, 0, 0, 0]
        ]).float()
        self.p_inv = torch.linalg.inv(self.p)
        
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)
    
    def getJointFieldType(self, is_action):
        max_dim = self.config.max_action_dim if is_action else self.config.max_state_dim
        return enn.FieldType(
            self.group,
            self.num_hand * 4 * [self.group.irrep(1)] # pos xy, rot 6, left and right
            + (max_dim - ((self.ee_dim - 1) * self.num_hand)) * [self.group.trivial_repr], # gripper 1, z from both ee is 2
        )
        
    def getActionRelFieldType(self, is_action):
        max_dim = self.config.max_action_dim if is_action else self.config.max_state_dim

        # Per-hand layout (matches getActionGTRelEachHand transform_features):
        # [rho2(2), xy(2), rho12(2), rho11(2), rho01(1), rho02(1), rho03(1)] = 11 dims
        # Equivariant: irrep(2) + 3*irrep(1) = 8 dims
        # Trivial from rotation decomp: 3 trivials (rho01/02/03)
        per_hand = (
            1 * [self.group.irrep(2)]       # rho2: 2 dims
            + 3 * [self.group.irrep(1)]     # xy, rho12, rho11: 6 dims
            + 3 * [self.group.trivial_repr] # rho01, rho02, rho03: 3 dims
        )

        # Remaining trivials: z per hand + hand_state = max_dim - (ee_dim-1)*num_hand
        return enn.FieldType(
            self.group,
            self.num_hand * per_hand
            + (max_dim - (self.ee_dim - 1) * self.num_hand) * [self.group.trivial_repr]
        )
        
    def getJointGeometricTensor(self, state, is_action):
        def getJointGeometricTensorEachHand(ee_state):
            ee_pos = ee_state[:, :, :3] # (bs, t, 3)
            ee_quat = ee_state[:, :, 3:self.ee_dim] # (bs, t, 4)
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
        if self.num_hand == 2:
            l_ee_state = state[:, :, :self.ee_dim] # bs, t, 7 
            r_ee_state = state[:, :, self.ee_dim:self.ee_dim*2] # bs, t, 7  
            hand_state = state[:, :, self.ee_dim*2:]
            
            l_tf, l_pos_z = getJointGeometricTensorEachHand(l_ee_state)
            r_tf, r_pos_z = getJointGeometricTensorEachHand(r_ee_state)

            state_features = torch.cat([l_tf, r_tf, l_pos_z, r_pos_z, hand_state], dim=-1)
 
        else:
            l_ee_state = state[:, :, :self.ee_dim] # bs, t, 7 
            hand_state = state[:, :, self.ee_dim:]
            
            l_tf, l_pos_z = getJointGeometricTensorEachHand(l_ee_state)
            state_features = torch.cat([l_tf, l_pos_z, hand_state], dim=-1)      
        state_features = einops.rearrange(state_features, 'b t c -> (b t) c')
        return enn.GeometricTensor(state_features, self.getJointFieldType(is_action))
    
    def getActionGT(self, action):
        def getActionGTRelEachHand(ee_state):
            ee_pos = ee_state[:, :, :3] # (bs, t, 3)
            ee_quat = ee_state[:, :, 3:self.ee_dim] # (bs, t, 4)
            ee_rho = self.getMatrixRotation(ee_quat) # (bs, t, 9)
            pos_xy = ee_pos[:, :, 0:2] # 2
            pos_z = ee_pos[:, :, 2:3] # 1
            
            ee_rho = torch.matmul(self.p.to(ee_state.device), ee_rho.transpose(-1, -2)).transpose(-1, -2) # (bs, t, 9)
            
            ee_rho01, ee_rho02, ee_rho03, ee_rho11, ee_rho12, ee_rho2 = ee_rho[:, :, 0:1], ee_rho[:, :, 1:2], ee_rho[:, :, 2:3], ee_rho[:, :, 3:5], ee_rho[:, :, 5:7], ee_rho[:, :, 7:9]
            
            transform_features = torch.cat(
                [
                    ee_rho2, #2
                    pos_xy, #2
                    ee_rho12,
                    ee_rho11,
                    ee_rho01,
                    ee_rho02,
                    ee_rho03,
                ],
                dim=-1
            ) 
            
            return transform_features, pos_z
        
        def getActionGTEachHand(ee_state):
            ee_pos = ee_state[:, :, :3] # (bs, t, 3)
            ee_quat = ee_state[:, :, 3:self.ee_dim] # (bs, t, 4)
            ee_rot = self.get6DRotation(ee_quat)
            pos_xy = ee_pos[:, :, 0:2] # 2
            pos_z = ee_pos[:, :, 2:3] # 1
            transform_features = torch.cat(
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
            return transform_features, pos_z
        action_transform = getActionGTRelEachHand if self.rel_action else getActionGTEachHand
        if self.num_hand == 2:
            l_ee_state = action[:, :, :self.ee_dim] # bs, t, 7 
            r_ee_state = action[:, :, self.ee_dim:self.ee_dim*2] # bs, t, 7  
            hand_state = action[:, :, self.ee_dim*2:]
            
            l_tf, l_pos_z = action_transform(l_ee_state)
            r_tf, r_pos_z = action_transform(r_ee_state)

            state_features = torch.cat([l_tf, r_tf, l_pos_z, r_pos_z, hand_state], dim=-1)
        else:
            l_ee_state = action[:, :, :self.ee_dim] # bs, t, 7 
            hand_state = action[:, :, self.ee_dim:]
            
            l_tf, l_pos_z = action_transform(l_ee_state)
            state_features = torch.cat([l_tf, l_pos_z, hand_state], dim=-1)
        return state_features
    
    
    def getActionOutput(self, pred):
        def getActionRelOutputEachHand(ee_pred):
            ee_rho2 = ee_pred[:, :, 0:2] # (bs, t, 2)
            pos_xy = ee_pred[:, :, 2:4]
            ee_rho12 = ee_pred[:, :, 4:6]
            ee_rho11 = ee_pred[:, :, 6:8]
            ee_rho01 = ee_pred[:, :, 8:9]
            ee_rho02 = ee_pred[:, :, 9:10]
            ee_rho03 = ee_pred[:, :, 10:11]
            rho = torch.cat([ee_rho01, ee_rho02, ee_rho03, ee_rho11, ee_rho12, ee_rho2], dim=-1)
            m_rh0 = torch.matmul(self.p_inv.to(ee_pred.device), rho.transpose(-1, -2)).transpose(-1, -2)
            quat = self.getQuaternionFromMatrix(m_rh0)
            return pos_xy, quat
        
        def getActionOutputEachHand(ee_pred):
            ee_xy = ee_pred[:, :, 0:2] # (bs, t, 2)
            ee_cos1 = ee_pred[:, :, 2:3]
            ee_sin1 = ee_pred[:, :, 3:4]
            ee_cos2 = ee_pred[:, :, 4:5]
            ee_sin2 = ee_pred[:, :, 5:6]
            ee_cos3 = ee_pred[:, :, 6:7]
            ee_sin3 = ee_pred[:, :, 7:8]

            rot_6d = torch.cat([ee_cos1, ee_cos2, ee_cos3, ee_sin1, ee_sin2, ee_sin3], dim=-1)
            quat = self.getQuaternionFrom6D(rot_6d)
            return ee_xy, quat
        real_dim_wo_z = self.real_dim - 1
        action_transform = getActionRelOutputEachHand if self.rel_action else getActionOutputEachHand
        if self.num_hand == 2:
            l_xy, l_quat = action_transform(pred[:, :, :real_dim_wo_z]) # bs, t, 8
            r_xy, r_quat = action_transform(pred[:, :, real_dim_wo_z:real_dim_wo_z*2]) # bs, t, 8

            l_z, r_z = pred[:, :, real_dim_wo_z*2:real_dim_wo_z*2+1], pred[:, :, real_dim_wo_z*2+1:real_dim_wo_z*2+2] # bs, t, 1
            hand_state = pred[:, :, real_dim_wo_z*2+2:] # bs, t, rest
            
            action_output = torch.cat(
                [l_xy, l_z, l_quat, r_xy, r_z, r_quat, hand_state],
                dim=-1
            )
        else:
            l_xy, l_quat = action_transform(pred[:, :, :real_dim_wo_z]) # bs, t, 8
            
            l_z = pred[:, :, real_dim_wo_z:real_dim_wo_z+1] # bs, t, 1
            hand_state = pred[:, :, real_dim_wo_z+1:] # bs, t, rest
            
            action_output = torch.cat(
                [l_xy, l_z, l_quat, hand_state],
                dim=-1
            )
            
        return action_output
    
    def getMatrixRotation(self, quat):
        if quat.shape[-1] == 4:
            return self.quaternion_to_matrix.forward(quat[:, :, [3, 0, 1, 2]]) 
        else:
            return self.axisangle_to_matrix.forward(quat)
    
    def getQuaternionFromMatrix(self, matrix):
        if self.ee_dim == 7:
            quat = self.quaternion_to_matrix.inverse(matrix)
            return quat[:, :, [1, 2, 3, 0]]  # xyzw
        else:
            return self.axisangle_to_matrix.inverse(matrix)

    def get6DRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        if quat.shape[-1] == 4:
            return self.quaternion_to_sixd.forward(quat[:, :, [3, 0, 1, 2]]) 
        else:
            return self.axisangle_to_sixd.forward(quat)
    
    def getQuaternionFrom6D(self, rot_6d):
        if self.ee_dim == 7:
            quat = self.quaternion_to_sixd.inverse(rot_6d)
            return quat[:, :, [1, 2, 3, 0]]  # xyzw
        else:
            return self.axisangle_to_sixd.inverse(rot_6d)

    def transform_action_mask(self, action_mask: torch.Tensor, velocity_dim: int) -> torch.Tensor:
        """
        Transform action_mask from original (quat/axisangle) space to velocity (rot6d) space.

        getActionGT reorganises each hand's EE dims:
          original per hand : [x, y, z, rot...]  = ee_dim  (7 quat | 6 axisangle)
          velocity per hand : [x, y, rot6d(6)]   = 8 dims  (placed first for all hands)
                            + [z]                = 1 dim   (placed after all hands' 8-d blocks)

        Total real dims in velocity space = num_hand * 9 + hand_state_dims

        Mapping by input type:
          quat   (ee_dim=7): per-hand  7 -> 9, 2 hands: n+4  (4 = 2*(9-7))
          axisangle (ee_dim=6): per-hand  6 -> 9, 2 hands: n+6  (6 = 2*(9-6))

        Args:
            action_mask : [B, T, max_action_dim] — 1 = real feature dim, 0 = padding
            velocity_dim: last dim of the velocity tensor (= self.action_type.size)
        Returns:
            new_mask    : [B, T, velocity_dim]  — same dtype / device as action_mask
        """
        B, T, _ = action_mask.shape
        device = action_mask.device

        # Number of real feature dims in original space (uniform across B, T)
        n_real_dims = int(action_mask[0, 0].float().sum().item())

        # Hand-state (gripper, etc.) dims = original real dims minus all EE dims
        hand_state_dims = max(0, n_real_dims - self.ee_dim * self.num_hand)

        # In velocity space each hand contributes 8 (xy + rot6d) + 1 (z) = 9 real dims
        real_dims = 12 if self.rel_action else 9
        n_real_trans = self.num_hand * real_dims + hand_state_dims

        new_mask = torch.zeros(B, T, velocity_dim, device=device, dtype=action_mask.dtype)
        new_mask[:, :, :n_real_trans] = 1
        return new_mask

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
            if self.config.add_pos_embed:
                self.temporal_pos_embed.requires_grad_(False)
                self.temporal_pos_proj.requires_grad_(False)
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
            if 'state_encoder.layer' in key or 'action_encoder' in key or 'action_decoder' in key or "model" in key or "future_tokens_equi_proj" in key or "vl_equi_proj" in key or "vlln" in key or "vl_self_attention" in key or "language_proj" in key or "language_lift" in key or "equi_vis_proj" in key:
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
                    self.temporal_pos_embed.eval()
                    self.temporal_pos_proj.eval()
            if not self.tune_diffusion_model:
                print("not tune diffusion model")
                self.model.eval()

    def _add_temporal_pos_embed(self, features: torch.Tensor) -> torch.Tensor:
        """Add equivariant temporal position embeddings to a [B, T, D] feature tensor."""
        T = features.shape[1]
        device = features.device
        pos_ids = torch.arange(T, device=device)
        pos_emb = self.temporal_pos_proj(
            enn.GeometricTensor(self.temporal_pos_embed(pos_ids), self.temporal_pos_trivial_type)
        ).tensor  # [T, D]
        return features + pos_emb.unsqueeze(0)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """
        Process backbone features into DiT cross-attention context.

        Equivariant cameras (e.g. top-down):
          backbone_equi_vision_features [B, n_equi, T, D]
            → + 2-D pos embed
            → equi_vis_self_attn         equivariant self-attn → [B*n_equi, T, cross_attn_dim]
            → reshape                    [B, n_equi*T, cross_attn_dim]

        Non-equivariant cameras (e.g. wrist), optional:
          backbone_noequi_vision_features [B, n_noequi, T, D]
            → noequi_vis_proj            plain nn.Linear → [B, n_noequi*T, cross_attn_dim]

        Both concatenated → vl_features [B, (n_equi+n_noequi)*T, cross_attn_dim]
        """
        equi_vis = backbone_output.backbone_equi_vision_features
        if equi_vis.dim() == 3:
            # Backbone returns flat [B, n_equi*T_vis + T_lang, D] — use as-is
            B = equi_vis.shape[0]
            vl_features = equi_vis
            if "backbone_attention_mask" not in backbone_output:
                attn_mask = torch.ones(B, vl_features.shape[1], dtype=torch.long, device=equi_vis.device)
                backbone_output.data["backbone_attention_mask"] = attn_mask
        else:
            # Legacy 4D path: [B, n_equi, T, D]
            B, n_equi, T, D = equi_vis.shape
            equi_proj = equi_vis.reshape(B * n_equi, T, D)
            vl_features = equi_proj.reshape(B, n_equi * T, -1)
            attn_mask = torch.ones(B, vl_features.shape[1], dtype=torch.long, device=equi_vis.device)
            backbone_output.data["backbone_attention_mask"] = attn_mask

        backbone_output.data["vl_features"] = vl_features
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

        # Equi cross-attention context from process_backbone_output.
        vl_embs = backbone_output.vl_features                     # [B, n_equi*K, cross_attn_dim]
        encoder_mask = backbone_output.backbone_attention_mask    # [B, n_equi*K], all-ones

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
        
        noisy_trajectory = enn.GeometricTensor(noisy_trajectory, self.action_type)

        action_encoder_embodiment_id = embodiment_id.repeat((actions_gt.shape[1]))
        action_features = self.action_encoder(noisy_trajectory, t_discretized, action_encoder_embodiment_id)
        action_features = action_features.tensor
        action_features = einops.rearrange(
            action_features,
            '(b t) c -> b t c',
            b=actions_gt.shape[0],
            t=actions_gt.shape[1]
        )

        if self.config.add_pos_embed:
            state_features = self._add_temporal_pos_embed(state_features)
            action_features = self._add_temporal_pos_embed(action_features)

        # State + action only — vision is cross-attention context, not a prefix
        sa_embs = torch.cat((state_features, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_mask,
            timestep=t_discretized,
            return_all_hidden_states=False
        )

        N_sa = model_output.shape[1]

        action_decoder_embodiment_id = embodiment_id.repeat(N_sa)

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
            t=N_sa,
        )

        pred_actions = pred[:, -actions_gt.shape[1] :, :]

        # pred_actions = self.getActionOutput(pred_actions)
        # Slice out only the action portion of pred and target.
        with torch.no_grad():
            velocity_dim = velocity.shape[-1]
            # Transform mask from original rotation space (quat/axisangle) to rot6d velocity
            # space, accounting for the dimension expansion done by getActionGT:
            #   quat (7) -> 9 per hand (+2), axisangle (6) -> 9 per hand (+3)
            action_mask = self.transform_action_mask(
                action_input.action_mask.float(), velocity_dim
            ).to(velocity.device)
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

        # Equi cross-attention context from process_backbone_output.
        vl_embs = backbone_output.vl_features                     # [B, n_equi*K, cross_attn_dim]
        encoder_mask = backbone_output.backbone_attention_mask    # [B, n_equi*K], all-ones

        embodiment_id = action_input.embodiment_id

        # Embed state.
        B, T, _ = action_input.state.shape
        state_input = self.getJointGeometricTensor(action_input.state, is_action=False)
        state_features = self.state_encoder(state_input, embodiment_id)
        state_features = state_features.tensor
        state_features = einops.rearrange(state_features, '(b t) c -> b t c', b=B, t=T)

        if self.config.add_pos_embed:
            state_features = self._add_temporal_pos_embed(state_features)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
                    size=(batch_size, self.config.action_horizon, self.action_type.size),
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
            noisy_trajectory = enn.GeometricTensor(noisy_trajectory, self.action_type)
            action_features = self.action_encoder(noisy_trajectory, timesteps_tensor, action_encoder_embodiment_id)
            action_features = action_features.tensor
            action_features = einops.rearrange(
                action_features,
                '(b t) c -> b t c',
                b=actions.shape[0],
                t=actions.shape[1]
            )

            if self.config.add_pos_embed:
                action_features = self._add_temporal_pos_embed(action_features)

            # State + action only — vision is cross-attention context, not a prefix
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_mask,
                timestep=timesteps_tensor,
            )

            N_sa = model_output.shape[1]

            action_decoder_embodiment_id = embodiment_id.repeat(N_sa)

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
                t=N_sa,
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

        # Two-stream context from process_backbone_output.
        equi_vis_embeds = backbone_output.get("equi_vis_features")  # [B, n_img*T_vis, D_cross] or None
        vl_embeds       = backbone_output.vl_features               # [B, T_text, vl_cross_dim]
        encoder_mask_rt = backbone_output.backbone_attention_mask   # [B, T_text]
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

                if equi_vis_embeds is not None:
                    N_vis_rt = equi_vis_embeds.shape[1]
                    sa_embs = torch.cat((equi_vis_embeds, state_features, action_features), dim=1)
                else:
                    N_vis_rt = 0
                    sa_embs = torch.cat((state_features, action_features), dim=1)

                # Run model forward.
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    encoder_attention_mask=encoder_mask_rt,
                    timestep=timesteps_tensor,
                )
                model_output = model_output[:, N_vis_rt:, :]  # strip vis prefix tokens
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
