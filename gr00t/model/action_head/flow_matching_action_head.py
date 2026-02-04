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

from .equivariant_cross_attention_dit import EDiT, EquivariantLayerNorm
from .cross_attention_dit import SelfAttentionTransformer

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
    
    # Separate vision and language feature dimensions
    vision_embedding_dim: int = field(
        default=1152, metadata={"help": "Vision embedding dimension from backbone (regular repr, equivariant)."}
    )
    language_embedding_dim: int = field(
        default=2048, metadata={"help": "Language embedding dimension from backbone (trivial repr, invariant)."}
    )
    
    rot_type: str = field(
        default="quaternion", metadata={"help": "Define rot type: quaternion, euler_angles"}
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
        self.n_group = config.n_group

        self.num_hand = self.config.num_hand

        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Inject n_group into diffusion model config
        diffusion_cfg = {**config.diffusion_model_cfg, "n_group": self.n_group}
        self.model = EDiT(**diffusion_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # equi state
        self.ee_dim = 7 if self.config.rot_type == "quaternion" else 6
        self.group = gspaces.no_base_space(CyclicGroup(self.n_group))
        self.state_in_type = self.getJointFieldType(is_action=False)
        self.state_hidden_type = enn.FieldType(self.group, int(config.hidden_size / self.n_group) * [self.group.regular_repr])
        self.state_out_type = enn.FieldType(self.group, int(config.input_embedding_dim / self.n_group) * [self.group.regular_repr])
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')
        self.eulerangle_to_sixd = RotationTransformer('euler_angles', 'rotation_6d', from_convention="ZYX")

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

        # Vision projection: projects vision features (regular repr) to match sa_embs dimension
        # Vision features are equivariant in regular repr from frame averaging
        self.vision_in_type = enn.FieldType(
            self.group, 
            int(config.vision_embedding_dim / self.n_group) * [self.group.regular_repr]
        )
        self.vision_projection = enn.Linear(self.vision_in_type, self.state_out_type)
        
        # Language projection: projects language features (trivial repr) for cross-attention
        # Language features are invariant (trivial repr) - used for cross-attention conditioning
        self.language_projection = nn.Linear(config.language_embedding_dim, config.diffusion_model_cfg.get("cross_attention_dim", 2048))

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
            self.num_hand * 4 * [self.group.irrep(1)] # pos xy, rot 6, left and right
            + (max_dim - ((self.ee_dim - 1) * self.num_hand)) * [self.group.trivial_repr], # gripper 1, z from both ee is 2
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
        def getActionGTEachHand(ee_state):
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
            l_ee_state = action[:, :, :self.ee_dim] # bs, t, 7 
            r_ee_state = action[:, :, self.ee_dim:self.ee_dim*2] # bs, t, 7  
            hand_state = action[:, :, self.ee_dim*2:]
            
            l_tf, l_pos_z = getActionGTEachHand(l_ee_state)
            r_tf, r_pos_z = getActionGTEachHand(r_ee_state)

            state_features = torch.cat([l_tf, r_tf, l_pos_z, r_pos_z, hand_state], dim=-1)
        else:
            l_ee_state = action[:, :, :self.ee_dim] # bs, t, 7 
            hand_state = action[:, :, self.ee_dim:]
            
            l_tf, l_pos_z = getActionGTEachHand(l_ee_state)
            state_features = torch.cat([l_tf, l_pos_z, hand_state], dim=-1)
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
        
        if self.num_hand == 2:
            l_xy, l_quat = getActionOutputEachHand(pred[:, :, :8]) # bs, t, 8
            r_xy, r_quat = getActionOutputEachHand(pred[:, :, 8:16]) # bs, t, 8
            
            l_z, r_z = pred[:, :, 16:17], pred[:, :, 17:18] # bs, t, 1
            hand_state = pred[:, :, 18:] # bs, t, rest
            
            action_output = torch.cat(
                [l_xy, l_z, l_quat, r_xy, r_z, r_quat, hand_state],
                dim=-1
            )
        else:
            l_xy, l_quat = getActionOutputEachHand(pred[:, :, :8]) # bs, t, 8
            
            l_z = pred[:, :, 8:9] # bs, t, 1
            hand_state = pred[:, :, 9:] # bs, t, rest
            
            action_output = torch.cat(
                [l_xy, l_z, l_quat, hand_state],
                dim=-1
            )
            
        return action_output

    def get6DRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        if quat.shape[-1] == 4:
            return self.quaternion_to_sixd.forward(quat[:, :, [3, 0, 1, 2]]) 
        else:
            return self.eulerangle_to_sixd.forward(quat)
    
    def getQuaternionFrom6D(self, rot_6d):
        if self.ee_dim == 7:
            quat = self.quaternion_to_sixd.inverse(rot_6d)
            return quat[:, :, [1, 2, 3, 0]]  # xyzw
        else:
            return self.eulerangle_to_sixd.inverse(rot_6d)




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
            if 'state_encoder.layer' in key or 'action_encoder' in key or 'action_decoder' in key or "model" in key or "future_tokens_equi_proj" in key or "vl_equi_proj" in key or "vlln" in key or "vl_self_attention" in key or "vision_projection" in key or "language_projection" in key:
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
        """
        Process backbone output to separate vision and language features.
        
        Vision features (regular repr, equivariant): Used in self-attention with state/action
        Language features (trivial repr, invariant): Used for cross-attention conditioning
        """
        # Get separated features from backbone
        vision_features = backbone_output["backbone_vision_features"]  # (B, num_imgs, vision_dim)
        language_features = backbone_output["backbone_language_features"]  # (B, T_lang, lang_dim)
        
        # Project vision features through equivariant linear (regular repr -> regular repr)
        B, num_imgs, _ = vision_features.shape
        vision_flat = einops.rearrange(vision_features, "b n d -> (b n) d")
        vision_geo = enn.GeometricTensor(vision_flat, self.vision_in_type)
        vision_projected = self.vision_projection(vision_geo)
        vision_features = einops.rearrange(vision_projected.tensor, "(b n) d -> b n d", b=B, n=num_imgs)
        
        # Project language features through standard linear (trivial repr for cross-attention)
        language_features = self.language_projection(language_features)
        
        backbone_output["vision_features"] = vision_features  # Equivariant, for self-attention
        backbone_output["language_features"] = language_features  # Invariant, for cross-attention
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

        # Get vision features (equivariant, regular repr) and language features (invariant, trivial repr)
        vision_features = backbone_output["vision_features"]  # (B, num_imgs, D) - for self-attention
        language_features = backbone_output["language_features"]  # (B, T_lang, D) - for cross-attention
        device = vision_features.device

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
        
        # Self-attention embeddings: vision (equivariant) + state (equivariant) + action (equivariant)
        # All in regular repr for equivariant self-attention
        sa_embs = torch.cat((vision_features, state_features, action_features), dim=1)
        B, T_sa, C = sa_embs.shape
        
        # Language features for cross-attention (invariant conditioning)
        lang_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=language_features,  # Language for cross-attention (trivial repr)
            encoder_attention_mask=lang_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        
        # Only decode the action portion (exclude vision and state tokens)
        num_vision_tokens = vision_features.shape[1]
        num_state_tokens = state_features.shape[1]
        action_output = model_output[:, num_vision_tokens + num_state_tokens:, :]  # Only action tokens
        
        action_decoder_embodiment_id = embodiment_id.repeat((action_output.shape[1]))

        action_output = einops.rearrange(
            action_output,
            'b t c -> (b t) c',
        )
        action_output = enn.GeometricTensor(action_output, self.state_hidden_type)
        pred = self.action_decoder(action_output, action_decoder_embodiment_id)

        pred_actions = einops.rearrange(
            pred.tensor,
            '(b t) c -> b t c',
            b=B,
            t=actions_gt.shape[1]
        )

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

        # Get vision features (equivariant, regular repr) and language features (invariant, trivial repr)
        vision_features = backbone_output["vision_features"]  # (B, num_imgs, D) - for self-attention
        language_features = backbone_output["language_features"]  # (B, T_lang, D) - for cross-attention
        
        embodiment_id = action_input.embodiment_id

        # Embed state.
        B, T, _ = action_input.state.shape
        state_input = self.getJointGeometricTensor(action_input.state, is_action=False)
        state_features = self.state_encoder(state_input, embodiment_id)
        state_features = state_features.tensor
        state_features = einops.rearrange(state_features, '(b t) c -> b t c', b=B, t=T)

        # Set initial actions as the sampled noise.
        batch_size = vision_features.shape[0]
        device = vision_features.device
        actions = torch.randn(
                    size=(batch_size, self.config.action_horizon, self.action_type.size),
                    dtype=vision_features.dtype,
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

            # Self-attention embeddings: vision (equivariant) + state (equivariant) + action (equivariant)
            sa_embs = torch.cat((vision_features, state_features, action_features), dim=1)

            # Run model forward with language for cross-attention
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=language_features,  # Language for cross-attention (trivial repr)
                timestep=timesteps_tensor,
            )
            
            # Only decode the action portion (exclude vision and state tokens)
            num_vision_tokens = vision_features.shape[1]
            num_state_tokens = state_features.shape[1]
            action_output = model_output[:, num_vision_tokens + num_state_tokens:, :]
            
            action_decoder_embodiment_id = embodiment_id.repeat((action_output.shape[1]))

            action_output = einops.rearrange(
                action_output,
                'b t c -> (b t) c',
            )
            action_output = enn.GeometricTensor(action_output, self.state_hidden_type)
            pred = self.action_decoder(action_output, action_decoder_embodiment_id)

            pred_velocity = einops.rearrange(
                pred.tensor,
                '(b t) c -> b t c',
                b=batch_size,
                t=self.action_horizon
            )

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

        # Get vision features (equivariant, regular repr) and language features (invariant, trivial repr)
        vision_features = backbone_output["vision_features"]  # (B, num_imgs, D) - for self-attention
        language_features = backbone_output["language_features"]  # (B, T_lang, D) - for cross-attention
        embodiment_id  = action_input.embodiment_id

        # Embed state.
        B, T_state, _ = action_input.state.shape
        state_input = self.getJointGeometricTensor(action_input.state, is_action=False)
        state_features = self.state_encoder(state_input, embodiment_id)
        state_features = state_features.tensor
        state_features = einops.rearrange(state_features, '(b t) c -> b t c', b=B, t=T_state)

        # Set initial actions as the sampled noise.
        batch_size = vision_features.shape[0]
        device = vision_features.device
        x_t = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_type.size),
            dtype=vision_features.dtype,
            device=device,
        )

        # Store token counts for decoding
        num_vision_tokens = vision_features.shape[1]
        num_state_tokens = state_features.shape[1]

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
                action_encoder_embodiment_id = embodiment_id.repeat((x_t_.shape[1]))
                noisy_trajectory = einops.rearrange(x_t_, "b t c -> (b t) c")
                noisy_trajectory = enn.GeometricTensor(noisy_trajectory, self.getJointFieldType(True))
                action_features = self.action_encoder(noisy_trajectory, timesteps_tensor, action_encoder_embodiment_id)
                action_features = action_features.tensor
                action_features = einops.rearrange(
                    action_features,
                    '(b t) c -> b t c',
                    b=x_t_.shape[0],
                    t=x_t_.shape[1]
                )

                # Self-attention embeddings: vision (equivariant) + state (equivariant) + action (equivariant)
                sa_embs = torch.cat((vision_features, state_features, action_features), dim=1)

                # Run model forward with language for cross-attention
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=language_features,  # Language for cross-attention (trivial repr)
                    timestep=timesteps_tensor,
                )
                
                # Only decode the action portion (exclude vision and state tokens)
                action_output = model_output[:, num_vision_tokens + num_state_tokens:, :]
                
                action_decoder_embodiment_id = embodiment_id.repeat((action_output.shape[1]))
                action_output = einops.rearrange(action_output, 'b t c -> (b t) c')
                action_output = enn.GeometricTensor(action_output, self.state_hidden_type)
                pred = self.action_decoder(action_output, action_decoder_embodiment_id)

                pred_velocity = einops.rearrange(
                    pred.tensor,
                    '(b t) c -> b t c',
                    b=batch_size,
                    t=self.action_horizon
                )
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

        x_t = self.getActionOutput(x_t)
        x_t = x_t.clone().detach()
        return BatchFeature(data={"action_pred": x_t})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
