import copy
from typing import Any, Dict

import numpy as np
import torch
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.policy import Gr00tPolicy, squeeze_dict_values, unsqueeze_dict_values
from transformers.feature_extraction_utils import BatchFeature

COMPUTE_DTYPE = torch.bfloat16


class Gr00tPolicy_CFG(Gr00tPolicy):

    def get_action(
        self, observations: Dict[str, Any],
        scale: float = 1.3,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        # let the get_action handles both batch and single input
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        # prepare uncond
        observations_uncond = copy.deepcopy(obs_copy)
        observations_uncond["annotation.human.task_description"] = " "

        # Apply transforms
        normalized_input = self.apply_transforms(obs_copy)
        normalized_input_uncond = self.apply_transforms(observations_uncond)

        normalized_action = self._get_action_from_normalized_input(
            normalized_input, normalized_input_uncond,
            scale=scale,
        )
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if self.smooth_option == "te":
            for k in unnormalized_action.keys():
                unnormalized_action[k] = unnormalized_action[k].squeeze(1)  # remove the 1 dimension
        elif not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def _get_action_from_normalized_input(
        self, normalized_input: Dict[str, Any], normalized_input_uncond: Dict[str, Any],
        scale: float = 1.0,
    ) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(
                normalized_input, normalized_input_uncond,
                scale=scale,
            )

        normalized_action = model_pred["action_pred"].float()
        if self.smooth_option == "te":
            normalized_action = self.process_output(normalized_action)
        return normalized_action


class GR00T_N1_CFG(GR00T_N1_5):

    def get_action(
        self,
        inputs: dict,
        inputs_uncond: dict,
        scale: float = 1.0,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_inputs_uncond, _ = self.prepare_input(inputs_uncond)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        backbone_outputs_uncond = self.backbone(backbone_inputs_uncond)
        action_head_outputs = self.action_head.get_action(
            backbone_outputs, action_inputs,
            backbone_outputs_uncond,
            scale=scale,
        )
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs


class FlowmatchingActionHead_CFG(FlowmatchingActionHead):

    @torch.no_grad()
    def get_action(
        self, backbone_output: BatchFeature, action_input: BatchFeature,
        backbone_output_uncond: BatchFeature,
        scale: float,
    ) -> BatchFeature:
        
        backbone_output = self.process_backbone_output(backbone_output)
        backbone_output_uncond = self.process_backbone_output(backbone_output_uncond)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        vl_embs_uncond = backbone_output_uncond.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)  # (B, 1, 1536)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )  # (B, T=16, D=32)
        
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
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            if scale != 1.0:
                model_output_uncond = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs_uncond,
                    timestep=timesteps_tensor,
                )
                pred_uncond = self.action_decoder(model_output_uncond, embodiment_id)
                pred = pred + (scale - 1) * (pred - pred_uncond)

            pred_velocity = pred[:, -self.action_horizon:]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity

        return BatchFeature(data={"action_pred": actions})
