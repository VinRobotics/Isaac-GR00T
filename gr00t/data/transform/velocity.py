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

"""
B-spline based velocity computation transform for GR00T velocity adaptation.

This module provides transforms to compute synthetic velocity profiles from
position action chunks using B-spline differentiation. The computed velocities
serve as supervision targets for training a dual-head velocity adapter.
"""

from typing import Any

import numpy as np
from pydantic import Field
from scipy import interpolate

from .base import ModalityTransform


class BSplineVelocityTransform(ModalityTransform):
    """
    Compute velocity from position action chunks using cubic B-spline differentiation.
    
    This transform fits a cubic B-spline to the position trajectory and computes
    the analytical derivative to obtain smooth velocity profiles. The velocity
    is computed for each degree of freedom independently.
    
    Args:
        apply_to: List of action keys to compute velocity for (e.g., ["action"]).
        smoothing_factor: Smoothing factor for B-spline fitting. 
            - 0.0: Interpolating spline (passes through all points)
            - >0.0: Smoothing spline (trades accuracy for smoothness)
            Recommended: 0.0 for clean demos, 0.01-0.1 for noisy data.
        spline_degree: Degree of the B-spline (default 3 for cubic).
        dt: Time step between action waypoints (default 1.0, normalized).
        velocity_key_suffix: Suffix to append to create velocity key names.
    
    Example:
        >>> transform = BSplineVelocityTransform(
        ...     apply_to=["action"],
        ...     smoothing_factor=0.0,
        ... )
        >>> data = {"action": np.random.randn(16, 7)}  # [horizon, action_dim]
        >>> result = transform(data)
        >>> result["velocity"].shape  # (16, 7)
    """
    
    smoothing_factor: float = Field(
        default=0.0,
        description="Smoothing factor for B-spline. 0.0 = interpolating, >0 = smoothing.",
        ge=0.0,
    )
    spline_degree: int = Field(
        default=3,
        description="Degree of B-spline (3 = cubic).",
        ge=1,
        le=5,
    )
    dt: float = Field(
        default=1.0,
        description="Time step between waypoints (normalized).",
        gt=0.0,
    )
    velocity_key_suffix: str = Field(
        default="_velocity",
        description="Suffix to create velocity key from position key.",
    )
    output_velocity_key: str = Field(
        default="velocity",
        description="Key name for the output velocity tensor.",
    )
    
    def _compute_bspline_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute velocity from positions using B-spline differentiation.
        
        Args:
            positions: Position trajectory of shape [horizon, n_dofs].
            
        Returns:
            Velocity trajectory of shape [horizon, n_dofs].
        """
        horizon, n_dofs = positions.shape
        velocities = np.zeros_like(positions)
        
        # Time points for the trajectory
        t = np.arange(horizon) * self.dt
        
        for dof in range(n_dofs):
            pos_dof = positions[:, dof]
            
            # Handle edge case: if all positions are the same, velocity is zero
            if np.allclose(pos_dof, pos_dof[0]):
                velocities[:, dof] = 0.0
                continue
            
            # Fit B-spline to positions
            # splrep returns (t, c, k) - knots, coefficients, degree
            try:
                # For interpolating spline (s=0), we need enough points
                if horizon <= self.spline_degree:
                    # Fall back to finite difference for very short trajectories
                    velocities[:, dof] = np.gradient(pos_dof, self.dt)
                    continue
                
                tck = interpolate.splrep(
                    t, 
                    pos_dof, 
                    s=self.smoothing_factor,
                    k=min(self.spline_degree, horizon - 1),
                )
                
                # Evaluate derivative of spline at original time points
                velocities[:, dof] = interpolate.splev(t, tck, der=1)
                
            except Exception:
                # Fall back to finite difference if spline fitting fails
                velocities[:, dof] = np.gradient(pos_dof, self.dt)
        
        return velocities
    
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply B-spline velocity computation to action data.
        
        Args:
            data: Dictionary containing action data with keys matching `apply_to`.
            
        Returns:
            Data dictionary with added velocity key.
        """
        # Concatenate all action keys to get the full action tensor
        action_arrays = []
        for key in self.apply_to:
            if key in data:
                action_arrays.append(data[key])
        
        if not action_arrays:
            return data
        
        # Concatenate along action dimension
        positions = np.concatenate(action_arrays, axis=-1)  # [horizon, total_action_dim]
        
        # Compute velocity using B-spline
        velocities = self._compute_bspline_velocity(positions)
        
        # Add velocity to data
        data[self.output_velocity_key] = velocities.astype(np.float32)
        
        return data


class FiniteDifferenceVelocityTransform(ModalityTransform):
    """
    Compute velocity from position action chunks using finite difference.
    
    This is a simpler alternative to B-spline that uses numpy's gradient
    function with optional Gaussian smoothing.
    
    Args:
        apply_to: List of action keys to compute velocity for.
        dt: Time step between waypoints.
        smooth_sigma: Gaussian smoothing sigma. 0.0 = no smoothing.
        output_velocity_key: Key name for output velocity.
    """
    
    dt: float = Field(default=1.0, gt=0.0)
    smooth_sigma: float = Field(default=0.0, ge=0.0)
    output_velocity_key: str = Field(default="velocity")
    
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply finite difference velocity computation."""
        action_arrays = []
        for key in self.apply_to:
            if key in data:
                action_arrays.append(data[key])
        
        if not action_arrays:
            return data
        
        positions = np.concatenate(action_arrays, axis=-1)
        
        # Compute velocity using finite difference
        velocities = np.gradient(positions, self.dt, axis=0)
        
        # Optional Gaussian smoothing
        if self.smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            velocities = gaussian_filter1d(velocities, sigma=self.smooth_sigma, axis=0)
        
        data[self.output_velocity_key] = velocities.astype(np.float32)
        
        return data


class VelocityNormalizationTransform(ModalityTransform):
    """
    Normalize velocity using the same statistics as position actions.
    
    Since velocity is the derivative of position, we use the position
    statistics scaled by the time step for consistent normalization.
    
    Args:
        apply_to: Velocity key to normalize.
        position_stats_key: Key to look up position statistics from metadata.
        normalization_mode: "min_max" or "mean_std".
        dt: Time step used for velocity computation.
    """
    
    position_stats_key: str = Field(
        default="action",
        description="Key to look up position statistics.",
    )
    normalization_mode: str = Field(
        default="min_max",
        description="Normalization mode: 'min_max' or 'mean_std'.",
    )
    dt: float = Field(default=1.0, gt=0.0)
    
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply velocity normalization."""
        for key in self.apply_to:
            if key not in data:
                continue
            
            velocity = data[key]
            
            if self.normalization_mode == "min_max":
                # For velocity, we scale by the position range / dt
                # This is a heuristic that works well in practice
                # Velocity is normalized to roughly [-1, 1] similar to position
                stats = self.dataset_metadata.statistics
                if hasattr(stats, self.position_stats_key):
                    stat_entry = getattr(stats, self.position_stats_key)
                    pos_min = np.array(stat_entry.min)
                    pos_max = np.array(stat_entry.max)
                    pos_range = pos_max - pos_min
                    vel_scale = pos_range / self.dt
                    
                    # Normalize velocity to [-1, 1] using estimated velocity range
                    velocity = velocity / (vel_scale + 1e-8)
                    velocity = np.clip(velocity, -1.0, 1.0)
                    
            elif self.normalization_mode == "mean_std":
                # Compute running statistics from the velocity itself
                mean = np.mean(velocity, axis=0, keepdims=True)
                std = np.std(velocity, axis=0, keepdims=True) + 1e-8
                velocity = (velocity - mean) / std
            
            data[key] = velocity.astype(np.float32)
        
        return data
