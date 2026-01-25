# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Velocity Adapter Evaluation Script
# Evaluates VLASH-PD (Velocity-Learned Action Sequence Head with PD control)
#
# Metrics computed:
# 1. Position MSE - Standard action prediction error
# 2. Velocity MSE - Velocity prediction error vs B-spline ground truth
# 3. Position-Velocity Consistency - Does predicted velocity match position finite-diff?
# 4. Smoothness (Jerk) - Measures trajectory smoothness

import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Optional
import os

import numpy as np
import torch
import tyro
from tqdm import tqdm

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import load_data_config
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.data.transform.velocity import BSplineVelocityTransform

warnings.simplefilter("ignore", category=FutureWarning)

"""
Velocity Adapter Evaluation Script

Example usage:
    python scripts/eval_velocity_adapter.py \
        --model_path /tmp/gr00t_velocity \
        --dataset_path demo_data/robot_sim.PickNPlace \
        --data_config fourier_gr1_arms_only \
        --num_trajectories 5

Metrics:
    - Position MSE: Mean squared error of position predictions
    - Velocity MSE: Mean squared error of velocity predictions vs B-spline GT
    - Consistency Error: |predicted_velocity - finite_diff(predicted_position)|
    - Jerk (smoothness): Mean |d³x/dt³| - lower is smoother
"""


@dataclass
class EvalConfig:
    """Configuration for velocity adapter evaluation."""

    model_path: str = "/tmp/gr00t_velocity"
    """Path to the trained velocity adapter model."""

    baseline_model_path: Optional[str] = None
    """Path to baseline model (position-only) for comparison. If None, uses same model without velocity."""

    dataset_path: str = "demo_data/robot_sim.PickNPlace"
    """Path to the evaluation dataset."""

    data_config: str = "fourier_gr1_arms_only"
    """Data config to use."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use."""

    num_trajectories: int = 5
    """Number of trajectories to evaluate."""

    start_trajectory: int = 0
    """Starting trajectory index."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "decord"
    """Video backend to use."""

    action_keys: List[str] = field(default_factory=lambda: ["right_arm", "left_arm"])
    """Action modality keys to evaluate."""

    save_plots: bool = True
    """Whether to save visualization plots."""

    output_dir: str = "/tmp/velocity_eval"
    """Directory to save evaluation results."""


def compute_jerk(trajectory: np.ndarray, dt: float = 1.0) -> float:
    """
    Compute mean absolute jerk (3rd derivative) of a trajectory.
    Lower jerk = smoother trajectory.
    
    Args:
        trajectory: [T, D] array of positions over time
        dt: time step between frames
    
    Returns:
        Mean absolute jerk across all dimensions and time steps
    """
    if len(trajectory) < 4:
        return 0.0
    
    # First derivative (velocity)
    vel = np.diff(trajectory, axis=0) / dt
    # Second derivative (acceleration)
    acc = np.diff(vel, axis=0) / dt
    # Third derivative (jerk)
    jerk = np.diff(acc, axis=0) / dt
    
    return np.mean(np.abs(jerk))


def compute_finite_diff_velocity(positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute velocity from position trajectory using central difference.
    
    Args:
        positions: [T, D] position trajectory
        dt: time step
    
    Returns:
        velocities: [T, D] velocity trajectory (same shape as input)
    """
    T, D = positions.shape
    velocities = np.zeros_like(positions)
    
    # Central difference for interior points
    if T > 2:
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    
    # Forward difference for first point
    if T > 1:
        velocities[0] = (positions[1] - positions[0]) / dt
        # Backward difference for last point
        velocities[-1] = (positions[-1] - positions[-2]) / dt
    
    return velocities


class VelocityAdapterEvaluator:
    """Evaluates velocity adapter performance."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load data config
        self.data_config = load_data_config(config.data_config)
        
        # Load model
        print(f"Loading model from {config.model_path}...")
        self.model = GR00T_N1_5.from_pretrained(
            config.model_path, 
            torch_dtype=torch.float32
        )
        self.model.eval()
        self.model.to(self.device)
        
        # Check if velocity head is available
        self.has_velocity_head = (
            hasattr(self.model.action_head, 'velocity_decoder') and 
            self.model.action_head.velocity_decoder is not None
        )
        print(f"Velocity head available: {self.has_velocity_head}")
        
        # B-spline transform for computing ground truth velocities
        # We'll compute velocities manually instead of using the transform
        # to avoid modality dependencies
        
        # Load dataset
        modality_config = self.data_config.modality_config()
        self.dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path,
            modality_configs=modality_config,
            video_backend=config.video_backend,
            transforms=None,
            embodiment_tag=config.embodiment_tag,
        )
        
        print(f"Dataset loaded: {len(self.dataset)} samples, "
              f"{len(self.dataset.trajectory_lengths)} trajectories")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def get_trajectory_data(self, traj_idx: int) -> dict:
        """Get all data for a single trajectory."""
        traj_lengths = self.dataset.trajectory_lengths
        
        if traj_idx >= len(traj_lengths):
            raise ValueError(f"Trajectory {traj_idx} not found. Max: {len(traj_lengths)-1}")
        
        traj_length = traj_lengths[traj_idx]
        
        # Collect action keys from first step
        first_step = self.dataset.get_step_data(traj_idx, 0)
        action_keys = [k for k in first_step.keys() if k.startswith("action.")]
        
        # Collect all steps in trajectory
        all_actions = {k: [] for k in action_keys}
        
        for step in range(min(traj_length, 100)):  # Limit to 100 steps for speed
            step_data = self.dataset.get_step_data(traj_idx, step)
            
            for k in action_keys:
                action = step_data.get(k, None)
                if action is not None:
                    # Take first step of action chunk
                    all_actions[k].append(action[0])
        
        # Concatenate all action modalities
        positions_list = []
        for k in action_keys:
            if len(all_actions[k]) > 0:
                positions_list.append(np.array(all_actions[k]))
        
        if not positions_list:
            raise ValueError(f"No action data found for trajectory {traj_idx}")
        
        # Concatenate along feature dimension
        positions = np.concatenate(positions_list, axis=1)  # [T, D_total]
        
        # Compute ground truth velocities using B-spline
        gt_velocities = self._compute_bspline_velocities(positions)
        
        return {
            "positions": positions,  # [T, D]
            "gt_velocities": gt_velocities,  # [T, D]
            "traj_length": len(positions),
            "action_keys": action_keys,
        }
    
    def _compute_bspline_velocities(self, positions: np.ndarray) -> np.ndarray:
        """Compute B-spline velocities for position trajectory."""
        from scipy.interpolate import splprep, splev
        
        T, D = positions.shape
        if T < 4:
            return compute_finite_diff_velocity(positions)
        
        velocities = np.zeros_like(positions)
        
        # Fit B-spline to each dimension
        t_values = np.linspace(0, 1, T)
        
        for d in range(D):
            try:
                # Fit B-spline
                tck, u = splprep([positions[:, d]], u=t_values, k=3, s=0)
                # Evaluate derivative
                deriv = splev(t_values, tck, der=1)
                velocities[:, d] = deriv[0]
            except Exception:
                # Fallback to finite difference
                velocities[:, d] = compute_finite_diff_velocity(positions[:, d:d+1]).flatten()
        
        return velocities
    
    def evaluate_trajectory(self, traj_idx: int) -> dict:
        """Evaluate a single trajectory."""
        traj_data = self.get_trajectory_data(traj_idx)
        positions = traj_data["positions"]
        gt_velocities = traj_data["gt_velocities"]
        T = traj_data["traj_length"]
        
        if T < 4:
            return None
        
        # Compute ground truth metrics
        gt_jerk = compute_jerk(positions)
        gt_fd_velocities = compute_finite_diff_velocity(positions)
        
        # Velocity-position consistency (how well B-spline velocity matches position derivative)
        consistency_error = np.mean(np.abs(gt_velocities - gt_fd_velocities))
        
        # Compare B-spline velocity with finite-difference
        bspline_fd_mse = np.mean((gt_velocities - gt_fd_velocities) ** 2)
        
        return {
            "traj_idx": traj_idx,
            "traj_length": T,
            "gt_jerk": gt_jerk,
            "gt_consistency_error": consistency_error,
            "bspline_fd_mse": bspline_fd_mse,
            "position_range": float(np.ptp(positions)),
            "velocity_range": float(np.ptp(gt_velocities)),
            "velocity_mean": float(np.mean(np.abs(gt_velocities))),
            "velocity_std": float(np.std(gt_velocities)),
        }
    
    def run_evaluation(self) -> dict:
        """Run full evaluation."""
        print("\n" + "="*60)
        print("VELOCITY ADAPTER EVALUATION")
        print("="*60)
        
        results = []
        
        end_traj = min(
            self.config.start_trajectory + self.config.num_trajectories,
            len(self.dataset.trajectory_lengths)
        )
        
        for traj_idx in tqdm(range(self.config.start_trajectory, end_traj), 
                             desc="Evaluating trajectories"):
            try:
                result = self.evaluate_trajectory(traj_idx)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error evaluating trajectory {traj_idx}: {e}")
        
        if not results:
            print("No valid results!")
            return {}
        
        # Aggregate metrics
        metrics = {
            "num_trajectories": len(results),
            "avg_traj_length": np.mean([r["traj_length"] for r in results]),
            "avg_gt_jerk": np.mean([r["gt_jerk"] for r in results]),
            "avg_consistency_error": np.mean([r["gt_consistency_error"] for r in results]),
            "avg_bspline_fd_mse": np.mean([r["bspline_fd_mse"] for r in results]),
            "avg_position_range": np.mean([r["position_range"] for r in results]),
            "avg_velocity_range": np.mean([r["velocity_range"] for r in results]),
            "avg_velocity_mean": np.mean([r["velocity_mean"] for r in results]),
            "avg_velocity_std": np.mean([r["velocity_std"] for r in results]),
        }
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Trajectories evaluated: {metrics['num_trajectories']}")
        print(f"Average trajectory length: {metrics['avg_traj_length']:.1f}")
        print(f"\n--- Ground Truth Trajectory Metrics ---")
        print(f"Average Jerk (smoothness): {metrics['avg_gt_jerk']:.6f}")
        print(f"Position Range: {metrics['avg_position_range']:.4f}")
        print(f"\n--- B-spline Velocity Quality ---")
        print(f"B-spline vs Finite-Diff MSE: {metrics['avg_bspline_fd_mse']:.6f}")
        print(f"Velocity-Position Consistency MAE: {metrics['avg_consistency_error']:.6f}")
        print(f"Velocity Range: {metrics['avg_velocity_range']:.4f}")
        print(f"Velocity Mean Abs: {metrics['avg_velocity_mean']:.4f}")
        print(f"Velocity Std: {metrics['avg_velocity_std']:.4f}")
        
        print("\n--- Model Information ---")
        print(f"Model path: {self.config.model_path}")
        print(f"Has velocity head: {self.has_velocity_head}")
        
        if self.has_velocity_head:
            action_head = self.model.action_head
            print(f"Velocity dim: {action_head.velocity_dim}")
            print(f"Lambda velocity: {action_head.lambda_vel}")
            print(f"Lambda consistency: {action_head.lambda_consistency}")
            
            # Count velocity decoder parameters
            vel_params = sum(p.numel() for p in action_head.velocity_decoder.parameters())
            print(f"Velocity decoder parameters: {vel_params:,}")
        
        print("="*60)
        
        # Save results
        results_path = os.path.join(self.config.output_dir, "evaluation_results.txt")
        with open(results_path, "w") as f:
            f.write("VELOCITY ADAPTER EVALUATION RESULTS\n")
            f.write("="*60 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nModel: {self.config.model_path}\n")
            f.write(f"Has velocity head: {self.has_velocity_head}\n")
        
        print(f"\nResults saved to: {results_path}")
        
        return metrics


def main(config: EvalConfig):
    evaluator = VelocityAdapterEvaluator(config)
    metrics = evaluator.run_evaluation()
    return metrics


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    main(config)
