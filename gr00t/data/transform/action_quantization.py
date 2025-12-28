
"""
Action Quantization Transform for VLASH-style action discretization.

This module implements action quantization as described in the VLASH paper,
which discretizes continuous actions into bins to reduce the action space complexity
and improve training stability for vision-language-action models.

The implementation supports:
1. Position-level quantization (0th order)
2. Velocity-level quantization (1st order derivative)
3. Acceleration-level quantization (2nd order derivative)
4. Higher-order derivative quantization (generalized)
"""

from typing import Any, Dict, List, Optional, Literal
import numpy as np
import torch
from pydantic import Field, PrivateAttr

from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform


class ActionQuantizer:
    """
    Quantizer for discretizing continuous action values into bins.
    
    Supports multiple quantization schemes:
    - uniform: Equal-width bins across the range
    - adaptive: Bins based on data distribution (using percentiles)
    - kmeans: Cluster-based bins (requires fitting)
    """
    
    def __init__(
        self,
        num_bins: int,
        mode: Literal["uniform", "adaptive"] = "uniform",
        range_min: float = -1.0,
        range_max: float = 1.0,
        percentiles: Optional[List[float]] = None,
    ):
        self.num_bins = num_bins
        self.mode = mode
        self.range_min = range_min
        self.range_max = range_max
        
        if mode == "uniform":
            # Create uniform bin edges
            self.bin_edges = torch.linspace(range_min, range_max, num_bins + 1)
            self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        elif mode == "adaptive" and percentiles is not None:
            # Create adaptive bins based on percentiles
            self.bin_edges = torch.tensor(percentiles, dtype=torch.float32)
            self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        else:
            # Default to uniform
            self.bin_edges = torch.linspace(range_min, range_max, num_bins + 1)
            self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous values to bin indices.
        
        Args:
            x: Continuous values tensor of any shape
            
        Returns:
            Tensor of same shape with quantized values (bin centers)
        """
        device = x.device
        dtype = x.dtype
        bin_edges = self.bin_edges.to(device=device, dtype=dtype)
        bin_centers = self.bin_centers.to(device=device, dtype=dtype)
        
        # Clamp to valid range
        x_clamped = torch.clamp(x, self.range_min, self.range_max)
        
        # Find bin indices using searchsorted
        # searchsorted returns index where value should be inserted to maintain sorted order
        bin_indices = torch.searchsorted(bin_edges[1:-1], x_clamped.contiguous())
        
        # Clamp indices to valid range [0, num_bins - 1]
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        # Return bin centers for the corresponding indices
        return bin_centers[bin_indices]
    
    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dequantize is identity for center-based quantization.
        The quantized values are already the bin centers.
        """
        return x
    
    def get_bin_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get bin indices for continuous values (useful for discrete action heads).
        
        Args:
            x: Continuous values tensor
            
        Returns:
            Tensor of bin indices (long type)
        """
        device = x.device
        dtype = x.dtype
        bin_edges = self.bin_edges.to(device=device, dtype=dtype)
        
        x_clamped = torch.clamp(x, self.range_min, self.range_max)
        bin_indices = torch.searchsorted(bin_edges[1:-1], x_clamped.contiguous())
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        return bin_indices.long()


class ActionQuantizationTransform(InvertibleModalityTransform):
    """
    VLASH-style action quantization transform.
    
    This transform discretizes continuous actions into bins, which can improve
    training stability and reduce the effective action space dimensionality.
    
    Supports quantization at different derivative levels:
    - 0: Position (raw action values)
    - 1: Velocity (first derivative)
    - 2: Acceleration (second derivative)
    
    Args:
        apply_to: List of action keys to quantize
        num_bins: Number of quantization bins per dimension
        quantization_mode: "uniform" or "adaptive"
        derivative_order: Order of derivative to quantize (0=position, 1=velocity, 2=acceleration)
        compute_derivatives: Whether to compute derivatives from position data
        dt: Time step for derivative computation (default: 1.0 for normalized time)
        range_min: Minimum value for quantization range (after normalization)
        range_max: Maximum value for quantization range (after normalization)
    """
    
    num_bins: int = Field(default=256, description="Number of quantization bins per dimension")
    quantization_mode: Literal["uniform", "adaptive"] = Field(
        default="uniform", description="Quantization mode"
    )
    derivative_order: int = Field(
        default=0, description="Order of derivative to quantize (0=position, 1=velocity, 2=accel)"
    )
    compute_derivatives: bool = Field(
        default=False, description="Whether to compute derivatives from position"
    )
    dt: float = Field(default=1.0, description="Time step for derivative computation")
    range_min: float = Field(default=-1.0, description="Minimum quantization range")
    range_max: float = Field(default=1.0, description="Maximum quantization range")
    
    # Private attributes
    _quantizers: Dict[str, ActionQuantizer] = PrivateAttr(default_factory=dict)
    _statistics: Dict[str, Dict] = PrivateAttr(default_factory=dict)
    
    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Initialize quantizers with dataset statistics."""
        super().set_metadata(dataset_metadata)
        
        for key in self.apply_to:
            # Get statistics for adaptive quantization if needed
            if self.quantization_mode == "adaptive":
                modality, action_key = key.split(".")
                if hasattr(dataset_metadata.statistics, modality):
                    stats = getattr(dataset_metadata.statistics, modality).get(action_key)
                    if stats and hasattr(stats, "percentiles"):
                        percentiles = stats.percentiles
                        self._statistics[key] = {"percentiles": percentiles}
                        self._quantizers[key] = ActionQuantizer(
                            num_bins=self.num_bins,
                            mode="adaptive",
                            percentiles=percentiles,
                        )
                        continue
            
            # Default to uniform quantization
            self._quantizers[key] = ActionQuantizer(
                num_bins=self.num_bins,
                mode="uniform",
                range_min=self.range_min,
                range_max=self.range_max,
            )
    
    def _compute_derivative(self, x: torch.Tensor, order: int) -> torch.Tensor:
        """
        Compute the nth order derivative of the action sequence.
        
        Args:
            x: Action tensor of shape (T, D) or (B, T, D)
            order: Derivative order (1=velocity, 2=acceleration, etc.)
            
        Returns:
            Derivative tensor of same shape (with edge values repeated)
        """
        if order == 0:
            return x
        
        # Determine if batched
        if x.dim() == 2:
            # (T, D) -> add batch dim
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        result = x
        for _ in range(order):
            # Compute finite difference along time dimension
            # diff[t] = (x[t+1] - x[t]) / dt
            diff = (result[:, 1:, :] - result[:, :-1, :]) / self.dt
            
            # Pad to maintain shape (repeat first value)
            first_val = diff[:, :1, :]
            result = torch.cat([first_val, diff], dim=1)
        
        if squeeze_batch:
            result = result.squeeze(0)
        
        return result
    
    def _integrate_derivative(self, dx: torch.Tensor, x0: torch.Tensor, order: int) -> torch.Tensor:
        """
        Integrate derivative back to position.
        
        Args:
            dx: Derivative tensor
            x0: Initial condition(s)
            order: Derivative order to integrate
            
        Returns:
            Integrated position tensor
        """
        if order == 0:
            return dx
        
        if dx.dim() == 2:
            dx = dx.unsqueeze(0)
            x0 = x0.unsqueeze(0) if x0.dim() == 1 else x0.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        result = dx
        for _ in range(order):
            # Cumulative sum integration: x[t] = x[0] + sum(dx[0:t]) * dt
            integrated = torch.cumsum(result, dim=1) * self.dt
            # Add initial condition
            if x0 is not None:
                integrated = integrated + x0[:, :1, :]
            result = integrated
        
        if squeeze_batch:
            result = result.squeeze(0)
        
        return result
    
    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply action quantization transform."""
        for key in self.apply_to:
            if key not in data:
                continue
            
            action = data[key]
            assert isinstance(action, torch.Tensor), f"Expected torch.Tensor, got {type(action)}"
            
            # Store original for potential derivative computation
            original_action = action.clone()
            
            # Compute derivative if requested
            if self.compute_derivatives and self.derivative_order > 0:
                action = self._compute_derivative(action, self.derivative_order)
                # Store original for inverse transform
                data[f"_original_{key}"] = original_action
            
            # Apply quantization
            if key in self._quantizers:
                action = self._quantizers[key].quantize(action)
            else:
                # Fallback: create default quantizer
                quantizer = ActionQuantizer(
                    num_bins=self.num_bins,
                    mode="uniform",
                    range_min=self.range_min,
                    range_max=self.range_max,
                )
                action = quantizer.quantize(action)
            
            data[key] = action
        
        return data
    
    def unapply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Reverse the quantization transform."""
        for key in self.apply_to:
            if key not in data:
                continue
            
            action = data[key]
            
            # Dequantization (identity for center-based)
            if key in self._quantizers:
                action = self._quantizers[key].dequantize(action)
            
            # Integrate back if derivatives were computed
            if self.compute_derivatives and self.derivative_order > 0:
                original_key = f"_original_{key}"
                if original_key in data:
                    x0 = data[original_key][:1] if data[original_key].dim() == 2 else data[original_key][:, :1, :]
                    action = self._integrate_derivative(action, x0, self.derivative_order)
                    # Clean up temporary key
                    del data[original_key]
            
            data[key] = action
        
        return data


class VelocityQuantizationTransform(ActionQuantizationTransform):
    """
    Convenience class for velocity-level action quantization.
    
    This is equivalent to ActionQuantizationTransform with derivative_order=1.
    """
    
    derivative_order: int = Field(default=1, frozen=True)
    compute_derivatives: bool = Field(default=True, frozen=True)


class AccelerationQuantizationTransform(ActionQuantizationTransform):
    """
    Convenience class for acceleration-level action quantization.
    
    This is equivalent to ActionQuantizationTransform with derivative_order=2.
    """
    
    derivative_order: int = Field(default=2, frozen=True)
    compute_derivatives: bool = Field(default=True, frozen=True)


class HigherOrderDerivativeTransform(InvertibleModalityTransform):
    """
    Transform that computes and stores higher-order derivatives of actions.
    
    This is useful when we want to include velocity/acceleration as separate
    state features without quantizing them.
    
    Args:
        apply_to: List of action/state keys to compute derivatives for
        derivative_orders: List of derivative orders to compute (e.g., [1, 2] for velocity and acceleration)
        dt: Time step for derivative computation
        output_suffix: Suffix template for output keys (e.g., "_vel", "_acc")
    """
    
    derivative_orders: List[int] = Field(
        default=[1], description="Derivative orders to compute"
    )
    dt: float = Field(default=1.0, description="Time step for derivative computation")
    output_suffixes: Dict[int, str] = Field(
        default_factory=lambda: {1: "_velocity", 2: "_acceleration", 3: "_jerk"},
        description="Suffixes for output keys by derivative order"
    )
    
    def _compute_derivative(self, x: torch.Tensor, order: int) -> torch.Tensor:
        """Compute nth order derivative."""
        if order == 0:
            return x
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        result = x
        for _ in range(order):
            diff = (result[:, 1:, :] - result[:, :-1, :]) / self.dt
            first_val = diff[:, :1, :]
            result = torch.cat([first_val, diff], dim=1)
        
        if squeeze_batch:
            result = result.squeeze(0)
        
        return result
    
    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute and store derivatives."""
        for key in self.apply_to:
            if key not in data:
                continue
            
            value = data[key]
            assert isinstance(value, torch.Tensor)
            
            for order in self.derivative_orders:
                suffix = self.output_suffixes.get(order, f"_d{order}")
                derivative_key = f"{key}{suffix}"
                data[derivative_key] = self._compute_derivative(value, order)
        
        return data
    
    def unapply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove computed derivatives (they were auxiliary features)."""
        for key in self.apply_to:
            for order in self.derivative_orders:
                suffix = self.output_suffixes.get(order, f"_d{order}")
                derivative_key = f"{key}{suffix}"
                if derivative_key in data:
                    del data[derivative_key]
        
        return data
