import functools
from typing import Union

import numpy as np
import pytorch3d.transforms as pt
import torch


class RotationTransformer:
    valid_reps = [
        "axis_angle",
        "euler_angles",
        "quaternion",
        "rotation_6d",
        "matrix",
    ]

    def __init__(
        self,
        from_rep="axis_angle",
        to_rep="rotation_6d",
        from_convention=None,
        to_convention=None,
    ):
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = []
        inverse_funcs = []

        if from_rep != "matrix":
            funcs = [
                getattr(pt, f"{from_rep}_to_matrix"),
                getattr(pt, f"matrix_to_{from_rep}"),
            ]
            if from_convention is not None:
                funcs = [functools.partial(f, convention=from_convention) for f in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [
                getattr(pt, f"matrix_to_{to_rep}"),
                getattr(pt, f"{to_rep}_to_matrix"),
            ]
            if to_convention is not None:
                funcs = [functools.partial(f, convention=to_convention) for f in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(
        x: Union[np.ndarray, torch.Tensor], funcs: list
    ) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        for func in funcs:
            x_ = func(x_)
        if isinstance(x, np.ndarray):
            return x_.numpy()
        return x_

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)
