
import torch
import torch.nn.functional as F
from torch import nn

import escnn.nn as enn

class EquivariantGeLU(nn.Module):
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, bias=True):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.activation = nn.GELU()
        self.proj = enn.Linear(in_type, out_type, bias=bias)

    def forward(self, x: enn.GeometricTensor):
        x = self.proj(x)
        x = self.activation(x.tensor)
        return enn.GeometricTensor(x, self.out_type)

class EquivariantReLU(nn.Module):
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, bias=True):
        super().__init__()
        self.in_type = in_type
        self.activation = nn.ReLU()
        self.out_type = out_type
        self.proj = enn.Linear(in_type, out_type, bias=bias)

    def forward(self, x: enn.GeometricTensor):
        x = self.proj(x)
        x = self.activation(x.tensor)
        return enn.GeometricTensor(x, self.out_type)

class EquivariantSiLU(nn.Module):
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, bias=True):
        super().__init__()
        self.proj = enn.Linear(in_type, out_type, bias=bias)
        self.activation = nn.SiLU()
        self.out_type = out_type

    def forward(self, x: enn.GeometricTensor):
        x = self.proj(x)
        x = self.activation(x.tensor)
        return enn.GeometricTensor(x, self.out_type)

class Mish(nn.Module):
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, bias=True):
        super().__init__()
        self.proj = enn.Linear(in_type, out_type, bias=bias)
        self.activation = nn.Mish()
        self.out_type = out_type

    def forward(self, x: enn.GeometricTensor):
        x = self.proj(x)
        x = self.activation(x.tensor)
        return enn.GeometricTensor(x, self.out_type)

class ApproximateGELU(nn.Module):
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, bias=True):
        super().__init__()
        self.proj = enn.Linear(in_type, out_type, bias=bias)
        self.out_type = out_type

    def forward(self, x: enn.GeometricTensor):
        x = self.proj(x)
        xt = x.tensor
        xt = xt * torch.sigmoid(1.702 * xt)
        return enn.GeometricTensor(xt, self.out_type)


class EquivariantGEGLU(nn.Module):
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, bias=True):
        super().__init__()
        # two halves are each out_type
        doubled = out_type + out_type
        self.proj = enn.Linear(in_type, doubled, bias=bias)
        self.out_type = out_type

    def forward(self, x: enn.GeometricTensor):
        x = self.proj(x)
        v, g = x.tensor.chunk(2, dim=1)  # ESCNN uses (B,C,H,W)
        x = v * F.gelu(g)
        return enn.GeometricTensor(x, self.out_type)

class EquivariantSwiGLU(nn.Module):
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, bias=True):
        super().__init__()
        doubled = out_type + out_type
        self.proj = enn.Linear(in_type, doubled, bias=bias)
        self.activation = nn.SiLU()
        self.out_type = out_type

    def forward(self, x: enn.GeometricTensor):
        x = self.proj(x)
        v, g = x.tensor.chunk(2, dim=1)
        x = v * self.activation(g)
        return enn.GeometricTensor(x, self.out_type)

class EquivariantGate(nn.Module):
    """
    Gate-equivariant FFN (Equ-FFN):
        out = W_3^E( W_1^E(Z^E) ⊙ GELU(W_2^I(Z^I)) )

    Equivariance is preserved because the gate (from invariant Z^I) produces
    one scalar per equivariant field, which commutes with the group action.

    Args:
        in_type:    FieldType for Z^E (equivariant input and output).
        inner_type: FieldType for the hidden equivariant features.
        gate_type:  FieldType for Z^I (invariant input, typically trivial reprs).
        bias:       Whether to use bias in linear layers.
    """

    def __init__(
        self,
        in_type: enn.FieldType,
        inner_type: enn.FieldType,
        gate_type: enn.FieldType,
        bias: bool = True,
    ):
        super().__init__()
        self.inner_type = inner_type
        self.out_type = in_type

        # W_1^E: equivariant projection Z^E -> hidden
        self.proj_E = enn.Linear(in_type, inner_type, bias=bias)

        # W_2^I: invariant projection Z^I -> one scalar gate per field in inner_type
        num_gates = len(inner_type.representations)
        gate_out_type = enn.FieldType(
            gate_type.gspace, [gate_type.gspace.trivial_repr] * num_gates
        )
        self.proj_I = enn.Linear(gate_type, gate_out_type, bias=bias)

        # W_3^E: equivariant projection hidden -> in_type
        self.proj_out = enn.Linear(inner_type, in_type, bias=bias)

        # Field sizes used for broadcasting the gate signal
        self._field_sizes = [rep.size for rep in inner_type.representations]

    def forward(self, x: enn.GeometricTensor, y: enn.GeometricTensor) -> enn.GeometricTensor:
        # W_1^E(Z^E)  →  (B, inner_size)
        xe = self.proj_E(x).tensor

        # GELU(W_2^I(Z^I))  →  (B, num_gates)
        gate = F.gelu(self.proj_I(y).tensor)

        # Broadcast: gate[:, i] multiplies all channels that belong to field i
        gate_expanded = torch.cat(
            [gate[:, i : i + 1].expand(-1, s) for i, s in enumerate(self._field_sizes)],
            dim=1,
        )  # (B, inner_size)

        # Element-wise gate then project back to in_type
        gated = enn.GeometricTensor(xe * gate_expanded, self.inner_type)
        return self.proj_out(gated)
