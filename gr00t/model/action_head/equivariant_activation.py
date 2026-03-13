
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
