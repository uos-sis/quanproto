"""
Implements a non-negative linear layer.

reference: https://github.com/M-Nauta/PIPNet 
"""

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`"""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,), requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.relu(self.weight), self.bias)
