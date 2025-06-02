"""
Implementation of L2Conv2d and L2SquaredConv2d layers in PyTorch.

reference: https://github.com/cfchen-duke/ProtoPNet
           https://github.com/M-Nauta/ProtoTree
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2SquaredConv2d(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(L2SquaredConv2d, self).__init__()
        self.dtype = dtype

    def forward(self, input, weights):

        ones = torch.ones_like(weights, dtype=self.dtype, requires_grad=False)
        # Compute squared sum of inputs
        input2 = input.pow(2)  # Squaring each element in the input
        input2_patch_sum = F.conv2d(input=input2, weight=ones)
        # print("input2_patch_sum", input2_patch_sum.shape)

        # Compute squared sum of weights
        weights2 = weights.pow(2)  # Squaring each element in the weights
        weights2 = torch.sum(weights2, dim=(1, 2, 3))
        weights2 = weights2.view(-1, 1, 1)
        # print("weights2_patch_sum", weights2_patch_sum.shape)

        # Compute the dot product between input and weights
        input_weights = F.conv2d(input=input, weight=weights)
        # print("input_weights", input_weights.shape)

        # Final output by combining all terms
        output = input2_patch_sum - 2 * input_weights + weights2
        output = F.relu(output)

        if torch.isnan(output).any():
            raise Exception("Error: NaN values in L2SquaredConv2d")

        return output


class L2Conv2d(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(L2Conv2d, self).__init__()
        self.dtype = dtype

    def forward(self, input, weights):

        ones = torch.ones_like(weights, dtype=self.dtype, requires_grad=False)
        # Compute squared sum of inputs
        input2 = input.pow(2)  # Squaring each element in the input
        input2_patch_sum = F.conv2d(input=input2, weight=ones)
        # print("input2_patch_sum", input2_patch_sum.shape)

        # Compute squared sum of weights
        weights2 = weights.pow(2)  # Squaring each element in the weights
        weights2 = torch.sum(weights2, dim=(1, 2, 3))
        weights2 = weights2.view(-1, 1, 1)
        # print("weights2_patch_sum", weights2_patch_sum.shape)

        # Compute the dot product between input and weights
        input_weights = F.conv2d(input=input, weight=weights)
        # print("input_weights", input_weights.shape)

        # Final output by combining all terms
        output = input2_patch_sum - 2 * input_weights + weights2
        output = torch.sqrt(torch.abs(output) + 1e-14)

        if torch.isnan(output).any():
            raise Exception("Error: NaN values in L2Conv2d")

        return output
