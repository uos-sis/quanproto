import torch
import torch.nn as nn


class LogActivation(nn.Module):
    def __init__(self):
        super(LogActivation, self).__init__()

    def forward(self, input):

        return torch.log((1 + input) / (input + 1e-14))
