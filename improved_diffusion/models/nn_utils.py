from torch import nn
import torch
Nonlinearity = nn.SiLU

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

class SkipConnection(nn.Sequential):
    def forward(self, input):
        return input + super().forward(input)

class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)