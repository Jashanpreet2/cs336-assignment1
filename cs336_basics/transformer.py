import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        mean = 0
        std = (2/(in_features+out_features)) ** 0.5
        lower, upper = -3 * std, 3 * std
        W = torch.rand(out_features, in_features)
        W = nn.init.trunc_normal_(W, mean, std, lower, upper)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")