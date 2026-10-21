import torch
from torch import nn
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = nn.init.trunc_normal_(torch.rand((in_features, out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        