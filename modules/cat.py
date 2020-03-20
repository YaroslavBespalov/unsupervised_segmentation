from typing import List

import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self, dim : int):
        super().__init__()
        self.dim = dim

    def forward(self, x: List[torch.Tensor]):
        return torch.cat(x, dim=self.dim)

