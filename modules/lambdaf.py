from typing import Callable, List

import torch
from torch import nn


class LambdaF(nn.Module):
    def __init__(self,  module: nn.Module, f: Callable[[nn.Module, torch.Tensor], torch.Tensor]):
        super().__init__()
        self.module = module
        self.f = f

    def forward(self, input: List[torch.Tensor]):
        return self.f(self.module, *input)
