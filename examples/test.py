from typing import Callable

import torch
from torch import nn, Tensor

class PairwiseCost(nn.Module):

    def __init__(self, cost: Callable[[Tensor, Tensor], Tensor]):
        super().__init__()
        self.cost = cost

    def forward(self, x: Tensor, y: Tensor):

        B, N, D = x.shape

        assert y.shape[1] == N
        x = x[:, :, None, :]
        y = y[:, None, :, :]
        # x = x.repeat(1, 1, N, 1).view(B, N * N, D)
        # y = y.repeat(1, N, 1, 1).view(B, N * N, D)


        return self.cost(x, y)


c1 = torch.randn(1, 5, 2).sigmoid()
c2 = torch.randn(1, 5, 2).sigmoid()

print((c1[0, 2] - c2[0, 1]).pow(2).sum())
print((c1[0, 1] - c2[0, 2]).pow(2).sum())

M = PairwiseCost(lambda x, y: (x-y).pow(2).sum(-1))(c1, c2)
print(M[0, 2, 1])

