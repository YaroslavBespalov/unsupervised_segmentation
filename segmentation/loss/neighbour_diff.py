import math

import torch
from torch import Tensor, nn

from framework.Loss import Loss
from framework.nn.ops.pairwise_map import LocalPairwiseMap2D
from framework.parallel import ParallelConfig
from framework.segmentation.Mask import Mask


class NeighbourDiffLoss:

    def __init__(self, kernel_size: int):

        self.kernel_size = kernel_size
        self.mapper = LocalPairwiseMap2D(5)

    def diff(self, di, dj, p1, p2):
        r = math.sqrt(di**2 + dj**2)
        return (p1 - p2).abs().sum(dim=2, keepdim=True) * math.exp(- r / 2)

    def __call__(self, segm: Mask) -> Loss:

        return Loss(self.mapper(segm.tensor, self.diff).mean())
