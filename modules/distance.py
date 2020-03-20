from torch import nn, Tensor

from dataset.probmeasure import ProbabilityMeasure
from framework.loss import Loss


class ExpDist(nn.Module):

    def __init__(self, dist: nn.Module):
        super().__init__()
        self.dist = dist

    def forward(self, m1: ProbabilityMeasure, m2: ProbabilityMeasure) -> Loss:
        pass

