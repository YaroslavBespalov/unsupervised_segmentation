import torch
from torch import nn, Tensor
from framework.Loss import Loss


class SegmentationEntropy(nn.Module):

    def forward(self, segm: Tensor) -> Loss:

        return Loss((-segm * (segm + 1e-8).log()).sum(dim=1).mean())
