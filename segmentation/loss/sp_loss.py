from torch import Tensor
import torch

from framework.Loss import Loss
from framework.nn.modules.common.sp_pool import SPPoolMean
from framework.segmentation.loss.base import SegmentationLoss


class SuperPixelsLoss(SegmentationLoss):

    def __init__(self, weight: float = 1.0):
        self.pooling = SPPoolMean()
        self.weight = weight
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, image: Tensor, sp: Tensor, segm: Tensor) -> Loss:
        nc = segm.shape[1]
        sp = torch.cat([sp] * nc, dim=1).detach()

        sp_argmax = self.pooling.forward(
            segm.detach(),
            sp
        ).detach().max(dim=1)[1]

        return Loss(self.loss(segm, sp_argmax)) * self.weight
