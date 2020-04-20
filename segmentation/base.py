from typing import List

from torch import Tensor, nn
import torch

from framework.Loss import Loss, LossModule
from framework.nn.modules.common.sp_pool import SPPoolMean
from framework.parallel import ParallelConfig
from framework.segmentation.loss.base import SegmentationLoss


class PenalizedSegmentation:

    def __init__(self, segmentation: nn.Module):
        self.segmentation = nn.DataParallel(
             segmentation.to(ParallelConfig.MAIN_DEVICE),
             ParallelConfig.GPU_IDS
        )
        self.penalties: List[SegmentationLoss] = []
        self.opt = torch.optim.Adam(self.segmentation.parameters(), lr=0.001)
        # self.opt = torch.optim.SGD(self.segmentation.parameters(), lr=0.01, momentum=0.9)
        self.sp_pool = SPPoolMean()

    def forward(self, image: Tensor, sp: Tensor) -> Tensor:
        segm = self.segmentation(image)

        nc = segm.shape[1]
        sp = torch.cat([sp] * nc, dim=1)
        segm = self.sp_pool.forward(segm, sp)

        return segm

    def __call__(self, image: Tensor, sp: Tensor) -> Tensor:
        return self.forward(image, sp)

    def add_penalty(self, pen: LossModule):
        self.penalties.append(pen)

    def train(self, image: Tensor, sp: Tensor) -> Loss:
        segm = self.segmentation(image)
        loss: Loss = Loss.ZERO()
        for pen in self.penalties:
            loss = loss + pen.forward(image, sp, segm)
        loss.minimize_step(self.opt)

        return loss


