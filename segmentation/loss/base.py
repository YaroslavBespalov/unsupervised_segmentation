from abc import ABC, abstractmethod
from torch import Tensor

from framework.Loss import LossModule, Loss


class SegmentationLoss(LossModule):

    @abstractmethod
    def forward(self, image: Tensor, superpixels: Tensor, segmentation: Tensor) -> Loss: pass
