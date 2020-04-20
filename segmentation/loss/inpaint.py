from multiprocessing import Pool
from torch import Tensor
import numpy as np
import torch
from framework.Loss import LossModule, Loss
from framework.gan.GANModel import ConditionalGANModel
from framework.gan.conditional import ConditionalGenerator, ConditionalDiscriminator
from framework.gan.vgg.gan_loss import VggGeneratorLoss
from framework.nn.modules.common.sp_pool import SPPoolMean
from framework.segmentation.loss.base import SegmentationLoss


class InPaintLoss(SegmentationLoss):

    def __init__(self,
                 gan: ConditionalGANModel,
                 weight: float = 1.0):
        self.weight = weight
        self.gan = gan

    def forward(self, image: Tensor, sp: Tensor, segm: Tensor) -> Loss:

        painted = self.gan.generator(segm)
        generator_loss = self.gan.generator_loss(image, painted, segm)

        return generator_loss

