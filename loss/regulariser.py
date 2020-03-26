from typing import Generic, TypeVar, Callable, Tuple, Dict, List
import torch
import torch.nn as nn
import numpy as np
from albumentations import DualTransform
# from conda.models.dist import Dist
from torch import Tensor

from dataset.probmeasure import ProbabilityMeasure
from loss.losses import Samples_Loss
from loss_base import Loss
from modules.linear_ot import LinearTransformOT, SOT, PairwiseDistance


class RegularizerObject:
    @staticmethod
    def __call__(func):
        return Regularizer(func)


class Regularizer(nn.Module):
    def __init__(self, func):
        super(Regularizer, self).__init__()
        self.forward = func

    def __add__(self, reg: nn.Module):
        def forward_add(*x):
            return self.forward(*x) + reg.forward(*x)
        return RegularizerObject.__call__(forward_add)

    def __mul__(self, v: float):
        def forward_mul(*x):
            return self.forward(*x) * v
        return RegularizerObject.__call__(forward_mul)

    def __matmul__(self, array: List[float]):
        def apply(index: int):
            return self.__mul__(array[min(index, len(array) - 1)])
        return Apply(apply)


class ApplyObject:
    @staticmethod
    def __call__(func):
        return Apply(func)


class Apply:
    def __init__(self, apply):
        self.apply = apply

    def __add__(self, other):
        if isinstance(other, Regularizer) or isinstance(other, RegularizerObject):
            return ApplyObject.__call__(lambda i: self.apply(i) + other)
        else:
            return ApplyObject.__call__(lambda i: self.apply(i) + other.apply(i))





class DualTransformRegularizer:

    @staticmethod
    def __call__(transform: DualTransform,
                 loss: Callable[[Dict[str, object]], Loss]):

        return RegularizerObject.__call__(lambda image, mask: loss(transform(image=image, mask=mask)))


class BarycenterRegularizer:

    @staticmethod
    def __call__(barycenter, ct: float = 0.3, ca: float = 1, cw: float = 10):

        def loss(image: Tensor, mask: ProbabilityMeasure):
            with torch.no_grad():
                A, T = LinearTransformOT.forward(mask, barycenter, 100)
            #     P = SOT(200, 0.001).forward(mask, barycenter)

            t_loss = Samples_Loss(scaling=0.8)(mask, mask.detach() + T)
            a_loss = Samples_Loss(scaling=0.8)(mask.centered(), mask.centered().multiply(A).detach())
            w_loss = Samples_Loss(scaling=0.8)(mask.centered().multiply(A), barycenter.centered().detach())

            return a_loss * ca + w_loss * cw + t_loss * ct
            # return Samples_Loss()(mask, barycenter.detach()) * 10
            # M = PairwiseDistance()(mask.coord, barycenter.coord)
            # return Loss((M * P).mean(0).sum()) * 10

        return RegularizerObject.__call__(loss)
