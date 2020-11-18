import torch
from typing import Optional, Callable
from torch import nn, autograd, optim, Tensor

from dataset.probmeasure import UniformMeasure2DFactory
from dataset.toheatmap import heatmap_to_measure, CoordToGaussSkeleton, ToGaussHeatMap
from gan.loss.loss_base import Loss


def stariy_hm_loss(pred, target, coef=1.0):

    pred_mes = UniformMeasure2DFactory.from_heatmap(pred)
    target_mes = UniformMeasure2DFactory.from_heatmap(target)

    return Loss(
        nn.BCELoss()(pred, target) * coef +
        nn.MSELoss()(pred_mes.coord, target_mes.coord) * (0.001 * coef) +
        nn.L1Loss()(pred_mes.coord, target_mes.coord) * (0.001 * coef)
    )


def noviy_hm_loss(pred, target, coef=1.0):

    pred = pred / pred.sum(dim=[2, 3], keepdim=True).detach()
    target = target / target.sum(dim=[2, 3], keepdim=True).detach()

    return Loss(
        nn.BCELoss()(pred, target) * coef
    )


def coord_hm_loss(pred_coord: Tensor, target_hm: Tensor, coef=1.0):
    target_hm = target_hm / target_hm.sum(dim=[2, 3], keepdim=True)
    target_hm = target_hm.detach()

    heatmapper = ToGaussHeatMap(256, 1)

    target_coord = UniformMeasure2DFactory.from_heatmap(target_hm).coord.detach()
    sk = CoordToGaussSkeleton(target_hm.shape[-1], 1)
    pred_sk = sk.forward(pred_coord).sum(dim=1, keepdim=True)
    target_sk = sk.forward(target_coord).sum(dim=1, keepdim=True).detach()
    pred_hm = heatmapper.forward(pred_coord).sum(dim=1, keepdim=True)
    pred_hm = pred_hm / pred_hm.sum(dim=[2, 3], keepdim=True).detach()
    target_hm = heatmapper.forward(target_coord).sum(dim=1, keepdim=True).detach()
    target_hm = target_hm / target_hm.sum(dim=[2, 3], keepdim=True).detach()

    return Loss(
        nn.BCELoss()(pred_hm, target_hm) * coef * 0.5 +
        noviy_hm_loss(pred_sk, target_sk, coef).to_tensor() * 0.5 +
        nn.MSELoss()(pred_coord, target_coord) * (0.001 * coef) +
        nn.L1Loss()(pred_coord, target_coord) * (0.001 * coef)
    )



def hm_loss_bes_xy(pred, target):

    return Loss(nn.BCELoss()(pred, target))


def HMLoss(name: Optional[str], weight: float) -> Callable[[Tensor, Tensor], Loss]:

    # if name:
    #     counter.active[name] = True

    def compute(content: Tensor, target_hm: Tensor):

        content_xy, _ = heatmap_to_measure(content)
        target_xy, _ = heatmap_to_measure(target_hm)

        lossyash = Loss(
            nn.BCELoss()(content, target_hm) * weight +
            nn.MSELoss()(content_xy, target_xy) * weight * 0.0005
        )
        #
        # if name:
        #     writer.add_scalar(name, lossyash.item(), counter.get_iter(name))

        return lossyash

    return compute