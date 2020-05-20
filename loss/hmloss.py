import torch
from typing import Optional, Callable
from torch import nn, autograd, optim, Tensor

from dataset.toheatmap import heatmap_to_measure
from loss_base import Loss


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