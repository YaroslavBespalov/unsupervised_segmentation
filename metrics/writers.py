import time
from typing import List, Tuple, Type
import torch
from PIL import Image
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

from gan.loss_base import Loss
from optim.min_max import MinMaxLoss


def tensorboard_scatter(tensor : Tensor, writer: SummaryWriter, step: int):
    import matplotlib.pyplot as plt

    x, y = tensor[:, :, 0], tensor[:, :, 1]
    plt.switch_backend('agg')
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        ax[i].scatter(x[i].numpy(), y[i].numpy())
    writer.add_figure('Measure', fig, global_step=step)


def send_images_to_tensorboard(writer, data: Tensor, name: str, iter: int, count=8, normalize=True, range=(-1, 1)):
    with torch.no_grad():
        grid = make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=normalize, range=range,
            scale_each=False)
        writer.add_image(name, grid, iter)


class ItersCounter:

    def __init__(self):
        self.__iter = 0
        self.active = {}

    def update(self, iter):
        self.__iter = iter
        for k in self.active.keys():
            self.active[k] = True

    def get_iter(self, key: str):
        self.active[key] = False
        return self.__iter


def send_to_tensorboard(*name2type: str, counter: ItersCounter, skip: int = 1, writer): #SummaryWriter("runs")

    def decorator(fn):
        counter.active[str(fn)] = True

        def decorated(*args, **kwargs):
            res = fn(*args, **kwargs)
            if not counter.active[str(fn)]:
                return res

            iter = counter.get_iter(str(fn))

            if not iter % skip == 0:
                return res

            res_tuple: Tuple[Tensor] = res
            if not isinstance(res, (tuple, list)):
                res_tuple = (res,)

            for i in range(len(name2type)):
                if isinstance(res_tuple[i], MinMaxLoss):
                    writer.add_scalar(name2type[i], res_tuple[i].min_loss.to_tensor(), iter)
                    writer.add_scalar(name2type[i+1], res_tuple[i].max_loss.to_tensor(), iter)
                    break
                elif isinstance(res_tuple[i], float):
                    writer.add_scalar(name2type[i], res_tuple[i], iter)
                elif isinstance(res_tuple[i], Loss):
                    writer.add_scalar(name2type[i], res_tuple[i].to_tensor(), iter)
                elif isinstance(res_tuple[i], Tensor) and len(res_tuple[i].shape) == 4:

                    with torch.no_grad():
                        grid = make_grid(res_tuple[i][0:4], nrow=4, padding=2, pad_value=0, normalize=True, range=(-1, 1), scale_each=False)
                        writer.add_image(name2type[i], grid, iter)

            return res

        return decorated

    return decorator









