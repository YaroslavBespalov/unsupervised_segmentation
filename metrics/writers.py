import time
from typing import List, Tuple, Type
import torch
from PIL import Image
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


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


def send_to_tensorboard(*name2type: str, counter: ItersCounter, skip: int = 1, writer=SummaryWriter("runs")):

    def decorator(fn):
        counter.active[str(fn)] = True

        def decorated(*args, **kwargs):
            res = fn(*args, **kwargs)
            if not counter.active[str(fn)]:
                return res

            iter = counter.get_iter(str(fn))

            if not iter % skip == 0:
                return res

            if not isinstance(res, (tuple, list)):
                res = (res,)
            for i in range(len(name2type)):
                if isinstance(res[i], float):
                    writer.add_scalar(name2type[i], res[i], iter)
                elif isinstance(res[i], Tensor) and len(res[i].shape) == 4:

                    with torch.no_grad():
                        grid = make_grid(res[i][0:4], nrow=4, padding=2, pad_value=0, normalize=True, range=(-1, 1), scale_each=False)
                        writer.add_image(name2type[i], grid, iter)

            return res

        return decorated

    return decorator




