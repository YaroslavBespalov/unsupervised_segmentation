import torch
from torch import nn


class Accumulator(nn.Module):

    def __init__(self, storage_model: nn.Module, accumulate_every: int = 1):
        super().__init__()

        self.storage_model_params = dict(storage_model.named_parameters())
        self.storage_model = storage_model
        self.accumulate_every = accumulate_every

    def accumulate(self, model: nn.Module, i: int, decay=0.997):

        if i % self.accumulate_every == 0:

            params = dict(model.named_parameters())

            for k in params.keys():
                self.storage_model_params[k].data.mul_(decay)
                self.storage_model_params[k].data += (1 - decay) * params[k].data.cpu()

    def forward(self, *args, **kw):
        return self.storage_model.forward(*args, **kw)

    def write_to(self, model: nn.Module):
        params = dict(model.named_parameters())
        for k in params.keys():
            params[k].data = self.storage_model_params[k].data.clone().cuda()

