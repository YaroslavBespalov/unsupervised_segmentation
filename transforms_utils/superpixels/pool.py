import torch
from torch import nn, Tensor


class SPPoolMean(nn.Module):

    def _pool(self, src: Tensor, labels: Tensor):

        assert src.shape == labels.shape

        flat_shape = list(src.shape[:-2]) + [-1]

        return torch.zeros_like(src).view(flat_shape).scatter_add_(-1, labels.view(flat_shape), src.view(flat_shape))

    def forward(self, src: Tensor, labels: Tensor):

        pooled = self._pool(src, labels)
        norm = self._pool(torch.ones_like(src), labels)
        pooled /= norm
        flat_shape = list(src.shape[:-2]) + [-1]

        return torch.gather(pooled, dim=-1, index=labels.view(flat_shape)).view(src.shape)