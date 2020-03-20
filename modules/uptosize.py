import math
from typing import List

from torch import nn, Tensor
import torch
from torch.nn import functional as F
from stylegan2_pytorch.model import Blur, ConvLayer
from stylegan2_pytorch.op import FusedLeakyReLU


class ScaledConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        blur_kernel=[1, 3, 3, 1]):

        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        factor = 2
        p = (len(blur_kernel) - factor) - (kernel_size - 1)
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2 + 1

        self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )

        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input: Tensor):

        weight = self.scale * self.weight

        out = F.conv_transpose2d(input, weight, padding=0, stride=2)
        out = self.blur(out)
        out = self.activate(out)

        return out


class Uptosize(nn.Module):

    def __init__(self, channel1, channel2, size2):
        super().__init__()
        modules = [
            ScaledConvTranspose2d(channel1, channel1//2, 3)
        ]
        tmp_size = 2
        tmp_channel = channel1//2
        min_nc = 8
        while tmp_size < size2:
            nc_next = max(min_nc, tmp_channel//2)
            modules.append(ScaledConvTranspose2d(tmp_channel, nc_next, 3))
            tmp_channel = nc_next
            tmp_size *= 2
        assert(tmp_size == size2)
        nc_next = max(min_nc, tmp_channel // 2)
        modules.append(ConvLayer(nc_next, channel2, 1))
        self.main = nn.Sequential(*modules)

    def forward(self, input: Tensor):
        return self.main(input.view(input.shape[0], input.shape[1], 1, 1))


class UpsampleList(nn.Module):

    def __init__(self, num_layers: int, nc: int, nc_min: int):
        super().__init__()

        self.nc = nc

        self.upsamples = nn.ModuleList()
        self.upsamples.append(nn.Sequential(
            ScaledConvTranspose2d(nc, nc // 2, 3),
            ScaledConvTranspose2d(nc // 2, nc // 2, 3)
        ))

        tmp_channel = nc // 2

        for i in range(num_layers-1):
            nc_next = max(nc_min, tmp_channel // 2)
            self.upsamples.append(
                ScaledConvTranspose2d(tmp_channel, nc_next, 3)
            )
            tmp_channel = nc_next

    def forward(self, vector: Tensor) -> List[Tensor]:
        batch = vector.shape[0]
        res = [
            self.upsamples[0](
                vector.view(batch, self.nc, 1, 1)
            )
        ]

        for layer in self.upsamples[1:]:
            res.append(
                layer(res[-1])
            )

        return res


class MakeNoise(nn.Module):

    def __init__(self, num_layers: int, nc: int):
        super().__init__()

        nc_min = 4
        self.upsample = UpsampleList(num_layers, nc, nc_min)
        self.to_noise = nn.ModuleList()
        tmp_channel = nc

        for i in range(num_layers):
            tmp_channel = max(nc_min, tmp_channel // 2)
            self.to_noise.append(
                ConvLayer(tmp_channel, 1, 3)
            )

    def forward(self, vector: Tensor):

        noises = []

        for i, usample in enumerate(self.upsample(vector)):
            ni = self.to_noise[i](usample)
            noises.append(ni)
            if i > 0:
                noises.append(ni)

        return noises
