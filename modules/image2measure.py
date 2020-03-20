import math

import torch
from torch import Tensor, nn

from dataset.probmeasure import ProbabilityMeasure
from models.attention import SelfAttention2d
from useful_utils.spectral_functions import spectral_norm_init


class ResImageToMeasure(nn.Module):
    def __init__(self, measure_size: int, nc=3, ndf=64):
        super(ResImageToMeasure, self).__init__()

        self.measure_size = measure_size

        self.main = nn.Sequential(
            # 256 x 256
            spectral_norm_init(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf),
            nn.ReLU(inplace=True),
            # 128 x 128
            spectral_norm_init(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf),
            nn.ReLU(inplace=True),
            # 64 x 64
            spectral_norm_init(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf),
            nn.ReLU(inplace=True),
            # 32 x 32
            # SelfAttention2d(ndf),
            spectral_norm_init(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf * 2),
            nn.ReLU(inplace=True),
            spectral_norm_init(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf * 4),
            nn.ReLU(inplace=True),
            spectral_norm_init(nn.Conv2d(4 * ndf, ndf * 8, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf * 8),
            nn.ReLU(inplace=True),
            spectral_norm_init(nn.Conv2d(8 * ndf, ndf * 8, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf * 8),
            nn.ReLU(inplace=True),
        )

        self.coord = nn.Sequential(
            nn.Linear(ndf * 8 * 2 * 2, 2 * measure_size, bias=True),
            nn.Sigmoid()
        )
        self.prob = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, measure_size, bias=False),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> ProbabilityMeasure:
        conv = self.main(x)

        coord = self.coord(
            conv.view(conv.shape[0], -1)
        ).view(x.shape[0], self.measure_size, 2)
        coord = coord * 255 / 256

        # prob = self.prob(
        #     conv.view(conv.shape[0], -1)
        # ).view(x.shape[0], self.measure_size) + 1e-8
        prob = torch.ones(x.shape[0], coord.shape[1], device=coord.device, dtype=torch.float32)
        prob = prob / (prob.sum(dim=1, keepdim=True))

        return ProbabilityMeasure(prob, coord)
