import math
from typing import Dict

import torch
from torch import Tensor, nn

from dataset.probmeasure import ProbabilityMeasure
from framework.gan.generator import Generator as G
from framework.gan.noise import Noise
from framework.module import NamedModule
from framework.nn.modules.common.View import View
from framework.nn.modules.common.self_attention import SelfAttention2d
from framework.nn.modules.resnet.residual import Up2xResidualBlock, PaddingType, Up4xResidualBlock, PooledResidualBlock
from useful_utils.spectral_functions import spectral_norm_init

class MeasureToImage(G):

    def __init__(self, noise_size: int, image_size: int, ngf=64):
        super(MeasureToImage, self).__init__()
        n_up = int(math.log2(image_size / 4))
        assert 4 * (2 ** n_up) == image_size
        nc = 3

        self.preproc = nn.Sequential(
            nn.Linear(noise_size, noise_size, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        layers = [
            spectral_norm_init(nn.ConvTranspose2d(noise_size, ngf * 8, 4, 1, 0, bias=False)),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nc_l_next = -1
        for l in range(n_up):

            nc_l = max(ngf, (ngf * 8) // 2**l)
            nc_l_next = max(ngf, nc_l // 2)

            layers += [
                spectral_norm_init(nn.ConvTranspose2d(nc_l, nc_l_next, 4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(nc_l_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]

            if l == 2:
                layers += [SelfAttention2d(nc_l_next)]

        layers += [
            nn.Conv2d(nc_l_next, nc, 3, 1, 1, bias=False)
        ]

        self.main = nn.Sequential(*layers)

    def _device(self):
        return next(self.main.parameters()).device

    def forward(self, noise: Tensor) -> Tensor:
        noise = self.preproc(noise)
        return self.main(noise.view(*noise.size(), 1, 1))


class ResMeasureToImage(G):

    def __init__(self, noise_size: int, image_size: int, ngf=32):
        super(ResMeasureToImage, self).__init__()
        n_up = int(math.log2(image_size / 4))
        assert 4 * (2 ** n_up) == image_size
        nc = 3

        layers = [
            spectral_norm_init(nn.Linear(noise_size, noise_size // 2, bias=False), n_power_iterations=10),
            nn.LeakyReLU(0.2, inplace=True), #nn.ReLU(inplace=True),
            spectral_norm_init(nn.Linear(noise_size//2, noise_size//2, bias=False), n_power_iterations=10),
            nn.LeakyReLU(0.2, inplace=True),
            View(-1, noise_size // 2, 1, 1),
            spectral_norm_init(nn.ConvTranspose2d(noise_size // 2, ngf * 4, 4, 1, 0, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ngf * 4), # nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nc_l_next = -1
        for l in range(n_up):

            nc_l = max(ngf // 2, (ngf * 4) // 2**l)
            nc_l_next = max(ngf // 2, nc_l // 2)

            layers += [
                Up2xResidualBlock(nc_l, nc_l_next, PaddingType.REFLECT, nn.InstanceNorm2d, use_spectral_norm=True, activation=nn.LeakyReLU(0.2, inplace=True)), #nn.InstanceNorm2d
                PooledResidualBlock(nc_l_next, nc_l_next, nc_l_next, nn.Identity(), PaddingType.REFLECT, nn.InstanceNorm2d, use_spectral_norm=True,
                                  activation=nn.LeakyReLU(0.2, inplace=True))
            ]

            if l == 2:
                layers += [
                    # nn.Dropout2d(p=0.5),
                    SelfAttention2d(nc_l_next)]

        layers += [
            nn.Conv2d(nc_l_next, nc, 3, 1, 1, bias=False)
        ]

        self.main = nn.Sequential(*layers)

    def _device(self):
        return next(self.main.parameters()).device

    def forward(self, cond: Tensor, noise: Tensor = None) -> Tensor:
        return self.main(cond)


class Measure2imgTmp(NamedModule):
    def __init__(self,
                 module: nn.Module,
                 noise: Noise
                 ):
        super().__init__(module,
                 ['measure'],
                 ['image'])

        self.noise = noise

    def forward(self, name2tensor: Dict[str, ProbabilityMeasure]) -> Dict[str, Tensor]:
        measures: ProbabilityMeasure = name2tensor['measure']
        cond = measures.toChannels()
        z = self.noise.sample(cond.shape[0])
        cond = torch.cat((cond, z), dim=1)
        img = self.module(cond)
        return {'image': img}