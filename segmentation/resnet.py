import torch
from torch import nn, Tensor
import numpy as np
from framework.nn.modules.resnet.down_block import DownBlock
from framework.nn.modules.resnet.residual import ResidualNet, PaddingType
from framework.nn.modules.resnet.up_block import UpBlock


class ResidualSegmentation(nn.Module):

    def __init__(self,
                 output_nc: int,
                 input_nc: int = 3,
                 ngf: int = 32,
                 n_down: int = 2,
                 n_residual_blocks: int = 3,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 padding_type: PaddingType = PaddingType.REFLECT):
        super(ResidualSegmentation, self).__init__()

        self.n_down = n_down
        max_ngf = 128

        self.model_downsample = DownBlock(input_nc, min(ngf * (2 ** n_down), max_ngf), ngf, n_down, norm_layer)

        self.model_resnet = ResidualNet(min(ngf * (2 ** n_down), max_ngf), n_residual_blocks, norm_layer, padding_type)

        self.model_upsample = UpBlock(2 * min(ngf * (2 ** n_down), max_ngf), output_nc * 2, ngf, n_down, norm_layer)

        self.output = nn.Sequential(
            nn.Conv2d(2 * output_nc, output_nc, 3, 1, 1),
            nn.BatchNorm2d(output_nc),
            nn.ReLU(inplace=True)
        )

    def forward(self, image: Tensor) -> Tensor:

        assert image.shape[-1] == image.shape[-2]

        downsample = self.model_downsample(image)
        resnet = self.model_resnet(downsample)
        upsample = self.model_upsample(torch.cat([resnet, downsample], dim=1))

        assert image.shape[-1] == upsample.shape[-1]

        return self.output(upsample)


