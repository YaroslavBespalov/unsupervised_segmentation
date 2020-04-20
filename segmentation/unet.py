import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.nn.modules.unet.unet import UNet


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch, 0.8),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNetSegmentation(UNet):
    def __init__(self, n_classes: int, n_down: int = 5, nc_base: int = 32):

        nc_max = 256

        nc_middle = min(int(nc_base * (2 ** n_down)), nc_max)

        def down_block_factory(i: int) -> nn.Module:
            in_ch = min(
                int(nc_base * (2 ** i)),
                nc_max
            )
            out_ch = min(2 * in_ch, nc_max)

            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def up_block_factory(i: int) -> nn.Module:

            out_ch = min(
                int(nc_base * (2 ** (n_down - i))),
                nc_max
            )
            in_ch = min(
                3 * out_ch if i > 0 else 2 * out_ch,
                2 * nc_max
            )

            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                double_conv(out_ch, out_ch)
            )

        super(UNetSegmentation, self).__init__(
            n_down,
            in_block=double_conv(3, nc_base),
            out_block=nn.Sequential(
                nn.Conv2d(2 * nc_base, n_classes, 1),
                nn.BatchNorm2d(n_classes),
                nn.ReLU()
            ),
            middle_block=double_conv(nc_middle, nc_middle),
            down_block=down_block_factory,
            up_block=up_block_factory
        )


