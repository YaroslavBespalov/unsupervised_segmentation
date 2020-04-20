from torch import nn
import torch


class FCNSegmentation(nn.Module):

    def __init__(self, output_nc: int, input_nc: int = 3, n_conv: int = 3):
        super(FCNSegmentation, self).__init__()

        mid_nc = 150

        layers = [
            nn.Conv2d(input_nc, mid_nc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]

        for i in range(n_conv - 1):
            layers.extend([
                nn.Conv2d(mid_nc, mid_nc, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(mid_nc),
            ])

        layers.extend([
            nn.Conv2d(mid_nc, output_nc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_nc),
            nn.ReLU(inplace=True)
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)
