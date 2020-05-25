from typing import List
import torch
from torch import Tensor
import numpy as np
from modules.ToImage import ToImage2D


class ToHeatMap:
    def __init__(self, size):
        self.size = size

    def forward(self, values: Tensor, coord: Tensor):
        B, N, D = coord.shape

        coord_0 = coord[:, :, 0]
        coord_0_f = coord_0.floor().type(torch.int64)
        coord_0_c = coord_0.ceil().type(torch.int64)

        coord_1 = coord[:, :, 1]
        coord_1_f = coord_1.floor().type(torch.int64)
        coord_1_c = coord_1.ceil().type(torch.int64)

        diff0c = (coord_0 - coord_0_c).abs()
        diff0c[coord_0_c == coord_0_f] = 1
        diff1c = (coord_1 - coord_1_c).abs()
        diff1c[coord_1_c == coord_1_f] = 1

        prob_ff = diff0c * diff1c * values
        prob_fc = diff0c * (coord_1 - coord_1_f).abs() * values
        prob_cf = (coord_0 - coord_0_f).abs() * diff1c * values
        prob_cc = (coord_0 - coord_0_f).abs() * (coord_1 - coord_1_f).abs() * values

        tmp = torch.zeros(B, N * self.size * self.size, device=coord.device)
        arangesik = torch.arange(N, device=coord.device)[None, :]
        tmp.scatter_add_(1, arangesik * self.size ** 2 + coord_1_f * self.size + coord_0_f,
                         prob_ff)
        tmp.scatter_add_(1, arangesik * self.size ** 2 + coord_1_c * self.size + coord_0_f,
                         prob_fc)
        tmp.scatter_add_(1, arangesik * self.size ** 2 + coord_1_f * self.size + coord_0_c,
                         prob_cf)
        tmp.scatter_add_(1, arangesik * self.size ** 2 + coord_1_c * self.size + coord_0_c,
                         prob_cc)

        return tmp.reshape((B, N, self.size, self.size))


def make_coords(B, x_dim, y_dim, device):

        xx_channel = torch.arange(x_dim, device=device, dtype=torch.float32).repeat(B, 1, y_dim, 1)
        yy_cahnnel = torch.arange(y_dim, device=device, dtype=torch.float32).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2)

        return torch.cat([xx_channel, yy_cahnnel], dim=1)


class ToGaussHeatMap:

    def __init__(self, size, sigma):
        self.size = size
        self.sigma = sigma

    def forward(self, coord: Tensor):
        B, N, D = coord.shape

        xy = make_coords(B, self.size, self.size, coord.device)\
            .view(B, 1, 2, self.size, self.size).repeat(1, N, 1, 1, 1)

        xy_dist = -(xy - coord.view(B, N, 2, 1, 1)).pow(2).sum(dim=2) / (self.sigma**2)
        xy_dist = xy_dist.exp()
        norm = xy_dist.sum(dim=[1, 2, 3], keepdim=True)

        return xy_dist / norm


def heatmap_to_measure(hm: Tensor):

        B, N, D, D = hm.shape

        x = torch.arange(D, device=hm.device).view(1, 1, -1)
        y = torch.arange(D, device=hm.device).view(1, 1, -1)
        px = hm.sum(dim=3)
        py = hm.sum(dim=2)
        p = hm.sum(dim=[2, 3]) + 1e-7
        coords_x = ((px * x).sum(dim=2) / p)[..., None]
        coords_y = ((py * y).sum(dim=2) / p)[..., None]
        coords = torch.cat([coords_y, coords_x], dim=-1) / (D - 1)

        return coords, p


def sparse_heatmap(hm: Tensor):
    B, N, D, D = hm.shape

    shm = hm.pow(2)
    norm = shm.sum(dim=[2, 3]) + 1e-7

    return shm / norm.view(B, N, 1, 1)

