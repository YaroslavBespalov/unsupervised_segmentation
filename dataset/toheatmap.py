from typing import List
import torch
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.neighbors import kneighbors_graph
from torch import Tensor
import numpy as np
from torch_cluster import knn_graph, graclus_cluster
from modules.ToImage import ToImage2D
from parameters.path import Paths


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

        assert coord.max() < 1.0001
        coord = coord * (self.size - 1)

        xy = make_coords(B, self.size, self.size, coord.device)\
            .view(B, 1, 2, self.size, self.size).repeat(1, N, 1, 1, 1)

        xy_dist = -(xy - coord.view(B, N, 2, 1, 1)).pow(2).sum(dim=2) / (self.sigma**2)
        xy_dist = xy_dist.clamp(-30, 30)

        xy_dist = xy_dist.exp()

        return xy_dist


class ToParabola:

    def __init__(self, size, sigma):
        self.size = size
        self.sigma = sigma

    def forward(self, coord: Tensor):
        B, N, D = coord.shape

        xy = make_coords(B, self.size, self.size, coord.device)\
            .view(B, 1, 2, self.size, self.size).repeat(1, N, 1, 1, 1)

        xy_dist = -(xy - coord.view(B, N, 2, 1, 1)).abs().sum(dim=2) / (self.sigma**2)

        # norm = xy_dist.max(dim=[2, 3], keepdim=True)

        return (14 + xy_dist).relu() - 7


class HeatMapToGaussHeatMap(ToGaussHeatMap):

    def forward(self, hm: Tensor):
        coords, p = heatmap_to_measure(hm)
        return super().forward(coords * (self.size - 1))


class HeatMapToParabola(ToParabola):

    def forward(self, hm: Tensor):
        coords, p = heatmap_to_measure(hm)
        return super().forward(coords * (self.size - 1))


def heatmap_to_measure(hm: Tensor):

        B, N, D, D = hm.shape

        hm = hm.relu()

        hm = hm / hm.sum(dim=[2,3], keepdim=True)
        hm = hm / hm.shape[1]

        x = torch.arange(D, device=hm.device).view(1, 1, -1)
        y = torch.arange(D, device=hm.device).view(1, 1, -1)
        px = hm.sum(dim=3)
        py = hm.sum(dim=2)
        p = hm.sum(dim=[2, 3]) + 1e-7
        coords_x = ((px * x).sum(dim=2) / p)[..., None]
        coords_y = ((py * y).sum(dim=2) / p)[..., None]
        coords = torch.cat([coords_y, coords_x], dim=-1) / (D - 1)

        # if coords.max() > 1.0:
        # print(coords.max())

        return coords, p


def sparse_heatmap(hm: Tensor):
    B, N, D, D = hm.shape

    shm = hm.pow(2)
    norm = shm.sum(dim=[2, 3]) + 1e-7

    return shm / norm.view(B, N, 1, 1)


def dist_to_line(p: Tensor, p0: Tensor, p1: Tensor):
    x, x0, x1 = p[:, :, 0], p0[:, :, 0], p1[:, :, 0]
    y, y0, y1 = p[:, :, 1], p0[:, :, 1], p1[:, :, 1]

    d01 = (p0 - p1).pow(2).sum(2).sqrt()

    return ((y0 - y1) * x - (x0 - x1) * y + (x0 * y1 - x1 * y0)).abs() / d01


def dist_to_snippet(p: Tensor, p0: Tensor, p1: Tensor):

    x, x0, x1 = p[:, :, 0], p0[:, :, 0], p1[:, :, 0]
    y, y0, y1 = p[:, :, 1], p0[:, :, 1], p1[:, :, 1]
    d01 = (p0 - p1).pow(2).sum(2) + 1e-4

    t =((x-x0)*(x1-x0) + (y-y0)*(y1-y0)) / d01
    t = t.relu()
    t = 1 - (1 - t).relu()
    t = t.detach()

    assert not torch.any(torch.isnan(t))

    return ((x0 - x + (x1-x0)*t).pow(2) + (y0 - y + (y1-y0)*t).pow(2))


class ToGaussSkeleton:

    def __init__(self, size, sigma):
        self.size = size
        self.sigma = sigma

    def forward(self, p0: Tensor, p1: Tensor):
        B, N, D = p0.shape

        xy = make_coords(B, self.size, self.size, p0.device)\
            .view(B, 1, 2, self.size, self.size).repeat(1, N, 1, 1, 1)

        xy_dist = -dist_to_snippet(xy, p0.view(B, N, 2, 1, 1), p1.view(B, N, 2, 1, 1)) / (self.sigma**2)
        xy_dist = xy_dist.clamp(-30, 30)

        p = xy_dist.exp()
        assert not torch.any(torch.isnan(p))

        return p


def crop_min(at: List[Tensor]):
    size = min([t.shape[1] for t in at])
    return [t[:, 0:size] for t in at]


class CoordToGaussSkeleton(ToGaussSkeleton):

    def find_pairs(self, coord: Tensor):

        B, N, D = coord.shape

        p0 = []
        p1 = []

        for i in range(B):
            edge_index = knn_graph(coord[i], k=2, loop=False,)

            coord0 = coord[i, edge_index[0]]
            coord1 = coord[i, edge_index[1]]
            p0.append(coord0[None, ])
            p1.append(coord1[None, ])

        return torch.cat(crop_min(p0)), torch.cat(crop_min(p1))

    def forward(self, coord: Tensor):

        p0, p1 = self.find_pairs(coord)
        assert not torch.any(torch.isnan(p0))
        assert not torch.any(torch.isnan(p1))
        assert p0.max().item() <= 1 + 1e-5
        assert p1.max().item() <= 1 + 1e-5

        return super().forward(p0 * (self.size-1), p1 * (self.size-1))


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from dataset.probmeasure import UniformMeasure2DFactory, UniformMeasure2D01
    from modules.linear_ot import PairwiseDistance, SOT

    N = 10
    B = 1
    size = 256

    barycenter: UniformMeasure2D01 = UniformMeasure2DFactory.load(
        f"{Paths.default.models()}/face_barycenter_68").batch_repeat(10)

    barycenter.coord = barycenter.coord[:, torch.randperm(68)]

    hm = CoordToGaussSkeleton(size, 5).forward(barycenter.coord)
    plt.imshow(hm[0].sum(0))

    plt.show()
