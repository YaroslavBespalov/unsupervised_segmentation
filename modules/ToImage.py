import torch
from torch import nn, Tensor


class ToImage2D(nn.Module):

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, values: Tensor, coord: Tensor):

        size = self.size
        batch = coord.shape[0]

        coord_0 = coord[:, :, 0]
        coord_0_f = coord_0.floor().type(torch.int64)
        coord_0_c = coord_0.ceil().type(torch.int64)

        coord_1 = coord[:, :, 1]
        coord_1_f = coord_1.floor().type(torch.int64)
        coord_1_c = coord_1.ceil().type(torch.int64)

        indexes_ff = coord_0_f * size + coord_1_f
        indexes_fc = coord_0_f * size + coord_1_c
        indexes_cf = coord_0_c * size + coord_1_f
        indexes_cc = coord_0_c * size + coord_1_c

        diff0c = (coord_0 - coord_0_c).abs()
        diff0c[coord_0_c == coord_0_f] = 1
        diff1c = (coord_1 - coord_1_c).abs()
        diff1c[coord_1_c == coord_1_f] = 1

        prob_ff = diff0c * diff1c * values
        prob_fc = diff0c * (coord_1 - coord_1_f).abs() * values
        prob_cf = (coord_0 - coord_0_f).abs() * diff1c * values
        prob_cc = (coord_0 - coord_0_f).abs() * (coord_1 - coord_1_f).abs() * values

        img = torch.zeros([batch, size * size], dtype=torch.float32, device=coord.device)

        assert indexes_ff.max().item() < size * size
        assert indexes_fc.max().item() < size * size
        assert indexes_cf.max().item() < size * size
        assert indexes_cc.max().item() < size * size

        img.scatter_add_(1, indexes_ff, prob_ff)
        img.scatter_add_(1, indexes_fc, prob_fc)
        img.scatter_add_(1, indexes_cf, prob_cf)
        img.scatter_add_(1, indexes_cc, prob_cc)

        return img.view(batch, 1, size, size)
