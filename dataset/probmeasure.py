from typing import List
import torch
from torch import Tensor
import numpy as np

from dataset.toheatmap import heatmap_to_measure
from modules.ToImage import ToImage2D


class ProbabilityMeasure:
    def __init__(self, param: Tensor, coord: Tensor):
        self.probability: Tensor = param
        self.coord = coord
        self.device = param.device
        assert (self.probability.shape[0] == self.coord.shape[0])
        assert (self.probability.shape[1] == self.coord.shape[1])
        assert (len(self.probability.shape) == 2 and len(self.coord.shape) == 3)
        assert (self.probability.min().item() > -0.0001)
        assert ((self.probability.sum(dim=1) - 1).abs().max().item() < 0.01)
        assert (self.coord.max().item().__abs__() < 10.01)

    def crop(self, size):
        prob = self.probability[:, 0:size]
        prob /= prob.sum(dim=1, keepdim=True)
        return ProbabilityMeasure(prob, self.coord[:, 0:size, :])

    def transpose(self):
        return ProbabilityMeasure(self.probability, self.coord[:, :, [1, 0]])

    def padding(self, size):
        coord_pad = torch.zeros((self.coord.shape[0],size - self.coord.shape[1], self.coord.shape[2]), device=self.device)
        prob_pad = torch.zeros((self.probability.shape[0], size - self.probability.shape[1]), device=self.device)
        return ProbabilityMeasure(torch.cat((self.probability, prob_pad), dim=1), torch.cat((self.coord, coord_pad), dim=1))

    def batch_repeat(self, batch_size):
        return ProbabilityMeasure(
            self.probability.repeat(batch_size, 1),
            self.coord.repeat(batch_size, 1, 1)
        )

    def domain_dim(self):
        return self.coord.shape[1]

    def random_permute(self):
        perm = torch.randperm(self.probability.shape[1], device=self.device)
        return ProbabilityMeasure(self.probability[:, perm], self.coord[:, perm, :])

    def plus(self, x):
        return ProbabilityMeasure(self.probability, self.coord + x)

    def minus(self, x):
        return ProbabilityMeasure(self.probability, self.coord - x)

    def multiply(self, A):
        return ProbabilityMeasure(self.probability, torch.matmul(self.coord, A))

    def centered(self):
        return ProbabilityMeasure(self.probability, self.coord - self.mean())

    def cuda(self):
        return ProbabilityMeasure(self.probability.cuda(), self.coord.cuda())

    def mean(self):
        return (self.coord * self.probability[:, :, None]).sum(dim=1, keepdim=True)

    def __add__(self, t: Tensor):
        return ProbabilityMeasure(self.probability, self.coord + t)

    def __sub__(self, t: Tensor):
        return ProbabilityMeasure(self.probability, self.coord - t)

    def __mul__(self, A: Tensor):
        return ProbabilityMeasure(self.probability, torch.matmul(self.coord, A))

    def norm(self):
        return (self.coord.norm(dim=2) * self.probability).sum(dim=1).mean() * self.domain_dim()

    def slice(self, i1, i2):
        return ProbabilityMeasure(self.probability[i1:i2], self.coord[i1:i2])

    def toImage(self, size: int) -> Tensor:
        coord = self.coord * (size-1)

        assert int(coord.min().floor().item()) >= 0
        assert int(coord.max().ceil().item()) <= (size-1)

        return ToImage2D(size)(self.probability, coord).transpose(2, 3)

    def toChannels(self) -> Tensor:

        res = torch.cat((self.coord.transpose(1, 2), self.probability[:, :, None].transpose(1, 2)), dim=1)
        return res.view(res.shape[0], self.domain_dim() * 3)

    def requires_grad_(self):
        self.probability.requires_grad_(True)
        self.coord.requires_grad_(True)
        return self

    def detach(self):
        return ProbabilityMeasure(self.probability.detach(), self.coord.detach())


class UniformMeasure2D01(ProbabilityMeasure):

    def __init__(self, coord: Tensor):
        B, N, D = coord.shape
        assert D == 2
        assert coord.max() < 1 + 1e-8
        assert coord.min() > - 1e-8
        prob = torch.ones(B, N, device=coord.device, dtype=torch.float32) / N
        super().__init__(prob, coord)


class ProbabilityMeasureFabric:
    def __init__(self, size):
        use_cuda = torch.cuda.is_available()
        self.dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.size = size

    def cat(self, list_measure: List[ProbabilityMeasure]):
        size = max([x.domain_dim() for x in list_measure])
        tmp = [i.padding(size) for i in list_measure]
        return ProbabilityMeasure(torch.cat([i.probability for i in tmp], dim=0), torch.cat([i.coord for i in tmp], dim=0))

    def __from_one_mask(self, image: Tensor):
        assert(len(image.size()) == 2)
        weights = image.type(self.dtype)
        indices = (weights > 1e-5).nonzero().type(self.dtype) / float(self.size)
        indices = indices[:,[1,0]]
        indices.view(-1,2)
        values = torch.ones_like(weights)[weights > 1e-5].view(-1).type(self.dtype)
        values = values/values.sum(dim=0)
        assert(indices.shape[1] == 2)
        return ProbabilityMeasure(values[None,], indices[None,])

    def from_mask(self, image: Tensor):
        if len(image.shape) == 4:
            assert image.shape[1] == 1
            image = image[:, 0, :, :]

        probabilymeasurelist = [self.__from_one_mask(image[i]) for i in range(len(image))]
        return self.cat(probabilymeasurelist)

    def from_channels(self, x: Tensor) -> ProbabilityMeasure:
        n = x.shape[1] // 3
        x = x.view(x.shape[0], 3, n)
        coord = x[:, 0:2, :].transpose(1, 2)
        prob = x[:, 2, :]
        return ProbabilityMeasure(prob, coord)

    def random(self, size: int) -> ProbabilityMeasure:
        prob = torch.ones(1, size).softmax(dim=1).type(torch.float32)
        coord = torch.rand(1, size, 2).type(torch.float32)
        return ProbabilityMeasure(prob, coord)

    @staticmethod
    def from_coord_tensor(data: Tensor):
        prob = torch.ones(data.shape[0], data.shape[1], dtype=torch.float32, device=data.device)
        pad_inds = data < -9999
        prob[pad_inds[:, :, 0]] = 0
        data[pad_inds] = 0
        return ProbabilityMeasure(prob / prob.sum(dim=1, keepdim=True), data)

    def save(self, path: str, measure: ProbabilityMeasure):
        np.save(path + "_coord", measure.coord.detach().cpu().numpy())
        np.save(path + "_prob", measure.probability.detach().cpu().numpy())

    def load(self, path: str) -> ProbabilityMeasure:
        coord = torch.from_numpy(np.load(path + "_coord.npy"))
        # prob = torch.ones(coord.shape[0], coord.shape[1]) / coord.shape[1]
        prob = torch.from_numpy(np.load(path + "_prob.npy"))
        return ProbabilityMeasure(prob, coord)


class UniformMeasure2DFactory:

    @staticmethod
    def from_heatmap(hm: Tensor) -> UniformMeasure2D01:
        coord, _ = heatmap_to_measure(hm)
        return UniformMeasure2D01(coord)

    @staticmethod
    def load(path: str) -> UniformMeasure2D01:
        coord = torch.from_numpy(np.load(path + "_coord.npy"))
        return UniformMeasure2D01(coord)
