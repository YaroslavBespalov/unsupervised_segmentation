from torch import nn
import torch
from dataset.probmeasure import ProbabilityMeasure

from torch import Tensor
import time

from loss.losses import Samples_Loss
from gan.loss_base import Loss


class PairwiseDistance(nn.Module):

    def forward(self, x: Tensor, y: Tensor):
        n = x.shape[1]
        assert y.shape[1] == n
        x = x[:, :, None, :]
        y = y[:, None, :, :]
        x = torch.cat([x] * n, dim=2)
        y = torch.cat([y] * n, dim=1)
        return (x - y).pow(2).sum(dim=-1)


class L2Norm2(nn.Module):
    def forward(self, pred: ProbabilityMeasure, targets: ProbabilityMeasure):
        assert torch.all(torch.eq(pred.probability, targets.probability)).item() == 1
        return Loss(((pred.coord - targets.coord).pow(2).sum(-1) * pred.probability).sum(dim=-1).mean())


class LinearTransformOT:

    @staticmethod
    def forward(pred: ProbabilityMeasure, targets: ProbabilityMeasure, iters: int = 200):
        lambd = 0.002

        with torch.no_grad():
            P = SOT(iters, lambd).forward(pred.centered(), targets.centered())

        xs = pred.centered().coord
        xsT = xs.transpose(1, 2)
        xt = targets.centered().coord

        a = pred.probability + 1e-8
        a /= a.sum(dim=1, keepdim=True)
        a = a.reshape(a.shape[0], -1, 1)

        A = torch.inverse(xsT.bmm(a * xs)).bmm(xsT.bmm(P.bmm(xt)))

        T = targets.mean() - pred.mean()

        return A.cuda().type_as(pred.coord), T.detach()


class SOT(nn.Module):
    def __init__(self, max_iters: int=100, reg=1e-2):
        super(SOT, self).__init__()
        self.pdist = PairwiseDistance()
        self.reg = reg
        self.max_iters = max_iters

    def sinkhorn(self, a: Tensor, b: Tensor, M: Tensor):

        device = a.device
        # init data
        dim_a = a.shape[1]
        dim_b = b.shape[1]
        batch_size = a.shape[0]
        assert a.shape[0] == b.shape[0]
        a = a.view(batch_size, dim_a, 1).type(torch.float64)
        b = b.view(batch_size, dim_b, 1).type(torch.float64)

        u = torch.ones((batch_size, dim_a, 1), device=device, dtype=torch.float64) / dim_a
        v = torch.ones((batch_size, dim_b, 1), device=device, dtype=torch.float64) / dim_b

        K = torch.exp(-M.type(torch.float64) / self.reg)
        Kt = K.transpose(1, 2)

        cpt = 0

        P = (u.reshape((batch_size, -1, 1)) * K * v.reshape((batch_size, 1, -1)))

        while cpt < self.max_iters:
            uprev = u
            vprev = v

            KtU = torch.bmm(Kt, u)
            v = b / KtU

            KV = K.bmm(v)
            u = a / KV

            if (torch.any(KtU == 0)
                    or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                    or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
                print('Warning: numerical errors at iteration', cpt)
                u = uprev
                v = vprev
                break

            cpt = cpt + 1

            if cpt % 10 == 0:
                P_new = (u.reshape((batch_size, -1, 1)) * K * v.reshape((batch_size, 1, -1)))
                if (P - P_new).abs().max() < 0.001:
                    P = P_new
                    break
                else:
                    P = P_new

        return P.type(torch.float32)

    def forward(self, m1: ProbabilityMeasure, m2: ProbabilityMeasure):

        M = self.pdist(m1.coord, m2.coord)
        M = M / M.max(dim=1)[0].max(dim=1)[0].view(-1, 1, 1)
        a = m1.probability + 1e-8
        a /= a.sum(dim=1, keepdim=True)
        b = m2.probability + 1e-8
        b /= b.sum(dim=1, keepdim=True)
        P = self.sinkhorn(a, b, M)

        return P
