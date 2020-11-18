import torch
from torch import nn, Tensor

from dataset.lazy_loader import LazyLoader
from dataset.probmeasure import UniformMeasure2D01
from modules.linear_ot import SOT, PairwiseDistance


def handmadew1(m1,m2, lambd=0.0005):
    with torch.no_grad():
        P = SOT(200, lambd).forward(m1, m2)
        M = PairwiseDistance()(m1.coord, m2.coord).sqrt()
        main_diag = (torch.diagonal(M, offset=0, dim1=1, dim2=2) * torch.diagonal(P, offset=0, dim1=1, dim2=2))
        # perm = P.argmax(dim=-1)[0]
        # res = (m1.coord[0] - m2.coord[0, perm]).pow(2).sum(-1).sqrt().mean()
        # res1 = ((M * P).sum(dim=(1,2)) + main_diag.sum(dim=1)) / 2
        # print(res, res1[0])

    # return ((M * P).sum(dim=(1,2)) + main_diag.sum(dim=1)) / 2
    return (M * P).sum(dim=(1, 2))


def liuboff(encoder: nn.Module):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].cuda()
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        landmarks[landmarks > 1] = 0.99999

        pred_measure = UniformMeasure2D01(encoder(data)["coords"])
        target = UniformMeasure2D01(torch.clamp(landmarks, max=1))

        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()

        sum_loss += (handmadew1(pred_measure, target) / eye_dist).sum().item()

    return sum_loss / len(LazyLoader.w300().test_dataset)


def liuboffMAFL(encoder: nn.Module):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.mafl().test_loader):
        data = batch['data'].cuda()
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        landmarks[landmarks > 1] = 0.99999

        pred_measure = UniformMeasure2D01(encoder(data)["coords"])
        target = UniformMeasure2D01(torch.clamp(landmarks, max=1))

        eye_dist = landmarks[:, 1] - landmarks[:, 0]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()

        sum_loss += (handmadew1(pred_measure, target, 0.005) / eye_dist).sum().item()

    return sum_loss / len(LazyLoader.mafl().test_dataset)