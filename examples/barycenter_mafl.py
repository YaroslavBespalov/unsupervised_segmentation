# %%

import json
import sys, os

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from torch.distributions import Dirichlet
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.replay_data import ReplayBuffer

sys.path.append(os.path.join(sys.path[0], '/home/ibespalov/unsupervised_pattern_segmentation/'))
sys.path.append(os.path.join(sys.path[0], '/home/ibespalov/unsupervised_pattern_segmentation/gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '/home/ibespalov/unsupervised_pattern_segmentation/gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '/home/ibespalov/unsupervised_pattern_segmentation/gans_pytorch/gan/'))

from typing import List, Tuple
import numpy as np
import albumentations
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
# from albumentations.torch import ToTensor as ToTensorV2
from torch import Tensor, nn
from torch import optim
from dataset.cardio_dataset import SegmentationDataset
from dataset.lazy_loader import LazyLoader, MAFL, W300DatasetLoader
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure, UniformMeasure2D01
from dataset.toheatmap import heatmap_to_measure
from loss.losses import Samples_Loss, WeightedSamplesLoss
from modules.hg import HG_softmax2020
from parameters.path import Paths

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

image_size = 256
batch_size = 64
padding = 68
MAFL.batch_size = batch_size
W300DatasetLoader.batch_size = batch_size

NC = 256

bc_net = nn.Sequential(
    nn.Linear(batch_size, NC),
    nn.ReLU(inplace=True),
    nn.Linear(NC, NC),
    nn.ReLU(inplace=True),
    nn.Linear(NC, NC),
    nn.ReLU(inplace=True),
    nn.Linear(NC, NC),
    nn.ReLU(inplace=True),
    nn.Linear(NC, padding * 2),
    nn.Sigmoid()
).cuda()

bc_net_opt = optim.Adam(bc_net.parameters(), lr=0.001)
sced = ReduceLROnPlateau(bc_net_opt)


replay_buf = ReplayBuffer(2)


def compute_wbc(measures: ProbabilityMeasure, weights: Tensor, opt_iters: int) -> Tuple[ProbabilityMeasure, float]:

    fabric = ProbabilityMeasureFabric(image_size)
    barycenter: ProbabilityMeasure = fabric.random(padding).cuda()
    barycenter.coord = bc_net(weights[None, :]).reshape(1, padding, 2).detach()
    barycenter.requires_grad_()

    coord = barycenter.coord
    opt = optim.Adam(iter([coord]), lr=0.0005)

    for _ in range(opt_iters):
        barycenter_cat = fabric.cat([barycenter] * batch_size)

        loss = WeightedSamplesLoss(weights)(barycenter_cat, measures)
        # print(loss.item())

        opt.zero_grad()
        loss.to_tensor().backward()
        opt.step()

        barycenter.probability.data = barycenter.probability.relu().data
        barycenter.probability.data /= barycenter.probability.sum(dim=1, keepdim=True)

    replay_buf.append(weights.cpu().detach()[None, :], barycenter.coord.cpu().detach())

    lll = 10

    if replay_buf.size() > 32:
        ws, bs = replay_buf.sample(32)

        bc_net.zero_grad()
        ll = (bc_net(ws).reshape(-1, padding, 2) - bs).pow(2).sum() / 32
        print(ll.item())
        lll = ll.item()
        ll.backward()
        bc_net_opt.step()

    # replay_buf = replay_buf[replay_buf.__len__() - 32:]

    barycenter.coord = bc_net(weights[None, :]).reshape(1, padding, 2).detach()
    return barycenter, lll


# mafl_dataloader = LazyLoader.w300().loader_train_inf
# mes = UniformMeasure2D01(next(mafl_dataloader)['meta']['keypts_normalized'].type(torch.float32)).cuda()
mes = UniformMeasure2D01(next(iter(LazyLoader.celeba_test(batch_size)))[1]).cuda()


for j in range(10000):
    weights = Dirichlet(torch.ones(batch_size)/10).sample().cuda()
    barycenter, lll = compute_wbc(mes, weights, min(200, j + 10))

    if j % 50 == 0:
       print(j)
       sced.step(lll)

    if j % 50 == 0:
        plt.imshow(barycenter.toImage(200)[0][0].detach().cpu().numpy())
        plt.show()

    starting_model_number = 0
    if j % 1000 == 0 and j > 0:
        torch.save(
            bc_net.state_dict(),
            f'{Paths.default.models()}/bc_model256_64_{str(j + starting_model_number).zfill(6)}.pt',
        )

# plt.imshow(barycenter.toImage(200)[0][0].detach().cpu().numpy())
# plt.show()

# fabric.save("./face_barycenter_5", barycenter)



