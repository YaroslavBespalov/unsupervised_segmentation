import argparse
import time
from itertools import chain
from typing import Callable, Any, List
import sys
import os

from metrics.writers import send_images_to_tensorboard, WR
from modules.nashhg import HG_skeleton
from train_procedure import content_trainer_supervised
from viz.image_with_mask import imgs_with_mask

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../dataset'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from dataset.toheatmap import ToHeatMap, heatmap_to_measure, CoordToGaussSkeleton, ToGaussHeatMap

import albumentations
import torch
from torch import optim
from torch import nn, Tensor
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from dataset.lazy_loader import LazyLoader, W300DatasetLoader
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure, UniformMeasure2D01

from matplotlib import pyplot as plt

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

encoder_HG = HG_skeleton(CoordToGaussSkeleton(256, 1)).cuda()

writer = SummaryWriter(f"/home/ibespalov/pomoika/w300{int(time.time())}")


def test():
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        content = encoder_HG(data)["coords"]
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((content - landmarks).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    return sum_loss / len(LazyLoader.w300().test_dataset)


cont_opt = optim.Adam(encoder_HG.parameters(), lr=3e-5, betas=(0.5, 0.97))

W300DatasetLoader.batch_size = 16
W300DatasetLoader.test_batch_size = 32

supervise_trainer = content_trainer_supervised(cont_opt, encoder_HG, LazyLoader.w300().loader_train_inf)

test_img = next(LazyLoader.w300().loader_train_inf)['data'].cuda()

for i in range(10000):
    supervise_trainer()

    if i % 100 == 0:
        with torch.no_grad():
            liuboff = test()
            print(liuboff)

            WR.writer.add_scalar("liuboff", liuboff, i)

            encoded_test = encoder_HG(test_img)
            pred_measures_test: UniformMeasure2D01 = UniformMeasure2D01(encoded_test["coords"])
            heatmaper_256 = ToGaussHeatMap(256, 1.0)
            sparse_hm_test_1 = heatmaper_256.forward(pred_measures_test.coord)

            sparce_mask = sparse_hm_test_1.sum(dim=1, keepdim=True)
            sparce_mask[sparce_mask < 0.0003] = 0
            iwm = imgs_with_mask(test_img, sparce_mask)
            send_images_to_tensorboard(WR.writer, iwm, "REAL", i)


