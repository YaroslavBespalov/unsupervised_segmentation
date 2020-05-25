from typing import List
import numpy as np
import albumentations
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
#from albumentations.torch import ToTensor as ToTensorV2
from torch import Tensor, nn
from torch import optim
from dataset.cardio_dataset import SegmentationDataset
from dataset.lazy_loader import LazyLoader
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from dataset.toheatmap import heatmap_to_measure
from loss.losses import Samples_Loss
from modules.hg import HG_softmax2020
from parameters.path import Paths

image_size = 256
batch_size = 8
padding = 68

fabric = ProbabilityMeasureFabric(image_size)
barycenter: ProbabilityMeasure = fabric.random(padding).cuda()
barycenter.requires_grad_()

coord = barycenter.coord

opt = optim.Adam(iter([coord]), lr=0.0006)

encoder_HG = HG_softmax2020(num_classes=68, heatmap_size=64)
encoder_HG.load_state_dict(torch.load(f"{Paths.default.models()}/hg2_e29.pt", map_location="cpu"))
encoder_HG = encoder_HG.cuda()

for iter in range(3000):

    img = next(LazyLoader.celeba().loader).cuda()
    content = encoder_HG(img)
    coord, p = heatmap_to_measure(content)
    mes = ProbabilityMeasure(p, coord)

    barycenter_cat = fabric.cat([barycenter] * batch_size)

    loss = Samples_Loss()(barycenter_cat, mes)

    opt.zero_grad()
    loss.to_tensor().backward()
    opt.step()

    barycenter.probability.data = barycenter.probability.relu().data
    barycenter.probability.data /= barycenter.probability.sum(dim=1, keepdim=True)

    if iter % 100 == 0:
        print(iter, loss.item())

        plt.imshow(barycenter.toImage(200)[0][0].detach().cpu().numpy())
        plt.show()

    # if iter % 1000 == 0:
        # fabric.save("face_barycenter_68", barycenter)
