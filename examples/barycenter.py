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
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from loss.losses import Samples_Loss
image_size = 256
batch_size = 4
padding = 70

device = torch.device("cuda")

dataset = SegmentationDataset(
    "/raid/data/celeba",
    "/raid/data/celeba_masks",
    transform_joint=albumentations.Compose([
                               albumentations.Resize(image_size, image_size),
                               albumentations.CenterCrop(image_size, image_size),
                               ToTensorV2()
    ]),
    # img_transform=None
    img_transform=torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
)

fabric = ProbabilityMeasureFabric(image_size)
barycenter: ProbabilityMeasure = fabric.random(padding).cuda()
barycenter.requires_grad_()

coord = barycenter.coord

opt = optim.Adam(iter([coord]), lr=0.001)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)


for iter in range(20):

    # loss_sum = Loss.ZERO()

    for i, (imgs, masks) in enumerate(dataloader, 0):

        barycenter_cat = fabric.cat([barycenter] * batch_size)

        mi = fabric.from_mask(masks).cuda()
        loss = Samples_Loss()(barycenter_cat, mi)

        opt.zero_grad()
        loss.to_tensor().backward()
        opt.step()

        barycenter.probability.data = barycenter.probability.relu().data
        barycenter.probability.data /= barycenter.probability.sum(dim=1, keepdim=True)

        if i % 20 == 0:
            print(i)
            print(iter, loss.item())

            plt.imshow(barycenter.toImage(200)[0][0].detach().cpu().numpy())
            plt.show()

        if i == 3000:
            fabric.save("face_barycenter", barycenter)
