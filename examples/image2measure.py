from typing import List
import numpy as np
import albumentations
import matplotlib
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn
from torch import optim

from dataset.cardio_dataset import SegmentationDataset
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from framework.loss import Loss
from loss.losses import Samples_Loss, linear_deformation, Tshift_linear_loss
from modules.image2measure import ResImageToMeasure
from modules.linear_ot import LinearTransformOT

image_size = 256
batch_size = 40
padding = 200
#/home/nazar/PycharmProjects/mrt
dataset = SegmentationDataset("/home/nazar/PycharmProjects/mrt",
                              transform=albumentations.Compose([
                          albumentations.Resize(image_size, image_size),
                          albumentations.CenterCrop(image_size, image_size),
                          albumentations.Normalize(),
                          ToTensorV2()
                      ]),
                              )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

model = ResImageToMeasure(padding, ndf=16).cuda()
opt = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

for epoch in range(100):
    for i, (imgs, masks) in enumerate(dataloader, 0):

        imgs = imgs.cuda()
        measures: ProbabilityMeasure = ProbabilityMeasureFabric(image_size).from_mask(masks).cuda().padding(padding)

        pred: ProbabilityMeasure = model(imgs)

        loss = Loss(Samples_Loss()(pred, measures))
        print(loss.item())
        loss.minimize_step(opt)




