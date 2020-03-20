from typing import List
import numpy as np
import albumentations
import matplotlib
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn
from torch import optim
from torchvision import utils
from dataset.cardio_dataset import SegmentationDataset, MRIImages
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from framework.loss import Loss
from framework.gan.cycle.model import CycleGAN
from framework.gan.dcgan.discriminator import DCDiscriminator, ResDCDiscriminator, ConditionalDCDiscriminator
from framework.gan.gan_model import GANModel, ConditionalGANModel
from framework.gan.image2image.discriminator import Discriminator
from framework.gan.loss.hinge import HingeLoss
from framework.gan.loss.wasserstein import WassersteinLoss
from framework.gan.noise.normal import NormalNoise
from framework.gan.noise.normalize import Normalization
from framework.module import NamedModule
from modules.image2measure import ResImageToMeasure
from modules.linear_ot import LinearTransformOT, SOT
from modules.measure2image import MeasureToImage, ResMeasureToImage, Measure2imgTmp
from loss.losses import Samples_Loss, linear_deformation

import os
import cv2

image_size = 256
batch_size = 4
padding = 200

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

if not os.path.exists(os.path.join("./segm_samples/")):
    os.makedirs(os.path.join("./segm_samples/"))

# /home/ibespalov/GANS_NAZAR/dataset/postprocessing
dataset = SegmentationDataset("/home/nazar/PycharmProjects/mrt",
                              transform=albumentations.Compose([
                               albumentations.Resize(image_size, image_size),
                               albumentations.CenterCrop(image_size, image_size),
                               ToTensorV2()
                           ]),
                              )


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)


fabric = ProbabilityMeasureFabric(image_size)
barycenter = fabric.load("barycenter").cuda().padding(padding)
print(barycenter.coord.shape)
barycenter = fabric.cat([barycenter for b in range(batch_size)])
print(barycenter.coord.shape)


for i, (imgs, masks) in enumerate(dataloader, 0):
    imgs = imgs.cuda().type(torch.float32)
    measures: ProbabilityMeasure = fabric.from_mask(masks).cuda().padding(padding)

    P = SOT().forward(measures, barycenter)
