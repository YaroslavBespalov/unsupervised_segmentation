from typing import List, Dict

import albumentations
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn, optim
from torchvision import utils

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from dataset.cardio_dataset import SegmentationDataset, ImageMeasureDataset
from dataset.probmeasure import ProbabilityMeasure, ProbabilityMeasureFabric
from framework.gan.dcgan.discriminator import ResDCDiscriminator, DCDiscriminator
from framework.gan.gan_model import GANModel, ConditionalGANModel
from framework.gan.loss.hinge import HingeLoss
from framework.gan.loss.wasserstein import WassersteinLoss
from framework.gan.noise.normal import NormalNoise
from framework.gan.noise.normalize import Normalization
from loss.losses import Samples_Loss
from modules.gen_dis_style import GeneratorWithStyle, StepScale, DiscriminatorWithStyle
from modules.image2measure import ResImageToMeasure
from metrics.collector import MetricCollector
import torchvision.datasets as dset
import os

from modules.linear_ot import LinearTransformOT
from modules.measure2image import ResMeasureToImage, MeasureToImage
from albumentations import ShiftScaleRotate

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

image_size = 256
batch_size = 16
measure_size = 70

full_dataset = ImageMeasureDataset(
    "/raid/data/celeba",
    "/raid/data/celeba_masks",
    img_transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
)

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [len(full_dataset) - 500, 500])

dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
noise = NormalNoise(100, device)

# measure2image = MeasureToImage(noise.size() + 3 * measure_size, image_size, 64).cuda()

# netD = DCDiscriminator(ndf=64).cuda()
# gan_model = GANModel(measure2image,
#                      WassersteinLoss(netD, penalty_weight=10).add_generator_loss(nn.L1Loss(), 1),
#                      lr=0.0003)

measure2image: GeneratorWithStyle = GeneratorWithStyle(alpha=0.5, in_dim=noise.size() + 3 * measure_size)
netD: DiscriminatorWithStyle = DiscriminatorWithStyle(alpha=0.5)
measure2image.set_step(1)
netD.set_step(1)

gan_model = ConditionalGANModel(measure2image,
                     HingeLoss(netD),
                     lr=0.0005,
                     do_init_ws=False)


step = 1
max_step = 5
stepScale = StepScale()

fabric = ProbabilityMeasureFabric(image_size)

barycenter = fabric.load("face_barycenter").cuda().padding(measure_size)
barycenter = fabric.cat([barycenter for b in range(batch_size)])
print(barycenter.coord.shape)
plt.imshow(barycenter.detach().toImage(100)[0][0].cpu())
plt.show()

image2measure = ResImageToMeasure(measure_size).cuda()
image2measure_opt = optim.Adam(image2measure.parameters(), lr=0.0001)


def test():
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=20)

    sum_loss = 0
    with torch.no_grad():
        for i, (imgs, measures) in enumerate(dataloader_test, 0):
            imgs = imgs.cuda()
            pred_measures: ProbabilityMeasure = image2measure(imgs)
            ref_measures: ProbabilityMeasure = ProbabilityMeasureFabric.from_coord_tensor(measures).cuda().padding(measure_size)
            ref_loss = Samples_Loss()(pred_measures, ref_measures)
            sum_loss += ref_loss.item()

    print(sum_loss / len(test_dataset))

for epoch in range(1000):
    print("epoch: ", epoch)

    for i, (imgs, masks) in enumerate(dataloader, 0):

        if imgs.shape[0] != batch_size:
            continue

        if i % 5000 == 0:
            step += 1
            step = min([step, max_step])
            print("step: ", step)
            measure2image.set_step(step)
            netD.set_step(step)

            gan_model = ConditionalGANModel(measure2image,
                                 HingeLoss(netD).add_generator_loss(nn.L1Loss(), 2.0),
                                 lr=0.0003 * (0.95 **step),
                                 do_init_ws=False)

        imgs = imgs.cuda()
        pred_measures: ProbabilityMeasure = image2measure(imgs)

        imgs_scaled = stepScale.forward(imgs, step)

        z = noise.sample(imgs.shape[0])
        cond = pred_measures.toChannels()
        zc = torch.cat((z, cond), dim=1)

        gan_model.train(imgs_scaled, pred_measures.toImage(imgs_scaled.shape[-1]).detach(), zc.detach())

        if i > 100:
            fake = measure2image(zc)
            g_loss = gan_model.generator_loss(imgs_scaled, fake, pred_measures.toImage(imgs_scaled.shape[-1]).detach())
            (g_loss * 0.01).minimize_step(image2measure_opt)
        zc = zc.detach()

        pred_measures: ProbabilityMeasure = image2measure(imgs)

        # with torch.no_grad():
            # A, T = LinearTransformOT.forward(pred_measures, barycenter)
        # bc_loss = Samples_Loss()(pred_measures, pred_measures.detach() + T) * 0.0001 + \
        #           Samples_Loss()(pred_measures.centered(), pred_measures.centered().multiply(A).detach()) * 0.0002 + \
        bc_loss = Samples_Loss()(pred_measures, barycenter) * 0.01

        bc_loss.minimize_step(image2measure_opt)

        if i % 100 == 0:
            print(i)
            test()
            with torch.no_grad():
                plt.imshow(pred_measures.detach().toImage(100)[0][0].cpu())
                plt.show()
                fake = measure2image(zc)
                utils.save_image(
                    fake[:9],
                    f'../sample/{str(i).zfill(6)}.png',
                    nrow=3,
                    normalize=True,
                    range=(-1, 1),
                )

