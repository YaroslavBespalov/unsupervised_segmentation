from typing import List, Dict
import numpy as np
import albumentations
import matplotlib
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
#from albumentations.torch import ToTensor as ToTensorV2
from torch import Tensor, nn
from torch import optim
from torchvision import utils

from dataset.cardio_dataset import SegmentationDataset, MRIImages
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from framework.loss import Loss
from framework.gan.dcgan.discriminator import DCDiscriminator, ResDCDiscriminator, ConditionalDCDiscriminator
from framework.gan.gan_model import GANModel, ConditionalGANModel
from framework.gan.image2image.discriminator import Discriminator
from framework.gan.loss.hinge import HingeLoss
from framework.gan.loss.wasserstein import WassersteinLoss
from framework.gan.noise.Noise import Noise
from framework.gan.noise.normal import NormalNoise
from framework.gan.noise.normalize import Normalization
from framework.module import NamedModule
from modules.image2measure import ResImageToMeasure
from modules.linear_ot import LinearTransformOT
from modules.measure2image import MeasureToImage, ResMeasureToImage
from loss.losses import Samples_Loss, linear_deformation
from metrics.collector import MetricCollector
from framework.gan.cycle.model import CycleGAN

import os
import cv2


image_size = 256
batch_size = 4
padding = 200

device = torch.device("cuda")

if not os.path.exists(os.path.join(f"./samples{batch_size}/")):
            os.makedirs(os.path.join(f"./samples{batch_size}/"))

#/home/nazar/PycharmProjects/mrt
dataset = SegmentationDataset("/home/nazar/PycharmProjects/mrt",
                              transform=albumentations.Compose([
                          albumentations.Resize(image_size, image_size),
                          albumentations.CenterCrop(image_size, image_size),
                          ToTensorV2()
                      ]),
                              )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

fabric = ProbabilityMeasureFabric(image_size)

noise = Normalization(NormalNoise(10, device))
measure2image = ResMeasureToImage(padding * 3 + noise.size(), image_size, 48).cuda()
image2measure = ResImageToMeasure(padding).cuda()


netD = DCDiscriminator(ndf=160).cuda()
gan_model = GANModel(measure2image,
                     HingeLoss(netD).add_generator_loss(nn.L1Loss(), 1),
                     lr=0.0004)

cycle_gan = CycleGAN[ProbabilityMeasure, Tensor](
    Measure2imgTmp(measure2image, noise),
    NamedModule(image2measure, ['image'], ['measure']),
    {'measure': Samples_Loss().forward}, {'image': lambda x, y: Loss(nn.L1Loss().forward(x, y))}
)



all_losses = MetricCollector()
all_losses.preprocessing(['measure'], gan_model.generator.forward, "fake")
all_losses.add_metric(['real', 'fake'], nn.L1Loss().forward, "L1")
all_losses.add_metric(['real', 'fake'], gan_model.generator_loss, "generator loss")
all_losses.add_metric(['real', 'fake'], gan_model.discriminator_loss, "discriminator loss")

for epoch in range(100):

    print("epoch: ", epoch)

    for i, (imgs, masks) in enumerate(dataloader, 0):

        imgs = imgs.cuda().type(torch.float32)
        measures: ProbabilityMeasure = fabric.from_mask(masks).cuda().padding(padding)

        cond = measures.toChannels()
        z = noise.sample(cond.shape[0])
        cond = torch.cat((cond, z), dim=1)

        gan_model.train(imgs, cond.detach())

        cycle_gan.train({'measure': measures.detach()}, {'image': imgs})

        all_losses.add({'real': imgs, 'measure': cond})

    all_losses.print_metrics()
    all_losses.clear_cash()









    with torch.no_grad():
        fake = measure2image(cond)
        # plt.imshow(np.transpose(fake[0].cpu().numpy(), (1, 2, 0)))
        # plt.show()
        # f_images = np.transpose(fake[j].cpu().numpy(), (1, 2, 0))
        # cv2.imwrite(f'./samples/fimg_{str(epoch + 1).zfill(6)}.tiff', f_images)
        utils.save_image(
            fake,
            f'./samples{batch_size}/fimg_{str(epoch + 1).zfill(6)}.tiff',
            nrow=batch_size // 2,
            normalize=True,
            range=(-1, 1),
        )




