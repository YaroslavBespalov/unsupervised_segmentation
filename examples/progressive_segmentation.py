from typing import List, Dict

import cv2
import numpy as np
import albumentations
import matplotlib
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn, optim
from torchvision import utils

from dataset.cardio_dataset import SegmentationDataset, MRIImages
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from framework.gan.gan_model import GANModel
from framework.gan.loss.hinge import HingeLoss
from framework.gan.loss.wasserstein import WassersteinLoss
from framework.gan.noise.normal import NormalNoise
from framework.gan.noise.normalize import Normalization
from loss.losses import Samples_Loss
from modules.gen_dis_style import GeneratorWithStyle, StepScale, DiscriminatorWithStyle
from modules.image2measure import ResImageToMeasure
from metrics.collector import MetricCollector

import os

from modules.linear_ot import LinearTransformOT

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

image_size = 256
batch_size = 16
padding = 600

device = torch.device("cuda")

if not os.path.exists(os.path.join(f"./samples{batch_size}/")):
            os.makedirs(os.path.join(f"./samples{batch_size}/"))

#/home/nazar/PycharmProjects/mrt
dataset_test = SegmentationDataset("/home/ibespalov/GANS_NAZAR/dataset/postprocessing",
                                   transform=albumentations.Compose([
                          # albumentations.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=1,
                          #                                 border_mode=cv2.BORDER_CONSTANT),
                          albumentations.Resize(image_size, image_size),
                          albumentations.CenterCrop(image_size, image_size),
                          ToTensorV2()
                      ]),
                                   )

dataset = MRIImages("/raid/data/unsup_segment/samples_without_masks",
                    transform=albumentations.Compose([
                        albumentations.OpticalDistortion(border_mode=0),
                        albumentations.ElasticTransform(border_mode=0),
                        albumentations.Resize(image_size, image_size),
                        albumentations.CenterCrop(image_size, image_size),
                        ToTensorV2()
                    ]),
                    )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

fabric = ProbabilityMeasureFabric(image_size)

barycenter = fabric.load("barycenter").cuda().padding(padding)
barycenter = fabric.cat([barycenter for b in range(40)])
print(barycenter.coord.shape)
image2measure = ResImageToMeasure(padding).cuda()
image2measure_opt = optim.Adam(image2measure.parameters(), lr=0.0002)

noise = NormalNoise(10, device)

measure2image: GeneratorWithStyle = GeneratorWithStyle(alpha=0.7, in_dim=10 + padding * 3)
netD: DiscriminatorWithStyle = DiscriminatorWithStyle(alpha=0.7)
measure2image.set_step(1)
netD.set_step(1)

gan_model = None


def test():
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=40, num_workers=20)

    sum_loss = 0
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(dataloader_test, 0):
            imgs = imgs.cuda().type(torch.float32)
            pred_measures: ProbabilityMeasure = image2measure(imgs)
            ref_measures: ProbabilityMeasure = fabric.from_mask(masks).cuda().padding(padding)
            ref_loss = Samples_Loss()(pred_measures, ref_measures)
            sum_loss += ref_loss.item()

    print(sum_loss)

step = 0
max_step = 5
stepScale = StepScale()
for epoch in range(1000):

    print("epoch: ", epoch)
    if epoch % 10 == 0:

        step += 1
        step = min([step, max_step])
        print("step: ", step)
        measure2image.set_step(step)
        netD.set_step(step)

        gan_model = GANModel(measure2image,
                             HingeLoss(netD).add_generator_loss(nn.L1Loss(), 1),
                             lr=0.0004 * (0.95 ** step) * (0.99 ** epoch),
                             do_init_ws=False)

    for i, imgs in enumerate(dataloader, 0):
        imgs = imgs.cuda().type(torch.float32)
        test_imgs = imgs

        pred_measures: ProbabilityMeasure = image2measure(imgs)
        cond = pred_measures.toChannels()
        z = noise.sample(cond.shape[0])
        cond = torch.cat((cond, z), dim=1)

        imgs = stepScale.forward(imgs, step)

        gan_model.train(imgs, cond.detach())

        n = cond.shape[0]
        barycenter_batch = barycenter.slice(0, n)

        with torch.no_grad():
            A, T = LinearTransformOT.forward(pred_measures, barycenter_batch, 100)

        bc_loss_T = Samples_Loss()(pred_measures, pred_measures.detach() + T)
        bc_loss_A = Samples_Loss()(pred_measures.centered(), pred_measures.centered().multiply(A).detach())
        bc_loss_W = Samples_Loss()(pred_measures.centered().multiply(A), barycenter_batch.centered())
        bc_loss = bc_loss_W * 0.002 + bc_loss_A * 0.0008 + bc_loss_T * 0.0001

        fake = measure2image(cond)
        g_loss = gan_model.generator_loss(imgs, fake)
        cg = min(epoch/10, 1) / 500
        (g_loss * cg + bc_loss).minimize_step(image2measure_opt)

    # all_losses.print_metrics()
    # all_losses.clear_cash()

    with torch.no_grad():
        fake = measure2image(cond)
        plt.imshow(np.transpose(fake[0].cpu().numpy(), (1, 2, 0)))
        plt.show()
        pred: ProbabilityMeasure = image2measure(test_imgs)
        f_mask = pred.toImage(100)[0][0].cpu().numpy()
        # cv2.imwrite(f'./segm_samples/mask_{str(epoch + 1).zfill(6)}.tiff', f_mask)
        plt.imshow(f_mask)
        plt.show()

    test()




