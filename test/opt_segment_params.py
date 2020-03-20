import argparse
import time
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
from framework.nn.ops.pairwise import L2Norm2
from loss.losses import Samples_Loss
from modules.image2measure import ResImageToMeasure
from modules.linear_ot import LinearTransformOT, LinearTransformOT_bk
from modules.measure2image import MeasureToImage, ResMeasureToImage, Measure2imgTmp
import os
import cv2

from parameters.dataset import DatasetParameters
from parameters.deformation import DeformationParameters
from parameters.gan import GanParameters

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    parents=[
        DatasetParameters(),
        GanParameters(),
        DeformationParameters()
    ]
)
args = parser.parse_args()
for k in vars(args):
    print(f"{k}: {vars(args)[k]}")


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)


dataset_test = SegmentationDataset("/home/nazar/PycharmProjects/mrt",
                                   transform=albumentations.Compose([
                          albumentations.Resize(args.image_size, args.image_size),
                          albumentations.CenterCrop(args.image_size, args.image_size),
                          ToTensorV2()
                      ]),
                                   )

dataset = MRIImages("/raid/data/unsup_segment/samples_without_masks",
                      transform=albumentations.Compose([
                          albumentations.Resize(args.image_size, args.image_size),
                          albumentations.CenterCrop(args.image_size, args.image_size),
                          ToTensorV2()
                      ]),
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)



def optimization_step():
    noise = NormalNoise(n_noise, device)
    measure2image = ResMeasureToImage(args.measure_size * 3 + noise.size(), args.image_size, ngf).cuda()

    netD = DCDiscriminator(ndf=ndf).cuda()
    gan_model = GANModel(measure2image,
                         HingeLoss(netD).add_generator_loss(nn.L1Loss(), L1),
                         lr=0.0004)

    fabric = ProbabilityMeasureFabric(args.image_size)
    barycenter = fabric.load("barycenter").cuda().padding(args.measure_size)
    print(barycenter.coord.shape)
    barycenter = fabric.cat([barycenter for b in range(args.batch_size)])
    print(barycenter.coord.shape)

    image2measure = ResImageToMeasure(args.measure_size).cuda()
    image2measure_opt = optim.Adam(image2measure.parameters(), lr=0.0002)

    def test():
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=40, num_workers=20)

        sum_loss = 0

        for i, (imgs, masks) in enumerate(dataloader_test, 0):
            imgs = imgs.cuda().type(torch.float32)
            pred_measures: ProbabilityMeasure = image2measure(imgs)
            ref_measures: ProbabilityMeasure = fabric.from_mask(masks).cuda().padding(args.measure_size)
            ref_loss = Samples_Loss()(pred_measures, ref_measures)
            sum_loss += ref_loss.item()

        return sum_loss

    for epoch in range(20):

        ot_iters = 100
        print("epoch",  epoch)
        test_imgs = None


        for i, imgs in enumerate(dataloader, 0):

            imgs = imgs.cuda().type(torch.float32)
            test_imgs = imgs
            pred_measures: ProbabilityMeasure = image2measure(imgs)
            cond = pred_measures.toChannels()
            n = cond.shape[0]
            barycenter_batch = barycenter.slice(0, n)

            z = noise.sample(n)
            cond = torch.cat((cond, z), dim=1)
            gan_model.train(imgs, cond.detach())

            with torch.no_grad():
                A, T = LinearTransformOT.forward(pred_measures, barycenter_batch, ot_iters)

            bc_loss_T = Samples_Loss()(pred_measures, pred_measures.detach() + T)
            bc_loss_A = Samples_Loss()(pred_measures.centered(), pred_measures.centered().multiply(A).detach())
            bc_loss_W = Samples_Loss()(pred_measures.centered().multiply(A), barycenter_batch.centered())
            bc_loss = bc_loss_W * cw + bc_loss_A * ca + bc_loss_T * ct

            fake = measure2image(cond)
            g_loss = gan_model.generator_loss(imgs, fake)
            (g_loss + bc_loss).minimize_step(image2measure_opt)

    return test()

n_noise, L1, ngf, ndf, ct, ca, cw = 10, 3, 64, 160, 0.0003, 0.001, 0.003

t1 = time.time()
Loss = 1000
L1_final = 3
for L1 in [1, 2, 3, 5, 10, 20]:
    New_Loss = optimization_step()
    if New_Loss < Loss:
        Loss = New_Loss
        L1_final = L1
L1 = L1_final
print(f" L1 time: {time.time() - t1}")


t1 = time.time()
Loss = 1000
ct_final = 0.0003
for ct in [0.0003, 0.0001, 0.00001]:
    New_Loss = optimization_step()
    if New_Loss < Loss:
        Loss = New_Loss
        ct_final = ct
ct = ct_final
print(f" ct time: {time.time() - t1}")


t1 = time.time()
Loss = 1000
ca_final = 0.001
for ca in [0.001, 0.0008, 0.0005, 0.0003]:
    New_Loss = optimization_step()
    if New_Loss < Loss:
        Loss = New_Loss
        ca_final = ca
ca = ca_final
print(f" ca time: {time.time() - t1}")


t1 = time.time()
Loss = 1000
cw_final = 0.003
for cw in [0.004, 0.003, 0.002, 0.001]:
    New_Loss = optimization_step()
    if New_Loss < Loss:
        Loss = New_Loss
        cw_final = cw
cw = cw_final
print(f" cw time: {time.time() - t1}")


t1 = time.time()
Loss = 1000
ndf_final = 160
for ndf in [120, 160, 200]:
    New_Loss = optimization_step()
    if New_Loss < Loss:
        Loss = New_Loss
        ndf_final = ndf
ndf = ndf_final
print(f" ndf time: {time.time() - t1}")


t1 = time.time()
Loss = 1000
ngf_final = 64
for ngf in [32, 48, 64]:
    New_Loss = optimization_step()
    if New_Loss < Loss:
        Loss = New_Loss
        ngf_final = ngf
ngf = ngf_final
print(f" ngf time: {time.time() - t1}")


t1 = time.time()
Loss = 1000
n_noise_final = 10
for n_noise in [1, 2, 5, 10, 20, 40]:
    New_Loss = optimization_step()
    if New_Loss < Loss:
        Loss = New_Loss
        n_noise_final = n_noise
n_noise = n_noise_final
print(f" n_noise time: {time.time() - t1}")


print("Final L1: ", L1)
print("Final ct: ", ct)
print("Final ca: ", ca)
print("Final cw: ", cw)
print("Final ndf: ", ndf)
print("Final ngf: ", ngf)
print("Final n_noise: ", n_noise)


