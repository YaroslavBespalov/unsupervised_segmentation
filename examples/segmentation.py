import argparse
import time
from typing import List
import numpy as np
import albumentations
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
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

if not os.path.exists(os.path.join(f"./segm_mask_19_12/")):
            os.makedirs(os.path.join(f"./segm_mask_19_12/"))

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

dataset_test = SegmentationDataset(
    "/home/ibespalov/GANS_NAZAR/dataset/postprocessing/init",
    "/home/ibespalov/GANS_NAZAR/dataset/postprocessing/masks",
    transform_joint=albumentations.Compose([
                               albumentations.Resize(args.image_size, args.image_size),
                               albumentations.CenterCrop(args.image_size, args.image_size),
                               ToTensorV2()
    ]),
    img_transform=torchvision.transforms.Normalize(mean=[57.0, 57.0, 57.0], std=[57.0, 57.0, 57.0])
)

dataset = MRIImages("/raid/data/unsup_segment/samples_without_masks",
                    transform=albumentations.Compose([
                        albumentations.Resize(args.image_size, args.image_size),
                        albumentations.OpticalDistortion(border_mode=0),
                        albumentations.ElasticTransform(border_mode=0),
                        albumentations.CenterCrop(args.image_size, args.image_size),
                        ToTensorV2()
                    ]),
                    )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)

noise = NormalNoise(args.noise_size, device)
measure2image = ResMeasureToImage(args.measure_size * 3 + noise.size(), args.image_size, args.ngf).cuda()

netD = DCDiscriminator(ndf=args.ndf).cuda()
gan_model = GANModel(measure2image,
                     HingeLoss(netD).add_generator_loss(nn.L1Loss(), args.L1),
                     lr=0.0005)

fabric = ProbabilityMeasureFabric(args.image_size)
barycenter = fabric.load("barycenter").cuda().padding(args.measure_size)
print(barycenter.coord.shape)
barycenter = fabric.cat([barycenter for b in range(40)])
print(barycenter.coord.shape)

image2measure = ResImageToMeasure(args.measure_size).cuda()
image2measure_opt = optim.Adam(image2measure.parameters(), lr=0.0002)

def test():
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=40, num_workers=20)

    sum_loss = 0
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(dataloader_test, 0):
            imgs = imgs.cuda().type(torch.float32)
            pred_measures: ProbabilityMeasure = image2measure(imgs)
            ref_measures: ProbabilityMeasure = fabric.from_mask(masks).cuda().padding(args.measure_size)
            # barycenter_batch = barycenter.slice(0, ref_measures.coord.shape[0])
            ref_loss = Samples_Loss()(pred_measures, ref_measures)
            sum_loss += ref_loss.item()

    print(sum_loss)


test()
# cycle_gan = CycleGAN[ProbabilityMeasure, Tensor](
#     Measure2imgTmp(measure2image, noise),
#     NamedModule(image2measure, ['image'], ['measure']),
#     {'measure': Samples_Loss().forward}, {'image': lambda x, y: Loss(nn.L1Loss().forward(x, y))}
# )


for epoch in range(300):

    ot_iters = 100

    if epoch % 12 == 0 and epoch > 0:
        gan_model.optimizer.update_lr(0.9)
        for i in image2measure_opt.param_groups:
            i['lr'] *= 0.9
    print("epoch", epoch)
    test_imgs = None

    t1 = time.time()

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
        bc_loss = bc_loss_W * args.cw + bc_loss_A * args.ca + bc_loss_T * args.ct

        fake = measure2image(cond)
        g_loss = gan_model.generator_loss(imgs, fake)
        cg = min(epoch/5.0, 2.0)
        (g_loss * cg + bc_loss).minimize_step(image2measure_opt)

        # cycle_gan.train({'measure': pred_measures.detach()}, {'image': imgs})

    print(f"epoch time: {time.time() - t1}")

    with torch.no_grad():

        pred_measures: ProbabilityMeasure = image2measure(test_imgs)
        cond = pred_measures.toChannels()
        z = noise.sample(cond.shape[0])
        cond = torch.cat((cond, z), dim=1)

        fake = measure2image(cond)
        f_images = np.transpose(fake[0].cpu().numpy(), (1, 2, 0))
        cv2.imwrite(f'./segm_mask_19_12/img_{str(epoch + 1).zfill(6)}.tiff', f_images)
        # plt.imshow(np.transpose(fake[0].cpu().numpy(), (1, 2, 0)))
        # plt.show()

        pred: ProbabilityMeasure = image2measure(test_imgs)
        f_mask = pred.toImage(100)[0][0].cpu().numpy()
        cv2.imwrite(f'./segm_mask_19_12/mask_{str(epoch + 1).zfill(6)}.tiff', f_mask)
        # plt.imshow(f_mask)
        # plt.show()

    test()