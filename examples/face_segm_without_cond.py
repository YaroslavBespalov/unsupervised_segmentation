import sys
import os

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

import argparse
import time
from itertools import chain
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
from dataset.cardio_dataset import SegmentationDataset, MRIImages, ImageMeasureDataset
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from gans_pytorch.gan.gan_model import GANModel, ConditionalGANModel, stylegan2_cond_transfer, stylegan2_transfer
from gans_pytorch.gan.loss.wasserstein import WassersteinLoss
from gans_pytorch.gan.loss.hinge import HingeLoss
from gans_pytorch.gan.noise.normal import NormalNoise
from gans_pytorch.gan.gan_model import stylegan2
from gans_pytorch.optim.min_max import MinMaxParameters, MinMaxOptimizer
from gans_pytorch.stylegan2_pytorch.model import ConvLayer, EqualLinear, PixelNorm
from gan.loss.gan_loss import GANLossObject
from loss.losses import Samples_Loss
from loss.regulariser import BarycenterRegularizer, DualTransformRegularizer
from loss_base import Loss
from modules.image2measure import ResImageToMeasure
from modules.lambdaf import LambdaF
from modules.cat import Concat
from torchvision import transforms, utils

from modules.uptosize import Uptosize
from parameters.dataset import DatasetParameters
from parameters.deformation import DeformationParameters
from parameters.gan import GanParameters
from transforms_utils.transforms import MeasureToMask, ToNumpy, ToTensor, MaskToMeasure, NumpyBatch
from useful_utils.save import save_image_with_mask

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

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

image_size = 256

full_dataset = ImageMeasureDataset(
    "/raid/data/celeba",
    "/raid/data/celeba_masks",
    img_transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
)


g_transforms: albumentations.DualTransform = albumentations.Compose([
    MeasureToMask(size=256),
    ToNumpy(),
    NumpyBatch(albumentations.ShiftScaleRotate(p=1, rotate_limit=20)),
    ToTensor(device),
    MaskToMeasure(size=256, padding=args.measure_size),
])


train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [len(full_dataset) - 1000, 1000])

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)

noise = NormalNoise(args.noise_size, device)
#
# cond_gan_model = stylegan2_cond_transfer(
#     "/home/ibespalov/stylegan2/stylegan2-pytorch/checkpoint/790000.pt",
#     "hinge",
#     0.002,
#     args.measure_size * 3,
#     args.measure_size * 3 + noise.size(),
#     image_size
# )


# cond_gan_model: ConditionalGANModel = stylegan2_cond_transfer(
#     "/home/ibespalov/stylegan2/stylegan2-pytorch/checkpoint/790000.pt",
#     "hinge",
#     0.001,
#     args.measure_size * 3,
#     args.measure_size * 3 + noise.size(),
#     image_size
# )

gan_model: GANModel = stylegan2_transfer(
    "/home/ibespalov/stylegan2/stylegan2-pytorch/checkpoint/790000.pt",
    "hinge",
    0.001,
    args.measure_size * 3,
    args.measure_size * 3 + noise.size()
)



fabric = ProbabilityMeasureFabric(args.image_size)
barycenter = fabric.load("/home/ibespalov/unsupervised_pattern_segmentation/examples/face_barycenter").cuda().padding(args.measure_size).batch_repeat(args.batch_size)

# cond_gan_model.loss += GANLossObject(
#                 lambda dx, dy: Loss.ZERO(),
#                 lambda dgz, real, fake: Loss(
#                     nn.L1Loss()(image2measure(fake[0]).coord, fabric.from_channels(real[1]).coord.detach())
#                 ) * 10,
#                 None
# )

image2measure = ResImageToMeasure(args.measure_size).cuda()
image2measure_opt_strong = optim.Adam(image2measure.parameters(), lr=0.0001)
image2measure_opt = optim.Adam(image2measure.parameters(), lr=0.0003)


R_b = BarycenterRegularizer.__call__(barycenter)
R_t = DualTransformRegularizer.__call__(
    g_transforms, lambda trans_dict:
    Samples_Loss()(image2measure(trans_dict['image']), trans_dict['mask'])
)

deform_array = list(np.linspace(0, 6, 1000))
Whole_Reg = R_t @ deform_array + R_b

for epoch in range(500):
    # if epoch > 0:
    #     cond_gan_model.optimizer.update_lr(0.5)
        # for i in image2measure_opt.param_groups:
        #     i['lr'] *= 0.5
    print("epoch", epoch)

    for i, (imgs, masks) in enumerate(dataloader, 0):
        if imgs.shape[0] != args.batch_size:
            continue

        imgs = imgs.cuda()
        pred_measures: ProbabilityMeasure = image2measure(imgs)
        loss_R: Loss = Whole_Reg.apply(i)(imgs, pred_measures)
        loss_R.minimize_step(image2measure_opt)

        cond = pred_measures.toChannels()
        z = noise.sample(cond.shape[0])



        loss_lossovichi = gan_model.train([imgs], cond.detach(), z)
        fake = gan_model.forward(cond.detach(), z)
        fake_measure = image2measure(fake)
        Loss(nn.L1Loss()(fake_measure.coord, pred_measures.coord.detach()) * 100).minimize_step(
            gan_model.optimizer.opt_min,
            image2measure_opt_strong
        )

        pred_measures: ProbabilityMeasure = image2measure(imgs)
        cond = pred_measures.toChannels()
        gan_gen_loss = (gan_model.generator_loss([imgs], cond, z) * 0.01)
        gan_gen_loss.minimize_step(image2measure_opt)
        # if i > 2000:
        #     fake = cond_gan_model.forward(cond.detach(), z).detach()
        #     fake_measure: ProbabilityMeasure = image2measure(fake)
        #     # Loss(nn.L1Loss()(fake_measure.coord, pred_measures.detach().coord)).minimize_step(image2measure_opt)
        #     loss_lossovichi = cond_gan_model.train([imgs], cond.detach(), z)
        #     pred_measures: ProbabilityMeasure = image2measure(imgs)
        #     cond = pred_measures.toChannels()
        #     gan_gen_loss = (cond_gan_model.generator_loss([imgs], cond, z) * 0.01)
        #     gan_gen_loss.minimize_step(image2measure_opt)


        if i % 100 == 0:
            print(i)
            # if i > 2000:
            print("Whole Loss: ", loss_R.item())
            print("gan_gen_loss: ", gan_gen_loss.item())
            with torch.no_grad():
                pred: ProbabilityMeasure = image2measure(imgs)
                # plt.scatter(barycenter.coord[0,:,0].cpu().numpy(), barycenter.coord[0,:,1].cpu().numpy())
                plt.scatter(pred.coord[0,:,0].cpu().numpy(), pred.coord[0,:,1].cpu().numpy())
                plt.show()
                save_image_with_mask(imgs, pred.toImage(256), f'../sample/{str(i).zfill(6)}.png')
                # if i > 2000:
                save_image_with_mask(fake, pred.toImage(256), f'../sample_gan/{str(i).zfill(6)}.png')





