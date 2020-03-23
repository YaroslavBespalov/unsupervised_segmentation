import sys
import os

from torch.utils.tensorboard import SummaryWriter

from munit.networks import StyleEncoder, ContentEncoder

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
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
from metrics.writers import send_to_tensorboard, ItersCounter
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn
from torch import optim
from torchvision import utils
from dataset.cardio_dataset import SegmentationDataset, MRIImages, ImageMeasureDataset
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from gans_pytorch.gan.gan_model import GANModel, ganmodel_munit
from gans_pytorch.gan.loss.wasserstein import WassersteinLoss
from gans_pytorch.gan.loss.hinge import HingeLoss
from gans_pytorch.gan.noise.normal import NormalNoise
from gans_pytorch.gan.gan_model import stylegan2
from gans_pytorch.optim.min_max import MinMaxParameters, MinMaxOptimizer
from gans_pytorch.stylegan2.model import ConvLayer, EqualLinear, PixelNorm
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
from parameters.gan import GanParameters, MunitParameters
from transforms_utils.transforms import MeasureToMask, ToNumpy, ToTensor, MaskToMeasure, NumpyBatch
from useful_utils.save import save_image_with_mask

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    parents=[
        DatasetParameters(),
        GanParameters(),
        DeformationParameters(),
        MunitParameters()
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

gan_model: GANModel = ganmodel_munit("hinge", 0.0001)

enc_style = StyleEncoder(n_downsample=4, input_dim=args.input_dim, dim=args.dim, style_dim=args.style_dim,
                         norm=args.norm, activ=args.activ, pad_type=args.pad_type).cuda()
enc_content: ContentEncoder = ContentEncoder(args.n_downsample, args.n_res, args.input_dim, args.dim, 'in', args.activ,
                                             args.pad_type).cuda()

counter = ItersCounter()
writer = SummaryWriter("/home/ibespalov/pomoika/munit")
gan_model.train = send_to_tensorboard("generator loss", "discriminator loss", counter=counter, writer=writer)(gan_model.train)
gan_model.generator.forward = send_to_tensorboard("Fake", counter=counter, skip=10, writer=writer)(
    gan_model.generator.forward
)

for epoch in range(500):

    print("epoch", epoch)

    for i, (imgs, masks) in enumerate(dataloader, 0):
        counter.update(i)
        if imgs.shape[0] != args.batch_size:
            continue

        imgs = imgs.cuda()
        content = enc_content(imgs)
        style = enc_style(imgs)
        restored = gan_model.generator(content, style)
        Loss(nn.L1Loss()(restored, imgs)).minimize_step(gan_model.optimizer.opt_min)
        content = enc_content(imgs)
        style_noise = noise.sample(8)[:, :, None, None]
        gan_model.train([imgs], content, style_noise)







