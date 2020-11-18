import json
import sys, os

from gan.models.stylegan import CondStyleGanModel
from gan.nn.stylegan.discriminator import CondDisc7
from gan.nn.stylegan.generator import CondGen7, CondGenDecode

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/models'))

from train_procedure import gan_trainer, content_trainer_with_gan, content_trainer_supervised, requires_grad, \
    train_content
from gan.loss.stylegan import StyleGANLoss
from modules.accumulator import Accumulator
import albumentations
# from matplotlib import pyplot as plt
from gan.noise.stylegan import mixing_noise
from loss.hmloss import noviy_hm_loss, coord_hm_loss
from metrics.measure import liuboff, liuboffMAFL
from viz.image_with_mask import imgs_with_mask

from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader
from dataset.toheatmap import heatmap_to_measure, ToHeatMap, sparse_heatmap, ToGaussHeatMap, HeatMapToGaussHeatMap, \
    HeatMapToParabola, CoordToGaussSkeleton
# from modules.hg import HG_softmax2020
from modules.nashhg import HG_softmax2020, HG_skeleton
from parameters.path import Paths

from albumentations.pytorch.transforms import ToTensor as AlbToTensor
from loss.tuner import CoefTuner, GoldTuner
from gan.loss.penalties.penalty import DiscriminatorPenalty
from loss.losses import Samples_Loss
from loss.regulariser import DualTransformRegularizer, BarycenterRegularizer, StyleTransformRegularizer, \
    UnoTransformRegularizer
from transforms_utils.transforms import MeasureToMask, ToNumpy, NumpyBatch, ToTensor, MaskToMeasure, ResizeMask, \
    NormalizeMask, ParTr
from stylegan2.op import upfirdn2d
import argparse
import math
import random
import os
import time
from typing import List, Optional, Callable, Any, Tuple

import numpy as np
import torch
from torch import nn, autograd, optim, Tensor
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from dataset.cardio_dataset import ImageMeasureDataset, ImageDataset
from dataset.probmeasure import ProbabilityMeasure, ProbabilityMeasureFabric, UniformMeasure2DFactory, \
    UniformMeasure2D01
from metrics.writers import ItersCounter, send_images_to_tensorboard, WR
from models.common import View
from models.munit.enc_dec import MunitEncoder, StyleEncoder
from models.uptosize import MakeNoise
from stylegan2.model import Generator, Discriminator, EqualLinear, EqualConv2d, Blur
from modules.linear_ot import SOT, PairwiseDistance


def train(generator, decoder, discriminator, encoder_HG, style_encoder, device, starting_model_number):
    latent_size = 512
    batch_size = 8
    sample_z = torch.randn(8, latent_size, device=device)
    Celeba.batch_size = batch_size
    W300DatasetLoader.batch_size = batch_size
    W300DatasetLoader.test_batch_size = 32

    test_img = next(LazyLoader.mafl().loader_train_inf)["data"][:8].cuda()

    model = CondStyleGanModel(generator, StyleGANLoss(discriminator), (0.001/4, 0.0015/4))

    style_opt = optim.Adam(style_encoder.parameters(), lr=5e-4, betas=(0.9, 0.99))
    cont_opt = optim.Adam(encoder_HG.parameters(), lr=3e-5, betas=(0.5, 0.97))

    g_transforms: albumentations.DualTransform = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.Compose([
            albumentations.ElasticTransform(p=0.7, alpha=150, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.9, rotate_limit=15),
        ])),
        ToTensor(device),
    ])

    g_transforms_without_norm: albumentations.DualTransform = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.Compose([
            albumentations.ElasticTransform(p=0.3, alpha=150, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.7, rotate_limit=15),
        ])),
        ToTensor(device),
    ])

    R_t = DualTransformRegularizer.__call__(
        g_transforms, lambda trans_dict, img:
        coord_hm_loss(encoder_HG(trans_dict['image'])["coords"], trans_dict['mask'])
    )

    R_s = UnoTransformRegularizer.__call__(
        g_transforms, lambda trans_dict, img, ltnt:
        WR.L1("R_s")(ltnt, style_encoder(trans_dict['image']))
    )

    barycenter: UniformMeasure2D01 = UniformMeasure2DFactory.load(f"{Paths.default.models()}/face_barycenter_5").cuda().batch_repeat(batch_size)

    R_b = BarycenterRegularizer.__call__(barycenter, 1.0, 2.0, 4.0)

    tuner = GoldTuner([0.37, 2.78, 0.58, 1.43, 3.23], device=device, rule_eps=0.001, radius=0.3, active=False)

    trainer_gan = gan_trainer(model, generator, decoder, encoder_HG, style_encoder, R_s, style_opt, g_transforms)
    content_trainer = content_trainer_with_gan(cont_opt, tuner, encoder_HG, R_b, R_t, model, generator, g_transforms, decoder, style_encoder)
    # supervise_trainer = content_trainer_supervised(cont_opt, encoder_HG, LazyLoader.w300().loader_train_inf)

    for i in range(100000):
        WR.counter.update(i)

        requires_grad(encoder_HG, False)
        real_img = next(LazyLoader.mafl().loader_train_inf)["data"].to(device)

        encoded = encoder_HG(real_img)
        internal_content = encoded["skeleton"].detach()

        trainer_gan(i, real_img, internal_content)
        # content_trainer(real_img)
        train_content(cont_opt, R_b, R_t, real_img, model, encoder_HG, decoder, generator, style_encoder)
        # supervise_trainer()

        encoder_ema.accumulate(encoder_HG.module, i, 0.97)
        if i % 50 == 0 and i > 0:
            encoder_ema.write_to(encoder_HG.module)

        if i % 100 == 0:
            coefs = json.load(open("../parameters/content_loss.json"))
            print(i, coefs)
            with torch.no_grad():

                # pred_measures_test, sparse_hm_test = encoder_HG(test_img)
                encoded_test = encoder_HG(test_img)
                pred_measures_test: UniformMeasure2D01 = UniformMeasure2D01(encoded_test["coords"])
                heatmaper_256 = ToGaussHeatMap(256, 1.0)
                sparse_hm_test_1 = heatmaper_256.forward(pred_measures_test.coord)

                latent_test = style_encoder(test_img)

                sparce_mask = sparse_hm_test_1.sum(dim=1, keepdim=True)
                sparce_mask[sparce_mask < 0.0003] = 0
                iwm = imgs_with_mask(test_img, sparce_mask)
                send_images_to_tensorboard(WR.writer, iwm, "REAL", i)

                fake_img, _ = generator(encoded_test["skeleton"], [sample_z])
                iwm = imgs_with_mask(fake_img, pred_measures_test.toImage(256))
                send_images_to_tensorboard(WR.writer, iwm, "FAKE", i)

                restored = decoder(encoded_test["skeleton"], latent_test)
                iwm = imgs_with_mask(restored, pred_measures_test.toImage(256))
                send_images_to_tensorboard(WR.writer, iwm, "RESTORED", i)

                content_test_256 = (encoded_test["skeleton"]).repeat(1, 3, 1, 1) * \
                    torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1)

                content_test_256 = (content_test_256 - content_test_256.min()) / content_test_256.max()
                send_images_to_tensorboard(WR.writer, content_test_256, "HM", i, normalize=False, range=(0, 1))

        if i % 50 == 0 and i >= 0:
            test_loss = liuboffMAFL(encoder_HG)
            print("liuboff", test_loss)
            # test_loss = nadbka(encoder_HG)
            tuner.update(test_loss)
            WR.writer.add_scalar("liuboff", test_loss, i)

        if i % 10000 == 0 and i > 0:
            torch.save(
                {
                    'g': generator.module.state_dict(),
                    'd': discriminator.module.state_dict(),
                    'c': encoder_HG.module.state_dict(),
                    "s": style_encoder.state_dict(),
                    "e": encoder_ema.storage_model.state_dict()
                },
                f'{Paths.default.models()}/stylegan2_mafl_{str(i + starting_model_number).zfill(6)}.pt',
            )


if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)
    # encoder_HG = HG_softmax2020(num_classes=5, heatmap_size=64)
    encoder_HG = HG_skeleton(CoordToGaussSkeleton(256, 4), num_classes=5, heatmap_size=64)
    encoder_ema = Accumulator(HG_skeleton(CoordToGaussSkeleton(256, 4), num_classes=5, heatmap_size=64))

    print("HG")

    latent = 512
    n_mlp = 5
    size = 256

    generator = CondGen7(Generator(
        size, latent, n_mlp, channel_multiplier=1,
    ), heatmap_channels=1)

    discriminator = CondDisc7(
        size, heatmap_channels=1, channel_multiplier=1
    )

    style_encoder = StyleEncoder(style_dim=latent)

    starting_model_number = 250000
    weights = torch.load(
        f'{Paths.default.models()}/stylegan2_mafl_{str(starting_model_number).zfill(6)}.pt',
        map_location="cpu"
    )

    discriminator.load_state_dict(weights['d'])
    generator.load_state_dict(weights['g'])
    style_encoder.load_state_dict(weights['s'])
    encoder_HG.load_state_dict(weights['e'])

    encoder_ema.storage_model.load_state_dict(encoder_HG.state_dict())

    generator = generator.cuda()
    discriminator = discriminator.to(device)
    encoder_HG = encoder_HG.cuda()
    style_encoder = style_encoder.cuda()
    decoder = CondGenDecode(generator)

    GPUS = [3, 1]

    generator = nn.DataParallel(generator, GPUS)
    discriminator = nn.DataParallel(discriminator, GPUS)
    encoder_HG = nn.DataParallel(encoder_HG, GPUS)
    decoder = nn.DataParallel(decoder, GPUS)

    train(generator, decoder, discriminator, encoder_HG, style_encoder, device, starting_model_number)
