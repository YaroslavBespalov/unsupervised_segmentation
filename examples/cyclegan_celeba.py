import json
import sys, os

import albumentations

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from models.lambdaf import MySequential, LambdaF
from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader
from dataset.toheatmap import heatmap_to_measure, ToHeatMap, sparse_heatmap, ToGaussHeatMap, HeatMapToGaussHeatMap
from modules.hg import HG_softmax2020
from parameters.path import Paths

from albumentations.pytorch.transforms import ToTensor as AlbToTensor
from loss.tuner import CoefTuner, GoldTuner
from gan.loss.base import StyleGANLoss
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
from gan.gan_model import CondStyleDisc2Wrapper, cont_style_munit_enc, CondStyleGanModel, CondGen2, CondGen3, CondDisc3, \
    CondGenDecode, StyleGanModel
from gan.loss.loss_base import Loss
from metrics.writers import ItersCounter, send_images_to_tensorboard
from models.common import View
from models.munit.enc_dec import MunitEncoder, StyleEncoder
from models.uptosize import MakeNoise
from stylegan2.model import Generator, Discriminator, EqualLinear, EqualConv2d, Blur
from modules.linear_ot import SOT, PairwiseDistance


def handmadew1(m1,m2):
    lambd = 0.002
    with torch.no_grad():
        P = SOT(200, lambd).forward(m1, m2)
        M = PairwiseDistance()(m1.coord, m2.coord).sqrt()
        main_diag = (torch.diagonal(M, offset=0, dim1=1, dim2=2) * torch.diagonal(P, offset=0, dim1=1, dim2=2))
    return ((M * P).sum(dim=(1,2)) + main_diag.sum(dim=1)) / 2 # (2 * m1.coord.shape[1])

def liuboff(encoder: nn.Module):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        landmarks[landmarks > 1] = 0.99999
        # content = heatmap_to_measure(encoder(data))[0]
        pred_measure = UniformMeasure2DFactory.from_heatmap(encoder(data))
        target = UniformMeasure2D01(torch.clamp(landmarks, max=1))
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += (handmadew1(pred_measure, target) / eye_dist).sum().item()
    return sum_loss / len(LazyLoader.w300().test_dataset)


def verka(encoder: nn.Module):
    res = []
    for i, (image, lm) in enumerate(LazyLoader.celeba_test(64)):
        content = encoder(image.cuda())
        mes = UniformMeasure2D01(lm.cuda())
        pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(content)
        res.append(Samples_Loss(p=1)(mes, pred_measures).item() * image.shape[0])
    return np.mean(res)/len(LazyLoader.celeba_test(1).dataset)


def nadbka(encoder: nn.Module):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        content = heatmap_to_measure(encoder(data))[0]
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((content - landmarks).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    return sum_loss / len(LazyLoader.w300().test_dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def imgs_with_mask(imgs, mask, color=[1.0,1.0,1.0]):
    # mask = torch.cat([mask, mask, mask], dim=1)
    mask = mask[:, 0, :, :]
    res: Tensor = imgs.cpu().detach()
    res = res.permute(0, 2, 3, 1)
    res[mask > 0.00001, :] = torch.tensor(color, dtype=torch.float32)
    res = res.permute(0, 3, 1, 2)
    return res

def stariy_hm_loss(pred, target, coef=1.0):

    pred_mes = UniformMeasure2DFactory.from_heatmap(pred)
    target_mes = UniformMeasure2DFactory.from_heatmap(target)

    return Loss(
        nn.BCELoss()(pred, target) * coef +
        nn.MSELoss()(pred_mes.coord, target_mes.coord) * (0.001 * coef) +
        nn.L1Loss()(pred_mes.coord, target_mes.coord) * (0.001 * coef)
    )


def hm_loss_bes_xy(pred, target):

    return Loss(
        nn.BCELoss()(pred, target)
    )


counter = ItersCounter()
writer = SummaryWriter(f"{Paths.default.board()}/stylegan{int(time.time())}")
print(f"{Paths.default.board()}/stylegan{int(time.time())}")
l1_loss = nn.L1Loss()


def L1(name: Optional[str], writer: SummaryWriter = writer) -> Callable[[Tensor, Tensor], Loss]:

    if name:
        counter.active[name] = True

    def compute(t1: Tensor, t2: Tensor):
        loss = l1_loss(t1, t2)
        if name:
            if counter.get_iter(name) % 10 == 0:
                writer.add_scalar(name, loss, counter.get_iter(name))
        return Loss(loss)

    return compute


def writable(name: str, f: Callable[[Any], Loss]):
    counter.active[name] = True

    def decorated(*args, **kwargs) -> Loss:
        loss = f(*args, **kwargs)
        iter = counter.get_iter(name)
        if iter % 10 == 0:
            writer.add_scalar(name, loss.item(), iter)
        return loss

    return decorated


def entropy(hm: Tensor):
    B, N, D, D = hm.shape
    return Loss(-(hm * hm.log()).sum() / (B * D * D))


def gan_tuda_trainer(model, generator):

    def gan_train(real_img, heatmap):
        batch_size = real_img.shape[0]
        latent_size = 512
        requires_grad(generator, True)

        noise = mixing_noise(batch_size, latent_size, 0.9, device)
        fake, _ = generator(heatmap, noise, return_latents=False)
        model.discriminator_train([real_img], [fake], heatmap)

        writable("Generator loss tuda", model.generator_loss)([real_img], [fake], [], heatmap) \
            .minimize_step(model.optimizer.opt_min)

    return gan_train


def gan_obratno_trainer(model, generator):

    def gan_train(real_img, heatmap):
        batch_size = real_img.shape[0]
        requires_grad(generator, True)
        fake_heatmaps = generator.forward(real_img)
        model.discriminator_train([heatmap], [fake_heatmaps], real_img)

        writable("Generator loss obratno", model.generator_loss)([heatmap], [fake_heatmaps], [], real_img) \
            .minimize_step(model.optimizer.opt_min)
    return gan_train

# def gan_obratno_trainer(model, generator):
#
#     def gan_train(real_img, heatmap):
#         requires_grad(generator, True)
#
#         fake_heatmaps = generator.forward(real_img)
#         model.disc_train([heatmap], [fake_heatmaps])
#
#         writable("Generator loss obratno", model.generator_loss)([heatmap], [fake_heatmaps]) \
#             .minimize_step(model.optimizer.opt_min)
#     return gan_train


def gan_tuda_obratno_trainer(generator1, generator2, decoder, style_enc, style_opt):
    generator1_opt = optim.Adam(generator1.parameters(), lr=0.001)
    generator2_opt = optim.Adam(generator2.parameters(), lr=2e-5)

    def gan_trainer(real_img, heatmap):

        coefs = json.load(open("../parameters/cycle_loss.json"))

        batch_size = real_img.shape[0]
        latent_size = 512
        requires_grad(generator1, True)
        requires_grad(generator2, True)
        noise = mixing_noise(batch_size, latent_size, 0.9, device)

        fake_img = decoder(generator2(real_img), style_enc(real_img))
        fake_heatmap = generator2(generator1(heatmap, noise, return_latents=False)[0])

        (
                L1("L1 real_fake_image")(fake_img, real_img) * coefs["img"] +
                writable("heatmaps_loss", stariy_hm_loss)(fake_heatmap, heatmap, coefs["hm"])
        ).minimize_step(generator1_opt, generator2_opt, style_opt)
    return gan_trainer


def train(generator, decoder, discriminator, discriminatorHG, encoder_HG, style_encoder, device, starting_model_number):
    latent_size = 512
    batch_size = 12
    sample_z = torch.randn(8, latent_size, device=device)
    Celeba.batch_size = batch_size
    W300DatasetLoader.batch_size = batch_size
    W300DatasetLoader.test_batch_size = 64

    encoder_HG_supervised = HG_softmax2020(num_classes=68, heatmap_size=64)
    encoder_HG_supervised.load_state_dict(torch.load(f'{Paths.default.models()}/hg2_e29.pt', map_location="cpu"))
    encoder_HG_supervised = encoder_HG_supervised.cuda()
    requires_grad(encoder_HG_supervised, False)

    style_opt = optim.Adam(style_encoder.parameters(), lr=5e-4, betas=(0.9, 0.99))

    test_img = next(LazyLoader.celeba().loader)[:8].cuda()

    loss_st: StyleGANLoss = StyleGANLoss(discriminator)
    model = CondStyleGanModel(generator, loss_st, (0.001, 0.0015))

    loss_st2: StyleGANLoss = StyleGANLoss(discriminatorHG)
    model2 = CondStyleGanModel(encoder_HG, loss_st2, (2e-5, 0.0005))

    heatmaper = ToGaussHeatMap(64, 1.0)
    # wide_heatmaper = HeatMapToGaussHeatMap(64, 5.0)

    tuda_trainer = gan_tuda_trainer(model, generator)
    obratno_trainer = gan_obratno_trainer(model2, encoder_HG)
    tuda_obratno_trainer = gan_tuda_obratno_trainer(generator, encoder_HG, decoder, style_encoder, style_opt)

    for i in range(100000):
        counter.update(i)

        # requires_grad(encoder_HG, False)  # REMOVE BEFORE TRAINING
        real_img = next(LazyLoader.celeba().loader).to(device)
        heatmap = encoder_HG_supervised(real_img).detach()
        # heatmap = wide_heatmaper.forward(encoder_HG_supervised(real_img)).detach()
        # heatmap = encoder_HG_supervised(next(LazyLoader.celeba().loader).to(device)).detach()

        # tuda_trainer(real_img, heatmap)
        obratno_trainer(real_img, heatmap)
        # tuda_obratno_trainer(real_img, heatmap)

        # pred_hm = encoder_HG(real_img)
        # stariy_hm_loss(pred_hm, heatmap).minimize_step(model2.optimizer.opt_min)

        if i % 100 == 0:
            with torch.no_grad():

                content_test = encoder_HG(test_img)
                pred_measures_test: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(content_test)
                heatmaper_256 = ToGaussHeatMap(256, 2.0)
                sparse_hm_test = heatmaper.forward(pred_measures_test.coord * 63)
                sparse_hm_test_1 = heatmaper_256.forward(pred_measures_test.coord * 255)

                latent_test = style_encoder(test_img)

                sparce_mask = sparse_hm_test_1.sum(dim=1, keepdim=True)
                sparce_mask[sparce_mask < 0.0003] = 0
                iwm = imgs_with_mask(test_img, sparce_mask)
                send_images_to_tensorboard(writer, iwm, "REAL", i)

                fake_img, _ = generator(sparse_hm_test, [sample_z])
                iwm = imgs_with_mask(fake_img, pred_measures_test.toImage(256))
                send_images_to_tensorboard(writer, iwm, "FAKE", i)

                restored = decoder(sparse_hm_test, latent_test)
                iwm = imgs_with_mask(restored, pred_measures_test.toImage(256))
                send_images_to_tensorboard(writer, iwm, "RESTORED", i)

                content_test_256 = nn.Upsample(scale_factor=4)(sparse_hm_test).sum(dim=1, keepdim=True).repeat(1, 3, 1, 1) * \
                    torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1)

                content_test_256 = (content_test_256 - content_test_256.min()) / content_test_256.max()
                send_images_to_tensorboard(writer, content_test_256, "HM", i, normalize=False, range=(0, 1))

        if i % 50 == 0 and i >= 0:
            test_loss = liuboff(encoder_HG)
            print("liuboff", test_loss)
            # test_loss = nadbka(encoder_HG)
            # tuner.update(test_loss)
            writer.add_scalar("liuboff", test_loss, i)

        if i % 10000 == 0 and i > 0:
            torch.save(
                {
                    'g': generator.module.state_dict(),
                    'd': discriminator.module.state_dict(),
                    'c': encoder_HG.module.state_dict(),
                    "s": style_encoder.state_dict(),
                    "d2": discriminatorHG.state_dict()
                },
                f'{Paths.default.models()}/cyclegan_{str(i + starting_model_number).zfill(6)}.pt',
            )


class SimpleDisc(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            EqualLinear(136, 256, activation='fused_lrelu'),
            EqualLinear(256, 256, activation='fused_lrelu'),
            EqualLinear(256, 256, activation='fused_lrelu'),
            EqualLinear(256, 256, activation='fused_lrelu'),
            EqualLinear(256, 1),
        )

    def forward(self, hm: Tensor):
        coord, p = heatmap_to_measure(hm)
        return self.net(coord.view(-1, 136))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)
    encoder_HG = HG_softmax2020(num_classes=68, heatmap_size=64)


    latent = 512
    n_mlp = 5
    size = 256

    generator = CondGen3(Generator(
        size, latent, n_mlp, channel_multiplier=1
    ), heatmap_channels=68)

    discriminator = CondDisc3(
        size, channel_multiplier=1, heatmap_channels=68
    )

    wide_heatmaper = HeatMapToGaussHeatMap(64, 2.0)

    discriminatorHG = MySequential(
        LambdaF([], lambda x, y: (y, wide_heatmaper.forward(x).sum(1, keepdim=True))),
        # discriminator
        CondDisc3(size, channel_multiplier=1, heatmap_channels=1)
    )
    # discriminatorHG = SimpleDisc()

    style_encoder = StyleEncoder(style_dim=latent)


    starting_model_number = 170000
    weights = torch.load(
        f'{Paths.default.models()}/stylegan2_new_{str(starting_model_number).zfill(6)}.pt',
        # f'{Paths.default.nn()}/stylegan2_w300_{str(starting_model_number).zfill(6)}.pt',
        map_location="cpu"
    )

    discriminator.load_state_dict(weights['d'])
    generator.load_state_dict(weights['g'])
    style_encoder.load_state_dict(weights['s'])
    # encoder_HG.load_state_dict(weights['c'])

    generator = generator.cuda()
    discriminator = discriminator.to(device)
    encoder_HG = encoder_HG.cuda()
    discriminatorHG = discriminatorHG.cuda()
    style_encoder = style_encoder.cuda()
    decoder = CondGenDecode(generator)

    generator = nn.DataParallel(generator, [0, 2, 3])
    discriminator = nn.DataParallel(discriminator, [0, 2, 3])
    encoder_HG = nn.DataParallel(encoder_HG, [0, 2, 3])
    decoder = nn.DataParallel(decoder, [0, 2, 3])
    discriminatorHG = nn.DataParallel(discriminatorHG, [0, 2, 3])

    train(generator, decoder, discriminator, discriminatorHG, encoder_HG, style_encoder, device, starting_model_number)
