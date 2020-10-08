import json
import sys, os

import albumentations


sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader, MAFL
from dataset.toheatmap import heatmap_to_measure, ToHeatMap, sparse_heatmap, ToGaussHeatMap
from modules.hg import HG_softmax2020
from parameters.path import Paths

from albumentations.pytorch.transforms import ToTensor as AlbToTensor
from loss.tuner import CoefTuner, GoldTuner
from gan.loss.gan_loss import StyleGANLoss
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
    CondGenDecode
from gan.loss_base import Loss
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
    for i, batch in enumerate(LazyLoader.mafl().test_loader):
        data = batch['data'].to(device)
        landmarks = batch["meta"]["keypts_normalized"].cuda().type(dtype=torch.float32)
        landmarks[landmarks > 1] = 0.99999
        # content = heatmap_to_measure(encoder(data))[0]
        pred_measure = UniformMeasure2DFactory.from_heatmap(encoder(data))
        target = UniformMeasure2D01(torch.clamp(landmarks, max=1))
        eye_dist = landmarks[:, 1] - landmarks[:, 0]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        # w1_loss = (handmadew1(pred_measure, target) / eye_dist).sum().item()
        # l1_loss = ((pred_measure.coord - target.coord).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
        # print(w1_loss, l1_loss)
        sum_loss += ((pred_measure.coord - target.coord).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    return sum_loss / len(LazyLoader.mafl().test_dataset)


# def verka(encoder: nn.Module):
#     res = []
#     for i, (image, lm) in enumerate(LazyLoader.celeba_test(64)):
#         content = encoder(image.cuda())
#         mes = UniformMeasure2D01(lm.cuda())
#         pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(content)
#         res.append(Samples_Loss(p=1)(mes, pred_measures).item() * image.shape[0])
#     return np.mean(res)/len(LazyLoader.celeba_test(1).dataset)
#
#
# def nadbka(encoder: nn.Module):
#     sum_loss = 0
#     for i, batch in enumerate(LazyLoader.w300().test_loader):
#         data = batch['data'].to(device)
#         landmarks = batch["meta"]["keypts_normalized"].cuda()
#         content = heatmap_to_measure(encoder(data))[0]
#         eye_dist = landmarks[:, 45] - landmarks[:, 36]
#         eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
#         sum_loss += ((content - landmarks).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
#     return sum_loss / len(LazyLoader.w300().test_dataset)


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
        # (pred - target).abs().mean() * 0.1 * coef
    )

def hm_svoego_roda_loss(pred, target):

    pred_xy, _ = heatmap_to_measure(pred)
    with torch.no_grad():
        t_xy, _ = heatmap_to_measure(target)

    return Loss(
        nn.BCELoss()(pred, target) +
        nn.MSELoss()(pred_xy, t_xy) * 0.001
        # (pred - target).abs().mean() * 0.1
    )

def hm_loss_bes_xy(pred, target):

    return Loss(
        nn.BCELoss()(pred, target)
        # (pred - target).abs().mean() * 0.1
    )

#
# def hm_svoego_roda_loss(pred: Tuple[UniformMeasure2D01, Tensor], target: Tensor, coef=1.0):
#
#     with torch.no_grad():
#         t_xy, _ = heatmap_to_measure(target)
#
#     return Loss(
#         nn.BCELoss()(pred[1], target) * coef +
#         nn.MSELoss()(pred[0].coord, t_xy) * (0.0005 * coef) +
#         nn.L1Loss()(pred[0].coord, t_xy) * (0.0005 * coef) +
#         (pred[1] - target).abs().mean() * (0.3 * coef)
#     )

# def rt_loss(pred: Tuple[UniformMeasure2D01, Tensor], target: Tensor):
#
#     with torch.no_grad():
#         t_xy, _ = heatmap_to_measure(target)
#
#     return Loss(
#         nn.L1Loss()(pred[0].coord, t_xy)
#     )


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


def gan_trainer(model, generator, decoder, encoder_HG, style_encoder, R_s, style_opt, heatmaper, g_transforms):

    def gan_train(i, real_img, pred_measures, sparse_hm, apply_g=True):
        batch_size = real_img.shape[0]
        latent_size = 512
        requires_grad(generator, True)

        coefs = json.load(open("../parameters/gan_loss.json"))

        if apply_g:
            trans_dict = g_transforms(image=real_img, mask=sparse_hm)
            trans_real_img = trans_dict["image"]
            trans_sparse_hm = trans_dict["mask"]
        else:
            trans_real_img = real_img
            trans_sparse_hm = sparse_hm

        noise = mixing_noise(batch_size, latent_size, 0.9, device)
        fake, _ = generator(trans_sparse_hm, noise, return_latents=False)

        # fake2, _ = generator(sparse_hm, noise, return_latents=True)
        model.disc_train([trans_real_img], [fake], trans_sparse_hm)
        # model.disc_train([real_img], [fake2], sparse_hm)

        writable("Generator loss", model.generator_loss)([trans_real_img], [fake], [], trans_sparse_hm) \
            .minimize_step(model.optimizer.opt_min)
        # writable("Generator loss 2", model.generator_loss)([real_img], [fake2], [], sparse_hm) \
        #     .minimize_step(model.optimizer.opt_min)

        if True:
            noise = mixing_noise(batch_size, latent_size, 0.9, device)

            fake, fake_latent = generator(trans_sparse_hm, noise, return_latents=True)


            fake_latent_test = fake_latent[:, [0, 13], :].detach()
            fake_latent_pred = style_encoder(fake)

            fake_content = encoder_HG(fake)
            # fake_content2 = encoder_HG(fake2)
            # fake_pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(fake_content)
            # fake_sparse_hm = heatmaper.forward(pred_measures.coord * 63).detach()
            style_enc = style_encoder(real_img)
            restored = decoder(trans_sparse_hm, style_enc)

            (
                    writable("BCE content gan", stariy_hm_loss)(fake_content, trans_sparse_hm, 5000) * coefs["BCE content gan"] +
                    L1("L1 restored")(restored, trans_real_img) * coefs["L1 restored"] +
                    L1("L1 style gan")(fake_latent_pred, fake_latent_test) * coefs["L1 style gan"] +
                    R_s(fake.detach(), fake_latent_pred) * coefs["R_s"]
            ).minimize_step(
                model.optimizer.opt_min,
                style_opt
            )

    return gan_train


def train_content(cont_opt, R_b, R_t, real_img, heatmaper, g_transforms):
    requires_grad(encoder_HG, True)

    coefs = json.load(open("../parameters/content_loss_sup.json"))
    # pred_measures, sparse_hm = encoder_HG(real_img)
    content = encoder_HG(real_img)
    pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(content)
    sparse_hm = heatmaper.forward(pred_measures.coord * 63)

    ll = (
        writable("R_b", R_b.__call__)(real_img, pred_measures) * coefs["R_b"] +
        writable("Sparse", hm_loss_bes_xy)(content, sparse_hm.detach()) * coefs["Sparse"] +
        writable("R_t", R_t.__call__)(real_img, sparse_hm) * coefs["R_t"]
    )
    ll.minimize_step(cont_opt)


def content_trainer_with_gan(cont_opt, tuner, heatmaper, encoder_HG, R_b, R_t, model, generator, g_transforms):
    latent_size = 512

    def do_train(real_img):

        batch_size = real_img.shape[0]
        requires_grad(encoder_HG, True)
        requires_grad(generator, False)
        requires_grad(model.loss.discriminator, False)
        # pred_measures, sparse_hm = encoder_HG(real_img)
        img_content = encoder_HG(real_img)
        pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(img_content)
        sparse_hm = heatmaper.forward(pred_measures.coord * 63).detach()
        # restored = decoder(sparse_hm, style_encoder(real_img))

        trans = g_transforms(image=real_img, mask=img_content)
        trans_content, trans_image = trans["mask"], trans["image"]

        restored = decoder(img_content, style_encoder(trans_image))

        noise1 = mixing_noise(batch_size, latent_size, 0.9, device)
        fake1, _ = generator(trans_content, noise1)
        trans_fake_content = encoder_HG(fake1.detach())

        coefs = json.load(open("../parameters/content_loss_sup.json"))

        tuner.sum_losses([
            writable("Fake-content D", model.loss.generator_loss)(
                real=None,
                fake=[fake1, img_content.detach()]) * coefs["Fake-content D"],  # 50 000
            writable("Real-content D", model.loss.discriminator_loss_as_is)(
                [real_img, img_content],
                [fake1.detach(), img_content]) * coefs["Real-content D"],  # 3000
            writable("R_b", R_b.__call__)(real_img, pred_measures) * coefs["R_b"],  # 3000
            writable("Sparse", hm_loss_bes_xy)(img_content, sparse_hm) * coefs["Sparse"],  # 1.5
            writable("R_t", R_t.__call__)(real_img, sparse_hm) * coefs["R_t"],  # 3
            L1("L1 image")(restored, real_img) * coefs["L1 image"],  # 10
            writable("fake_content loss", stariy_hm_loss)(
                trans_fake_content, trans_content
            ) * coefs["fake_content loss"]  # 6477
        ]).minimize_step(
            cont_opt
        )

    return do_train


def content_trainer_supervised(cont_opt, encoder_HG, loader):
    heatmaper = ToHeatMap(64)
    def do_train():
        requires_grad(encoder_HG, True)
        w300_batch = next(loader)
        w300_image = w300_batch['data'].to(device).type(torch.float32)
        w300_mes = ProbabilityMeasureFabric(256).from_coord_tensor(w300_batch["meta"]["keypts_normalized"]).cuda()
        w300_target_hm = heatmaper.forward(w300_mes.probability.type(torch.float32), w300_mes.coord.type(torch.float32) * 63).detach()
        content300 = encoder_HG(w300_image)

        coefs = json.load(open("../parameters/content_loss_sup.json"))

        writable("W300 Loss", hm_svoego_roda_loss)(content300, w300_target_hm).__mul__(coefs["borj4_w300"]).minimize_step(cont_opt)
    return do_train


def train(generator, decoder, discriminator, encoder_HG, style_encoder, device, starting_model_number):
    latent_size = 512
    batch_size = 12
    sample_z = torch.randn(8, latent_size, device=device)
    MAFL.batch_size = batch_size
    MAFL.test_batch_size = 64
    Celeba.batch_size = batch_size

    test_img = next(LazyLoader.mafl().loader_train_inf)["data"][:8].cuda()

    loss_st: StyleGANLoss = StyleGANLoss(discriminator)
    model = CondStyleGanModel(generator, loss_st, (0.001, 0.0015))

    style_opt = optim.Adam(style_encoder.parameters(), lr=5e-4, betas=(0.9, 0.99))
    cont_opt = optim.Adam(encoder_HG.parameters(), lr=2e-5, betas=(0.5, 0.97))

    g_transforms: albumentations.DualTransform = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.Compose([
            ResizeMask(h=256, w=256),
            albumentations.ElasticTransform(p=0.7, alpha=150, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.7, rotate_limit=15),
            ResizeMask(h=64, w=64),
            NormalizeMask(dim=(0, 1, 2))
        ])),
        ToTensor(device),
    ])

    R_t = DualTransformRegularizer.__call__(
        g_transforms, lambda trans_dict, img:
        # rt_loss(encoder_HG(trans_dict['image']), trans_dict['mask'])
        stariy_hm_loss(encoder_HG(trans_dict['image']), trans_dict['mask'])
    )

    R_s = UnoTransformRegularizer.__call__(
        g_transforms, lambda trans_dict, img, ltnt:
        L1("R_s")(ltnt, style_encoder(trans_dict['image']))
    )

    barycenter: UniformMeasure2D01 = UniformMeasure2DFactory.load(f"{Paths.default.models()}/face_barycenter_5").cuda().batch_repeat(batch_size)

    R_b = BarycenterRegularizer.__call__(barycenter, 1.0, 2.0, 4.0)
    tuner = GoldTuner([0.37, 1.55, 0.9393, 0.1264, 1.7687, 0.8648, 1.8609], device=device, rule_eps=0.01/2, radius=0.1, active=True)

    heatmaper = ToGaussHeatMap(64, 1.0)
    sparse_bc = heatmaper.forward(barycenter.coord * 63)
    sparse_bc = nn.Upsample(scale_factor=4)(sparse_bc).sum(dim=1, keepdim=True).repeat(1, 3, 1, 1) * \
                       torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1)
    sparse_bc = (sparse_bc - sparse_bc.min()) / sparse_bc.max()
    send_images_to_tensorboard(writer, sparse_bc, "BC", 0, normalize=False, range=(0, 1))

    trainer_gan = gan_trainer(model, generator, decoder, encoder_HG, style_encoder, R_s, style_opt, heatmaper, g_transforms)
    content_trainer = content_trainer_with_gan(cont_opt, tuner, heatmaper, encoder_HG, R_b, R_t, model, generator, g_transforms)
    supervise_trainer = content_trainer_supervised(cont_opt, encoder_HG, LazyLoader.mafl().loader_train_inf)

    for i in range(100000):
        counter.update(i)

        requires_grad(encoder_HG, False)  # REMOVE BEFORE TRAINING
        real_img = next(LazyLoader.mafl().loader_train_inf)["data"].to(device) \
            if i % 5 == 0 else next(LazyLoader.celeba().loader).to(device)

        img_content = encoder_HG(real_img)
        pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(img_content)
        sparse_hm = heatmaper.forward(pred_measures.coord * 63).detach()
        trainer_gan(i, real_img, pred_measures.detach(), sparse_hm.detach(), apply_g=False)
        supervise_trainer()

        if i % 4 == 0:
            # real_img = next(LazyLoader.mafl().loader_train_inf)["data"].to(device)
            trainer_gan(i, real_img, pred_measures.detach(), sparse_hm.detach(), apply_g=True)
            content_trainer(real_img)

        if i % 100 == 0:
            coefs = json.load(open("../parameters/content_loss_sup.json"))
            print(i, coefs)
            with torch.no_grad():

                # pred_measures_test, sparse_hm_test = encoder_HG(test_img)
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
            # test_loss = nadbka(encoder_HG)
            tuner.update(test_loss)
            writer.add_scalar("liuboff", test_loss, i)

        if i % 10000 == 0 and i > 0:
            torch.save(
                {
                    'g': generator.module.state_dict(),
                    'd': discriminator.module.state_dict(),
                    'c': encoder_HG.module.state_dict(),
                    "s": style_encoder.state_dict()
                },
                f'{Paths.default.models()}/stylegan2_MAFL_{str(i + starting_model_number).zfill(6)}.pt',
            )


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)
    encoder_HG = HG_softmax2020(num_classes=5, heatmap_size=64)
    # encoder_HG.load_state_dict(torch.load("/home/ibespalov/pomoika/hg2_e29.pt", map_location="cpu"))

    print("HG")

    latent = 512
    n_mlp = 5
    size = 256

    generator = CondGen3(Generator(
        size, latent, n_mlp, channel_multiplier=1,
    ), heatmap_channels=5)

    discriminator = CondDisc3(
        size, heatmap_channels=5, channel_multiplier=1
    )

    style_encoder = StyleEncoder(style_dim=latent)

    starting_model_number = 190000 #170000
    weights = torch.load(
        f'{Paths.default.models()}/stylegan2_MAFL_{str(starting_model_number).zfill(6)}.pt',
        # f'{Paths.default.models()}/zhores/stylegan2_w300_{str(starting_model_number).zfill(6)}.pt',
        map_location="cpu"
    )

    discriminator.load_state_dict(weights['d'])
    generator.load_state_dict(weights['g'])
    style_encoder.load_state_dict(weights['s'])
    encoder_HG.load_state_dict(weights['c'])

    generator = generator.cuda()
    discriminator = discriminator.to(device)
    encoder_HG = encoder_HG.cuda()
    style_encoder = style_encoder.cuda()
    decoder = CondGenDecode(generator)

    generator = nn.DataParallel(generator, [0, 2, 3])
    discriminator = nn.DataParallel(discriminator, [0, 2, 3])
    encoder_HG = nn.DataParallel(encoder_HG, [0, 2, 3])
    decoder = nn.DataParallel(decoder, [0, 2, 3])

    #encoder_HG = GHSparse(encoder_HG)

    # discriminator.load_state_dict(weights['d'])
    # generator.load_state_dict(weights['g'])
    # style_encoder.load_state_dict(weights['style'])

    train(generator, decoder, discriminator, encoder_HG, style_encoder, device, starting_model_number)
