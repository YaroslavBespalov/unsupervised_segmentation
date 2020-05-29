import json
import sys, os

import albumentations


sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from dataset.lazy_loader import LazyLoader, Celeba
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
from typing import List, Optional, Callable, Any

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




def verka(encoder: nn.Module):
    res = []
    for i, (image, lm) in enumerate(LazyLoader.celeba_test(64)):
        content = encoder(image.cuda())
        mes = UniformMeasure2D01(lm.cuda())
        pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(content)
        res.append(Samples_Loss(p=1)(mes, pred_measures).item() * image.shape[0])
    return np.mean(res)/len(LazyLoader.celeba_test(1).dataset)


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

def hm_svoego_roda_loss(pred, target, coef=1.0, l1_coef = 0.0):
    pred_mes = UniformMeasure2DFactory.from_heatmap(pred)
    target_mes = UniformMeasure2DFactory.from_heatmap(target)

    # pred = pred.relu() + 1e-15
    # target[target < 1e-7] = 0
    # target[target > 1 - 1e-7] = 1

    if torch.isnan(pred).any() or torch.isnan(target).any():
        print("nan in hm")
        return Loss.ZERO()

    bce = nn.BCELoss()(pred, target)

    if torch.isnan(bce).any():
        print("nan in bce")
        return Loss.ZERO()

    return Loss(
        bce * coef +
        nn.MSELoss()(pred_mes.coord, target_mes.coord) * (0.0005 * coef) +
        nn.L1Loss()(pred_mes.coord, target_mes.coord) * l1_coef
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


def gan_trainer(model, generator, decoder, encoder_HG, style_encoder, R_s, style_opt):

    def gan_train(i, real_img, img_content):
        batch_size = real_img.shape[0]
        latent_size = 512

        coefs = json.load(open("../parameters/gan_loss.json"))

        noise = mixing_noise(batch_size, latent_size, 0.9, device)
        fake, fake_latent = generator(img_content, noise, return_latents=True)

        model.disc_train([real_img], [fake], img_content)

        writable("Generator loss", model.generator_loss)([real_img], [fake], [], img_content) \
            .minimize_step(model.optimizer.opt_min)

        if i % 5 == 0:
            noise = mixing_noise(batch_size, latent_size, 0.9, device)

            fake, fake_latent = generator(img_content, noise, return_latents=True)

            fake_latent_test = fake_latent[:, [0, 13], :].detach()
            fake_latent_pred = style_encoder(fake)
            fake_content_pred = encoder_HG(fake)

            restored = decoder(img_content, style_encoder(real_img))
            (
                    writable("BCE content gan", hm_svoego_roda_loss)(fake_content_pred, img_content, 5000) * coefs["BCE content gan"] +
                    L1("L1 restored")(restored, real_img) * coefs["L1 restored"] +
                    L1("L1 style gan")(fake_latent_pred, fake_latent_test) * coefs["L1 style gan"] +
                    R_s(fake.detach(), fake_latent_pred) * coefs["R_s"]
            ).minimize_step(
                model.optimizer.opt_min,
                style_opt
            )

    return gan_train


def train_content(cont_opt, tuner_verka, heatmaper, encoder_HG, R_b, R_t, real_img):
    requires_grad(encoder_HG, True)
    img_content = encoder_HG(real_img)
    pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(img_content)
    # sparce_hm = blur(
    #     heatmaper.forward(pred_measures.probability, pred_measures.coord * 63).detach()
    # )
    sparce_hm = heatmaper.forward(pred_measures.coord * 63).detach()

    # sparce_hm = sparse_heatmap(img_content).detach()
    ll = tuner_verka.sum_losses([
        writable("R_b", R_b.__call__)(real_img, pred_measures) * 0.5,
        writable("Sparse", hm_svoego_roda_loss)(img_content, sparce_hm, 500) * 0.5,
        writable("R_t", R_t.__call__)(real_img, sparce_hm) * 2.5
    ])
    # print(i, ll.item())
    ll.minimize_step(cont_opt)


def train(generator, decoder, discriminator, encoder_HG, style_encoder, device, starting_model_number):
    latent_size = 512
    batch_size = 24
    sample_z = torch.randn(8, latent_size, device=device)
    Celeba.batch_size = batch_size
    test_img = next(LazyLoader.celeba().loader)[:8].cuda()

    loss_st: StyleGANLoss = StyleGANLoss(discriminator)
    model = CondStyleGanModel(generator, loss_st, (0.001, 0.0015))

    style_opt = optim.Adam(style_encoder.parameters(), lr=5e-4, betas=(0.9, 0.99))
    cont_opt = optim.Adam(encoder_HG.parameters(), lr=4e-5, betas=(0.5, 0.97))

    g_transforms: albumentations.DualTransform = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.Compose([
            ResizeMask(h=256, w=256),
            albumentations.ElasticTransform(p=0.7, alpha=150, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.7, rotate_limit=10),
            ResizeMask(h=64, w=64),
            NormalizeMask(dim=(0, 1, 2))
        ])),
        ToTensor(device),
    ])

    R_t = DualTransformRegularizer.__call__(
        g_transforms, lambda trans_dict, img:
        hm_svoego_roda_loss(encoder_HG(trans_dict['image']), trans_dict['mask'], 1000, 0.3)
    )

    R_s = UnoTransformRegularizer.__call__(
        g_transforms, lambda trans_dict, img, ltnt:
        L1("R_s")(ltnt, style_encoder(trans_dict['image']))
    )

    barycenter: UniformMeasure2D01 = UniformMeasure2DFactory.load(f"{Paths.default.models()}/face_barycenter_68").cuda().batch_repeat(batch_size)
    # plt.imshow(barycenter.toImage(256)[0][0].detach().cpu().numpy())
    # plt.show()

    R_b = BarycenterRegularizer.__call__(barycenter, 1.0, 2.0, 3.0)

    #                  4.5, 1.2, 1.12, 1.4, 0.07, 2.2
    #                  1.27, 3.55, 5.88, 3.83, 2.17, 0.22, 1.72
    tuner = GoldTuner([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], device=device, rule_eps=0.1, radius=1, active=True)
    # tuner_verka = GoldTuner([3.0, 1.2, 2.0], device=device, rule_eps=0.05, radius=1, active=True)

    best_igor = 100
    heatmaper = ToGaussHeatMap(64, 1.5)

    trainer_gan = gan_trainer(model, generator, decoder, encoder_HG, style_encoder, R_s, style_opt)

    for i in range(100000):
        counter.update(i)

        # decoder = CondGenDecode(generator.module)
        # decoder = nn.DataParallel(decoder, [0, 1, 2])
        # trainer_gan = gan_trainer(model, generator, decoder, encoder_HG, style_encoder, R_s, style_opt)

        requires_grad(encoder_HG, False)  # REMOVE BEFORE TRAINING
        real_img = next(LazyLoader.celeba().loader).to(device)

        img_content = encoder_HG(real_img).detach()

        # if i % 50 == 0:
        #     test_loss = verka(encoder_HG)
        #     if i > 300:
        #         tuner_verka.update(test_loss)
        #     writer.add_scalar("verka", test_loss, i)
        #     plt.imshow(pred_measures.toImage(256)[0][0].detach().cpu().numpy())
        #     plt.show()
        #     plt.imshow(img_content[0].sum(0).detach().cpu().numpy())
        #     plt.show()

        trainer_gan(i, real_img, img_content)


        if i % 3 == 0:
            # TODO: load new real image to reduce dependence between content and fake
            requires_grad(encoder_HG, True)
            img_content = encoder_HG(real_img)
            pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(img_content)
            # TODO: low val to zero
            sparce_hm = heatmaper.forward(pred_measures.coord * 63).detach()

            img_latent = style_encoder(real_img)
            restored = decoder(img_content, img_latent)

            noise1 = mixing_noise(batch_size, latent_size, 0.9, device)
            # noise2 = mixing_noise(batch_size, latent_size, 0.9, device)
            fake1, _ = generator(img_content, noise1)
            # fake2, _ = generator(img_content[:16].detach(), noise2)

            cont_fake1 = encoder_HG(fake1.detach())
            # cont_fake2 = encoder_HG(fake2)
            # TODO: add L1 to Sparse
            coefs = json.load(open("../parameters/content_loss.json"))
            tuner.sum_losses([
                writable("Fake-content D", model.loss.generator_loss)(real=None, fake=[fake1, img_content.detach()]) * coefs["Fake-content D"], # 800
                writable("Real-content D", model.loss.generator_loss)(real=None, fake=[real_img, img_content]) * coefs["Real-content D"],  # 5
                writable("R_b", R_b.__call__)(real_img, pred_measures) * coefs["R_b"],  # 3000
                writable("Sparse", hm_svoego_roda_loss)(img_content, sparce_hm, 500) * coefs["Sparse"], # 500
                writable("R_t", R_t.__call__)(real_img, sparce_hm) * coefs["R_t"],  # 16155
                # L1("L1 content between fake")(cont_fake1, cont_fake2),  # 1e-6
                L1("L1 image")(restored, real_img) * coefs["L1 image"],  # 10
                writable("fake_content loss", hm_svoego_roda_loss)(cont_fake1, img_content.detach(), 500, 0.1) * coefs["fake_content loss"] # 6477
            ]).minimize_step(
                cont_opt
            )

        if i % 100 == 0:
            print(i, coefs)
            with torch.no_grad():

                content_test = encoder_HG(test_img)
                latent_test = style_encoder(test_img)
                pred_measures = UniformMeasure2DFactory.from_heatmap(content_test)

                iwm = imgs_with_mask(test_img, pred_measures.toImage(256))
                send_images_to_tensorboard(writer, iwm, "REAL", i)

                fake_img, _ = generator(content_test, [sample_z])
                iwm = imgs_with_mask(fake_img, pred_measures.toImage(256))
                send_images_to_tensorboard(writer, iwm, "FAKE", i)

                restored = decoder(content_test, latent_test)
                iwm = imgs_with_mask(restored, pred_measures.toImage(256))
                send_images_to_tensorboard(writer, iwm, "RESTORED", i)

        if i % 50 == 0 and i > 0:
            test_loss = verka(encoder_HG)
            tuner.update(test_loss)
            writer.add_scalar("verka", test_loss, i)

        if i % 10000 == 0 and i > 0:
            torch.save(
                {
                    'g': generator.module.state_dict(),
                    'd': discriminator.module.state_dict(),
                    'c': encoder_HG.module.state_dict(),
                    "s": style_encoder.state_dict()
                },
                f'{Paths.default.models()}/stylegan2_new_{str(i + starting_model_number).zfill(6)}.pt',
            )


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)
    encoder_HG = HG_softmax2020(num_classes=68, heatmap_size=64)

    print("HG")

    latent = 512
    n_mlp = 5
    size = 256

    generator = CondGen3(Generator(
        size, latent, n_mlp, channel_multiplier=1
    ))

    discriminator = CondDisc3(
        size, channel_multiplier=1
    )

    style_encoder = StyleEncoder(style_dim=latent)

    starting_model_number = 30000
    weights = torch.load(
        f'{Paths.default.models()}/stylegan2_new_{str(starting_model_number).zfill(6)}.pt',
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

    generator = nn.DataParallel(generator, [0, 1, 3])
    discriminator = nn.DataParallel(discriminator, [0, 1, 3])
    encoder_HG = nn.DataParallel(encoder_HG, [0, 1, 3])
    decoder = nn.DataParallel(decoder, [0, 1, 3])

    train(generator, decoder, discriminator, encoder_HG, style_encoder, device, starting_model_number)
