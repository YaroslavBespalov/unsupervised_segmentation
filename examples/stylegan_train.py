import sys, os

import albumentations

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from albumentations.pytorch.transforms import ToTensor as AlbToTensor
from loss.tuner import CoefTuner
from gan.loss.gan_loss import StyleGANLoss
from gan.loss.penalties.penalty import DiscriminatorPenalty
from loss.losses import Samples_Loss
from loss.regulariser import DualTransformRegularizer, BarycenterRegularizer
from transforms_utils.transforms import MeasureToMask, ToNumpy, NumpyBatch, ToTensor, MaskToMeasure

import argparse
import math
import random
import os
import time
from typing import List, Optional, Callable

import numpy as np
import torch
from torch import nn, autograd, optim, Tensor
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from dataset.cardio_dataset import ImageMeasureDataset
from dataset.probmeasure import ProbabilityMeasure, ProbabilityMeasureFabric
from gan.gan_model import CondStyleDisc2Wrapper, cont_style_munit_enc, CondStyleGanModel, CondGen2
from gan.loss_base import Loss
from metrics.writers import ItersCounter, send_images_to_tensorboard
from models.common import View
from models.munit.enc_dec import MunitEncoder, StyleEncoder
from models.uptosize import MakeNoise
from parameters.dataset import DatasetParameters
from parameters.deformation import DeformationParameters
from parameters.gan import GanParameters, MunitParameters

try:
    import wandb

except ImportError:
    wandb = None

from stylegan2.model import Generator, Discriminator, EqualLinear, EqualConv2d
# from stylegan2.dataset import MultiResolutionDataset
# from stylegan2.distributed import (
#     get_rank,
#     synchronize,
#     reduce_loss_dict,
#     reduce_sum,
#     get_world_size,
# )

def tuning_process(args, tuner, loader, model, cont_style_encoder, R_b, R_t):
    W2 = Samples_Loss(scaling=0.85, p=2)
    real_img, masks = next(loader)
    real_img = real_img.cuda()
    masks: ProbabilityMeasure = MaskToMeasure(size=256, padding=140).apply_to_mask(masks).cuda()
    # mask_content = masks.coord.reshape(args.batch, 140)

    img_content = cont_style_encoder.enc_content(real_img).detach().requires_grad_(True)

    noise1 = mixing_noise(args.batch, args.latent, args.mixing, device)
    noise2 = mixing_noise(args.batch, args.latent, args.mixing, device)
    fake1, _ = model.generator(img_content, noise1)
    fake2, _ = model.generator(img_content, noise2)

    cont_fake1 = cont_style_encoder.enc_content(fake1)
    cont_fake2 = cont_style_encoder.enc_content(fake2)

    img_latent = cont_style_encoder.enc_style(real_img)
    restored = model.generator.decode(img_content, img_latent)

    # noise = mixing_noise(args.batch, args.latent, args.mixing, real_img.device)
    # fake, _ = generator(img_content, noise)
    # fake_content_pred = cont_style_encoder.get_content(fake)

    lr = 0.001

    tuner.tune_module(
        real_img,
        cont_style_encoder.enc_content,
        [
            # model.loss.generator_loss(real=None, fake=[fake1, img_content.detach()]),
            model.loss.generator_loss(real=None, fake=[real_img, img_content]),
            R_b(real_img, content_to_measure(img_content)),
            R_t(real_img, content_to_measure(img_content)),
            L1("L1 content between fake")(cont_fake1, cont_fake2),
            L1("L1 image")(restored, real_img)
         ],
        lambda x: W2(content_to_measure(x), masks) * 100,
        lr=lr
    )


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img, cond):
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=[real_img, cond], create_graph=True
    )
    batch = real_img.shape[0]
    grad_penalty = grad_real[0].pow(2).view(batch, -1).sum(1).mean() + \
        grad_real[1].pow(2).view(batch, -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, cond, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=[latents, cond], create_graph=True
    )
    path_lengths = torch.sqrt(grad[0].pow(2).sum(2).mean(1) + grad[1].pow(2).sum(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


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


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def content_to_measure(content):
    batch_size = content.shape[0]
    pred_measures: ProbabilityMeasure = ProbabilityMeasure(
            torch.ones(batch_size, 70, device=device) / 70,
            content.reshape(batch_size, 70, 2)
        )
    return pred_measures

def imgs_with_mask(imgs, mask, color=[1.0,1.0,1.0]):
    # mask = torch.cat([mask, mask, mask], dim=1)
    mask = mask[:, 0, :, :]
    res: Tensor = imgs.cpu().detach()
    res = res.permute(0, 2, 3, 1)
    res[mask > 0.00001, :] = torch.tensor(color, dtype=torch.float32)
    res = res.permute(0, 3, 1, 2)

    return res




counter = ItersCounter()
writer = SummaryWriter(f"/home/ibespalov/pomoika/stylegan{int(time.time())}")
l1_loss = nn.L1Loss()

def L1(name: Optional[str], writer: SummaryWriter = writer) -> Callable[[Tensor, Tensor], Loss]:

    if name:
        counter.active[name] = True

    def compute(t1: Tensor, t2: Tensor):
        loss = l1_loss(t1, t2)
        if name:
            writer.add_scalar(name, loss, counter.get_iter(name))
        return Loss(loss)

    return compute


def train(args, loader, generator, discriminator, device, cont_style_encoder, starting_model_number):
    loader = sample_data(loader)

    pbar = range(args.iter)

    sample_z = torch.randn(args.batch, args.latent, device=device)
    test_img, test_mask = next(loader)
    test_img = test_img.cuda()
    test_mask = test_mask.cuda()

    loss_st: StyleGANLoss = StyleGANLoss(discriminator)
    model = CondStyleGanModel(generator, loss_st, (0.001, 0.0014))

    style_opt = optim.Adam(cont_style_encoder.enc_style.parameters(), lr=3e-4, betas=(0.5, 0.9))
    cont_opt = optim.Adam(cont_style_encoder.enc_content.parameters(), lr=3e-5, betas=(0.5, 0.9))

    g_transforms: albumentations.DualTransform = albumentations.Compose([
        MeasureToMask(size=256),
        ToNumpy(),
        NumpyBatch(albumentations.ElasticTransform(p=0.5, alpha=100, alpha_affine=1, sigma=10)),
        NumpyBatch(albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10)),
        ToTensor(device),
        MaskToMeasure(size=256, padding=140),
    ])

    W1 = Samples_Loss(scaling=0.85, p=1)
    # W2 = Samples_Loss(scaling=0.85, p=2)

    # g_trans_res_dict = g_transforms(image=test_img, mask=MaskToMeasure(size=256, padding=140).apply_to_mask(test_mask))
    # g_trans_img = g_trans_res_dict['image']
    # g_trans_mask = g_trans_res_dict['mask']
    # iwm = imgs_with_mask(g_trans_img, g_trans_mask.toImage(256), color=[1, 1, 1])
    # send_images_to_tensorboard(writer, iwm, "RT", 0)

    R_t = DualTransformRegularizer.__call__(
        g_transforms, lambda trans_dict:
        W1(content_to_measure(cont_style_encoder.get_content(trans_dict['image'])), trans_dict['mask']) #+
        # W2(content_to_measure(cont_style_encoder.get_content(trans_dict['image'])), trans_dict['mask'])
    )

    fabric = ProbabilityMeasureFabric(256)
    barycenter = fabric.load("/raid/data/saved_models/barycenter/face_barycenter").cuda().padding(70).batch_repeat(args.batch)
    R_b = BarycenterRegularizer.__call__(barycenter)

    tuner = CoefTuner([2, 10, 4, 0.3, 0.3], device=device)

    for idx in pbar:
        i = idx + args.start_iter
        counter.update(i)

        if i > args.iter:
            print('Done!')
            break

        real_img = next(loader)[0]
        real_img = real_img.to(device)

        img_content = cont_style_encoder.get_content(real_img)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        img_content_variable = img_content.detach().requires_grad_(True)
        fake, fake_latent = generator(img_content_variable, noise, return_latents=True)
        fake_detach = fake.detach()
        fake_latent_test = fake_latent[:, [0, 13], :].detach()
        fake_content_pred = cont_style_encoder.get_content(fake)

        fake_latent_pred = cont_style_encoder.enc_style(fake)

        model.loss_pair([real_img], [fake], [fake_latent], img_content_variable).add_min_loss(
            L1("L1 content gan")(fake_content_pred, img_content.detach()) * 25 +
            L1("L1 style gan")(fake_latent_pred, fake_latent_test) * 0.5
        ).minimize_step(
            model.optimizer
        )

        if i % 5 == 0:
            fake_latent_pred = cont_style_encoder.enc_style(fake_detach)
            L1("L1 style gan")(fake_latent_pred, fake_latent_test).minimize_step(style_opt)
            img_latent = cont_style_encoder.enc_style(real_img)
            restored = model.generator.decode(img_content, img_latent)
            pred_measures: ProbabilityMeasure = content_to_measure(img_content)

            noise1 = mixing_noise(args.batch, args.latent, args.mixing, device)
            noise2 = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake1, _ = generator(img_content, noise1)
            fake2, _ = generator(img_content, noise2)

            cont_fake1 = cont_style_encoder.enc_content(fake1)
            cont_fake2 = cont_style_encoder.enc_content(fake2)

            #TUNER PART
            tuner.sum_losses([
                # model.loss.generator_loss(real=None, fake=[fake1, img_content.detach()]),
                model.loss.generator_loss(real=None, fake=[real_img, img_content]),
                R_b(real_img, pred_measures),
                R_t(real_img, pred_measures),
                L1("L1 content between fake")(cont_fake1, cont_fake2),
                L1("L1 image")(restored, real_img)
            ]).minimize_step(
                cont_opt,
                model.optimizer.opt_min
            )

            ##Without tuner part

            # (
            #         model.loss.generator_loss(real=None, fake=[real_img, img_content]) * 5 +
            #         (R_b + R_t * 0.4)(real_img, pred_measures) * 10 +
            #         L1("L1 content between fake")(cont_fake1, cont_fake2) * 1 +
            #         L1("L1 image")(restored, real_img) * 1
            #         # L1("L1 style gan")(fake_latent_pred, fake_latent_test) * 1
            # ).minimize_step(
            #     cont_opt,
            #     model.optimizer.opt_min
            # )



        if i % 100 == 0:
            print(i)
            with torch.no_grad():
                content, latent = cont_style_encoder(test_img)
                pred_measures: ProbabilityMeasure = content_to_measure(content)
                ref_measures: ProbabilityMeasure = MaskToMeasure(size=256, padding=140).apply_to_mask(test_mask)
                iwm = imgs_with_mask(test_img, ref_measures.toImage(256), color=[0, 0, 1])
                iwm = imgs_with_mask(iwm, pred_measures.toImage(256), color=[1, 1, 1])
                send_images_to_tensorboard(writer, iwm, "REAL", i)

                fake_img, _ = generator(content, [sample_z])
                iwm = imgs_with_mask(fake_img, pred_measures.toImage(256))
                send_images_to_tensorboard(writer, iwm, "FAKE", i)
                restored = model.generator.decode(content, latent)
                send_images_to_tensorboard(writer, restored, "RESTORED", i)

        if i % 100 == 0:
            for ten_raz in range(20):
                try:
                    tuning_process(args, tuner, loader, model, cont_style_encoder, R_b, R_t)
                except Exception as e:
                    print(e)

            print("coefficients: ", tuner.coefs.detach().cpu().numpy())


        if i % 10000 == 0 and i > 0:
            torch.save(
                {
                    'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    'enc': cont_style_encoder.state_dict(),
                    # 'g_ema': g_ema.state_dict(),
                    # 'g_optim': g_optim.state_dict(),
                    # 'd_optim': d_optim.state_dict(),
                },
                f'/home/ibespalov/pomoika/stylegan2_invertable_{str(i + starting_model_number).zfill(6)}.pt',
            )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('path', type=str)
    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--channel_multiplier', type=int, default=1)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    parser = argparse.ArgumentParser(
        parents=[
            DatasetParameters(),
            GanParameters(),
            DeformationParameters(),
            MunitParameters()
        ],
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    munit_args = parser.parse_args()

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)

    cont_style_encoder: MunitEncoder = cont_style_munit_enc(
        munit_args,
        None, # "/home/ibespalov/pomoika/munit_content_encoder15.pt",
        None  # "/home/ibespalov/pomoika/munit_style_encoder_1.pt"
    )#.to(device)

    args.latent = 512
    args.n_mlp = 5

    args.start_iter = 0

    generator = CondGen2(Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ))#.to(device)

    discriminator = CondStyleDisc2Wrapper(Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ))#.to(device)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    image_size = args.size
    transform = albumentations.Compose(
        [
            albumentations.HorizontalFlip(),
            albumentations.Resize(image_size, image_size),
            albumentations.ElasticTransform(p=0.5, alpha=100, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            AlbToTensor()
        ]
    )



    # dataset = ImageFolder("/raid/data/celeba", transform=transform)

    dataset = ImageMeasureDataset(
        "/raid/data/celeba",
        "/raid/data/celeba_masks",
        img_transform=transform
    )

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
    )

    # g_ema = CondGen2(Generator(
    #     args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    # )).to(device)
    # g_ema.eval()
    # accumulate(g_ema, generator, 0)

    starting_model_number = 320000
    weights = torch.load(f"/home/ibespalov/pomoika/stylegan2_invertable_{starting_model_number}.pt", map_location="cpu")
    generator.load_state_dict(weights['g'])
    discriminator.load_state_dict(weights['d'])
    cont_style_encoder.load_state_dict(weights['enc'])

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    cont_style_encoder = cont_style_encoder.to(device)

    train(args, loader, generator, discriminator, device, cont_style_encoder, starting_model_number)
