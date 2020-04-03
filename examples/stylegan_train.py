import sys, os

from gan.loss.gan_loss import StyleGANLoss
from gan.loss.penalties.penalty import DiscriminatorPenalty

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

import argparse
import math
import random
import os
import time
from typing import List

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
from dataset.probmeasure import ProbabilityMeasure
from gan.gan_model import CondStyleDisc2Wrapper, cont_style_munit_enc, CondStyleGanModel
from gan.loss_base import Loss
from metrics.writers import ItersCounter, send_images_to_tensorboard
from models.common import View
from models.munit.enc_dec import MunitEncoder
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


class CondGen2(nn.Module):

    def __init__(self, gen: Generator):
        super().__init__()

        self.gen: Generator = gen

        self.noise = MakeNoise(7, 140, [512, 512, 512, 512, 256, 128, 64])

        self.condition_preproc = nn.Sequential(
            EqualLinear(140, 256 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            View(-1, 4, 4),
            EqualConv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, cond: Tensor, z: List[Tensor], return_latents=False):

        noise = self.noise(cond)
        input = self.condition_preproc(cond)

        for i in range(len(noise)):
            if i > 1 and i % 2 == 0:
                noise[i] = None

        return self.gen(z, condition=input, noise=noise, return_latents=return_latents)


def content_to_measure(content):
    batch_size = content.shape[0]
    pred_measures: ProbabilityMeasure = ProbabilityMeasure(
            torch.ones(batch_size, 70, device=device) / 70,
            content.reshape(batch_size, 70, 2)
        )
    return pred_measures

def imgs_with_mask(imgs, mask):
    mask = torch.cat([mask, mask, mask], dim=1)
    res = imgs.cpu().detach()
    res[mask > 0.00001] = 1
    return res


def train(args, loader, generator, discriminator, device, cont_style_encoder):
    loader = sample_data(loader)

    pbar = range(args.iter)

    sample_z = torch.randn(16, args.latent, device=device)
    test_img = next(loader)[0].to(device)

    writer = SummaryWriter(f"/home/ibespalov/pomoika/stylegan{int(time.time())}")

    loss_st: StyleGANLoss = StyleGANLoss(discriminator)
    model = CondStyleGanModel(generator, loss_st, (0.0015, 0.002))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        real_img = next(loader)[0]
        real_img = real_img.to(device)

        content, _ = cont_style_encoder(real_img)
        content = content.detach()

        # D train

        # requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        # fake_img, _ = generator(content, noise)
        #
        # loss_st.discriminator_loss_with_penalty(
        #     [real_img, content],
        #     [fake_img.detach(), content]
        # ).minimize_step(model.optimizer.opt_max)
        noise_2 = mixing_noise(
                    args.batch // 2, args.latent, args.mixing, device
                )
        model.train([real_img], content, noise)

        print(i)

        # fake_pred = discriminator(fake_img.detach(), content)
        # real_pred = discriminator(real_img, content)
        #
        # Loss(d_logistic_loss(real_pred, fake_pred)).minimize_step(d_optim)
        #
        # d_regularize = i % args.d_reg_every == 0
        # if d_regularize:
        #     real_img.requires_grad = True
        #     content.requires_grad = True
        #     real_pred = discriminator(real_img, content)
        #     r1_loss = d_r1_loss(real_pred, real_img, content)
        #     Loss(r1_loss * args.d_reg_every * args.r1 / 2).minimize_step(d_optim)


        # G train

        # requires_grad(generator, True)
        # requires_grad(discriminator, False)

        # fake_pred = discriminator(fake_img, content)
        # Loss(g_nonsaturating_loss(fake_pred)).minimize_step(model.optimizer.opt_min)

        # g_regularize = i % args.g_reg_every == 0
        # if g_regularize:
        #     path_batch_size = max(1, args.batch // args.path_batch_shrink)
        #     noise = mixing_noise(
        #         path_batch_size, args.latent, args.mixing, device
        #     )
        #
        #     content.requires_grad = True
        #     content_path = content[:path_batch_size]
        #     fake_img, latents = generator(content_path, noise, return_latents=True)
        #
        #     path_loss, mean_path_length, path_lengths = g_path_regularize(
        #         fake_img, latents, content_path, mean_path_length
        #     )
        #
        #     Loss(args.path_regularize * args.g_reg_every * path_loss).minimize_step(model.optimizer.opt_min)

        # accumulate(g_ema, generator, 0.9)

        if i % 100 == 0 and i > 0:
            with torch.no_grad():
                content, _ = cont_style_encoder(test_img)
                fake_img, _ = generator(content, [sample_z])

                pred_measures: ProbabilityMeasure = content_to_measure(content)
                iwm = imgs_with_mask(fake_img, pred_measures.toImage(256))
                send_images_to_tensorboard(writer, iwm, "FAKE", i)

                # fake_img, _ = g_ema(content, [sample_z])
                # send_images_to_tensorboard(writer, fake_img, "FAKE EMA", i)

        if i % 10000 == 0 and i > 0:
            torch.save(
                {
                    'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    # 'g_ema': g_ema.state_dict(),
                    # 'g_optim': g_optim.state_dict(),
                    # 'd_optim': d_optim.state_dict(),
                },
                f'/home/ibespalov/pomoika/stylegan2_measure_v_konce_{str(i).zfill(6)}.pt',
            )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('path', type=str)
    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--batch', type=int, default=16)
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

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    cont_style_encoder: MunitEncoder = cont_style_munit_enc(
        munit_args,
        None, # "/home/ibespalov/pomoika/munit_content_encoder14.pt",
        None  # "/home/ibespalov/pomoika/munit_style_encoder_1.pt"
    ).to(device)

    args.latent = 512
    args.n_mlp = 4

    args.start_iter = 0

    generator = CondGen2(Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    )).to(device)

    discriminator = CondStyleDisc2Wrapper(Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    )).to(device)

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
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageFolder("/raid/data/celeba", transform=transform)

    # dataset = ImageMeasureDataset(
    #     "/raid/data/celeba",
    #     "/raid/data/celeba_masks",
    #     img_transform=transform
    # )

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

    # weights = torch.load("/home/ibespalov/pomoika/stylegan2_030000.pt")
    # g_ema.load_state_dict(weights['g_ema'])
    # generator.load_state_dict(weights['g'])

    train(args, loader, generator, discriminator, device, cont_style_encoder)
