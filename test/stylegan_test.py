import argparse
import unittest, random

import math
import torch
from torch.nn import functional as F
import time

from gan_model import CondStyleDisc2Wrapper, CondStyleGanModel
from loss.gan_loss import StyleGANLoss
from loss_base import Loss
from model import Generator, Discriminator
from stylegan_train import CondGen2

device = "cuda:2"

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
    grad_real = torch.autograd.grad(
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
    grad = torch.autograd.grad(
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


class TestGANModel(unittest.TestCase):

    def train(self, model, generator, discriminator):
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

        args.latent = 512
        args.n_mlp = 4
        args.start_iter = 0

        image = torch.randn(16, 3, 256, 256)

        mean_path_length = 0
        sample_z = torch.randn(16, args.latent, device=device)

        real_img = image.to(device)

        content = torch.randn(16, 140).to(device)
        content = content.detach()

        # D train

        # requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(content, noise)

        fake_pred = discriminator(fake_img.detach(), content)
        real_pred = discriminator(real_img.detach(), content)

        d_logistic_loss_item = Loss(d_logistic_loss(real_pred, fake_pred)).item()

        real_img.requires_grad = True
        content.requires_grad = True
        real_pred = discriminator(real_img, content)
        r1_loss = d_r1_loss(real_pred, real_img, content)

        deregular_loss_item = Loss(args.r1 / 2 * r1_loss * args.d_reg_every).item()

        # G train

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_pred = discriminator(fake_img, content)
        generator_loss_item = Loss(g_nonsaturating_loss(fake_pred)).item()

        path_batch_size = max(1, args.batch // args.path_batch_shrink)
        noise = mixing_noise(
            path_batch_size, args.latent, args.mixing, device
        )


        content.requires_grad = True
        content_path = content[:path_batch_size]
        fake_img, latents = generator(content_path, noise, return_latents=True)

        path_loss, mean_path_length, path_lengths = g_path_regularize(
            fake_img, latents, content_path, mean_path_length
        )

        generator_reg_loss_item = Loss(args.path_regularize * args.g_reg_every * path_loss).item()

        gen_loss, dis_loss = model.train([real_img], content, noise)
        print("GEN_LOSS_NAW: ", gen_loss)
        print("GEN_LOSS_NE_NAW: ", generator_loss_item)
        print("DIS_LOSS_NAW: ", dis_loss)
        print("DIS_LOSS_NE_NAW: ", d_logistic_loss_item + deregular_loss_item)

        # self.assertAlmostEqual(loss_v.item(), loss_v_1.item(), delta=1e-5)
        # self.assertAlmostEqual(loss_g.item(), loss_g1.item(), delta=1e-5)

        return d_logistic_loss_item, deregular_loss_item, generator_loss_item, generator_reg_loss_item

    def StyleGanHandMade(self):
        generator = CondGen2(Generator(
            256, 512, 4, channel_multiplier=1
        )).to(device)

        discriminator = CondStyleDisc2Wrapper(Discriminator(
            256, channel_multiplier=1
        )).to(device)
        loss = StyleGANLoss(discriminator)
        model = CondStyleGanModel(generator, loss)
        self.train(model, generator, discriminator)




