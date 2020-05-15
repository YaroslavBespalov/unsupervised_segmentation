import sys, os
from itertools import chain

import albumentations

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../dataset'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from dataset.d300w import ThreeHundredW
from albumentations.pytorch.transforms import ToTensor as AlbToTensor
from loss.tuner import CoefTuner, GoldTuner
from dataset.lazy_loader import LazyLoader
from gan.loss.gan_loss import StyleGANLoss
from gan.loss.penalties.penalty import DiscriminatorPenalty
from loss.losses import Samples_Loss
from loss.regulariser import DualTransformRegularizer, BarycenterRegularizer, StyleTransformRegularizer
from transforms_utils.transforms import MeasureToMask, ToNumpy, NumpyBatch, ToTensor, MaskToMeasure

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
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from dataset.cardio_dataset import ImageMeasureDataset
from dataset.probmeasure import ProbabilityMeasure, ProbabilityMeasureFabric
from gan.gan_model import CondStyleDisc2Wrapper, cont_style_munit_enc, CondStyleGanModel, CondGen2, cond_ganmodel_munit
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

class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder: MunitEncoder = encoder
        self.layer1 = nn.Sequential(
            nn.Linear(140, 136),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layer1(self.encoder.get_content(input)).view(-1, 68, 2)

    def parameters(self, recurse=True):
        return self.layer1.parameters()


def otdelnaya_function(content: Tensor, measure: ProbabilityMeasure):
    content_cropped = content
    lossyash = Loss(nn.L1Loss()(content_cropped, measure.coord))
    return lossyash


def test(cont_style_encoder, pairs):
    W1 = Samples_Loss(scaling=0.9, p=1)
    err_list = []
    for img, masks in pairs:
        mes: ProbabilityMeasure = MaskToMeasure(size=256, padding=140).apply_to_mask(masks).cuda()
        real_img = img.cuda()
        img_content = cont_style_encoder.get_content(real_img).detach()
        err_list.append(W1(content_to_measure(img_content), mes).item())

    print("test:", sum(err_list)/len(err_list))
    return sum(err_list)/len(err_list)



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


def writable(name: str, f: Callable[[Any], Loss]):
    counter.active[name] = True

    def decorated(*args, **kwargs) -> Loss:
        loss = f(*args, **kwargs)
        writer.add_scalar(name, loss.item(), counter.get_iter(name))
        return loss

    return decorated


def testw300(encoder_content_wrapper):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        mes = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        content = encoder_content_wrapper(data)
        content_cropped = content
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((content_cropped - mes.coord).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    return sum_loss / len(LazyLoader.w300().test_dataset)


def train(args, generator, discriminator, device, cont_style_encoder, starting_model_number):
    test_img, test_mask = next(LazyLoader.celeba_with_kps().loader)
    test_img = test_img.cuda()
    test_mask = test_mask.cuda()
    test_pairs = [next(LazyLoader.celeba_with_kps().loader) for _ in range(50)]

    encoder_content_wrapper = EncoderWrapper(cont_style_encoder).cuda()
    wrapper_cont_opt = optim.Adam(encoder_content_wrapper.parameters(), lr=1e-4, betas=(0.5, 0.9))


    pbar = range(args.iter)
    sample_z = torch.randn(args.batch, args.latent, device=device)

    loss_st: StyleGANLoss = StyleGANLoss(discriminator)
    model = CondStyleGanModel(generator, loss_st, (0.0006, 0.001))

    style_opt = optim.Adam(cont_style_encoder.enc_style.parameters(), lr=1e-3, betas=(0.5, 0.9))
    cont_opt = optim.Adam(cont_style_encoder.enc_content.parameters(), lr=2e-5, betas=(0.5, 0.9))

    g_transforms: albumentations.DualTransform = albumentations.Compose([
        MeasureToMask(size=256),
        ToNumpy(),
        NumpyBatch(albumentations.ElasticTransform(p=0.8, alpha=150, alpha_affine=1, sigma=10)),
        NumpyBatch(albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10)),
        ToTensor(device),
        MaskToMeasure(size=256, padding=140),
    ])

    W1 = Samples_Loss(scaling=0.85, p=1)

    R_t = DualTransformRegularizer.__call__(
        g_transforms, lambda trans_dict, img:
        W1(content_to_measure(cont_style_encoder.get_content(trans_dict['image'])), trans_dict['mask']) # +
        # W2(content_to_measure(cont_style_encoder.get_content(trans_dict['image'])), trans_dict['mask'])
    )

    R_s = DualTransformRegularizer.__call__(
        g_transforms, lambda trans_dict, img:
        L1("R_s")(cont_style_encoder.enc_style(img), cont_style_encoder.enc_style(trans_dict['image']))  # +
        # W2(content_to_measure(cont_style_encoder.get_content(trans_dict['image'])), trans_dict['mask'])
    )

    fabric = ProbabilityMeasureFabric(256)
    barycenter = fabric.load("/raid/data/saved_models/barycenter/face_barycenter").cuda().padding(70).transpose().batch_repeat(args.batch)

    R_b = BarycenterRegularizer.__call__(barycenter)

    tuner = GoldTuner([4.53, 9.97, 5.5, 0.01, 9.44, 1.05, 4.9], device=device, rule_eps=0.05, radius=2, active=False)
    gan_tuner = GoldTuner([70, 5, 22], device=device, rule_eps=1, radius=20, active=False)


    best_igor = 100

    for idx in pbar:
        i = idx + args.start_iter
        counter.update(i)

        if i > args.iter:
            print('Done!')
            break

        real_img = next(LazyLoader.celeba_with_kps().loader)[0].to(device)

        img_content = cont_style_encoder.get_content(real_img)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        img_content_variable = img_content.detach().requires_grad_(True)
        fake, fake_latent = generator(img_content_variable, noise, return_latents=True)

        model.disc_train([real_img], [fake], img_content)

        # fake_detach = fake.detach()
        fake_latent_test = fake_latent[:, [0, 13], :].detach()
        fake_content_pred = cont_style_encoder.get_content(fake)

        fake_latent_pred = cont_style_encoder.enc_style(fake)

        (
            writable("Generator loss", model.generator_loss)([real_img], [fake], [fake_latent], img_content_variable) +  # 3e-5
            gan_tuner.sum_losses([
                L1("L1 content gan")(fake_content_pred, img_content.detach()),
                L1("L1 style gan")(fake_latent_pred, fake_latent_test),
                R_s(fake, barycenter)
            ])
            # L1("L1 content gan")(fake_content_pred, img_content.detach()) * 50 +  # 3e-7
            # L1("L1 style gan")(fake_latent_pred, fake_latent_test) * 10 +  # 8e-7
            # R_s(fake, barycenter) * 20
        ).minimize_step(
            model.optimizer.opt_min,
            style_opt
        )

        if i % 5 == 0:

            img_latent = cont_style_encoder.enc_style(real_img)
            restored = model.generator.decode(img_content, img_latent)
            pred_measures: ProbabilityMeasure = content_to_measure(img_content)

            noise1 = mixing_noise(args.batch, args.latent, args.mixing, device)
            noise2 = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake1, _ = generator(img_content, noise1)
            fake2, _ = generator(img_content, noise2)

            cont_fake1 = cont_style_encoder.get_content(fake1)
            cont_fake2 = cont_style_encoder.get_content(fake2)

            tuner.sum_losses([
                writable("Real-content D", model.loss.generator_loss)(real=None, fake=[real_img, img_content]),  # 3e-5
                writable("R_b", R_b.__call__)(real_img, pred_measures),  # 7e-5
                writable("R_t", R_t.__call__)(real_img, pred_measures),  # -
                L1("L1 content between fake")(cont_fake1, cont_fake2),  # 1e-6
                L1("L1 image")(restored, real_img),  # 4e-5
                R_s(real_img, pred_measures),
                L1("L1 style restored")(cont_style_encoder.enc_style(restored), img_latent.detach())
            ]).minimize_step(
                cont_opt,
                model.optimizer.opt_min,
                style_opt
            )

        batch = next(LazyLoader.w300().loader_train_inf)
        real_img_w300 = batch['data'].cuda()
        measure_w300 = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        content = encoder_content_wrapper(real_img_w300)
        lossyash = writable("L2", otdelnaya_function)(content, measure_w300) * 2
        lossyash.minimize_step(wrapper_cont_opt, cont_opt)


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

                test_loss = testw300(encoder_content_wrapper)
                writer.add_scalar("test_w300_loss", test_loss, i)

        if i % 100 == 0 and i > 0:
            with torch.no_grad():
                igor = test(cont_style_encoder, test_pairs)
                writer.add_scalar("test error", igor, i)
                tuner.update(igor)
                gan_tuner.update(igor)

            if igor < best_igor:
                best_igor = igor
                print("best igor")
                torch.save(
                    {
                        'g': generator.state_dict(),
                        'd': discriminator.state_dict(),
                        'enc': cont_style_encoder.state_dict(),
                    },
                    f'/home/ibespalov/pomoika/stylegan2_igor_3.pt',
                )

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

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)


    args.latent = 512
    args.n_mlp = 5

    args.start_iter = 0

    starting_model_number = 250000

    generator, discriminator, cont_style_encoder = cond_ganmodel_munit(
        args,
        munit_args,
        path='/home/ibespalov/pomoika/',
        starting_model_number=starting_model_number
    )

    print("GOTOVO")
    train(args, generator, discriminator, device, cont_style_encoder, starting_model_number)
