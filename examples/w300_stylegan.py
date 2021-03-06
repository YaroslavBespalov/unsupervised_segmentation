import sys, os

import albumentations

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../dataset'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from dataset.toheatmap import heatmap_to_measure
from modules.hg import HG_softmax2020

from dataset.lazy_loader import LazyLoader, Celeba
from gan.loss.base import StyleGANLoss
from loss.regulariser import UnoTransformRegularizer
from transforms_utils.transforms import ToNumpy, NumpyBatch, ToTensor

import random
import time
from typing import Optional, Callable, Any

import torch
from torch import nn, optim, Tensor
from torch.utils.tensorboard import SummaryWriter

from dataset.probmeasure import ProbabilityMeasure
from gan.gan_model import CondStyleGanModel, \
    CondGen3, CondDisc3, requires_grad
from gan.loss_base import Loss
from metrics.writers import ItersCounter, send_images_to_tensorboard
from nn.munit.enc_dec import StyleEncoder
from stylegan2.model import Generator

counter = ItersCounter()
writer = SummaryWriter(f"/trinity/home/n.buzun/runs/stylegan{int(time.time())}")
l1_loss = nn.L1Loss()

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

def L1(name: Optional[str], writer: SummaryWriter = writer) -> Callable[[Tensor, Tensor], Loss]:

    if name:
        counter.active[name] = True

    def compute(t1: Tensor, t2: Tensor):
        loss = l1_loss(t1, t2)
        if name:
            writer.add_scalar(name, loss, counter.get_iter(name))
        return Loss(loss)

    return compute


def HMLoss(name: Optional[str], weight: float) -> Callable[[Tensor, Tensor], Loss]:

    if name:
        counter.active[name] = True

    def compute(content: Tensor, target_hm: Tensor):

        content_xy, _ = heatmap_to_measure(content)
        target_xy, _ = heatmap_to_measure(target_hm)

        lossyash = Loss(
            nn.BCELoss()(content, target_hm) * weight +
            nn.MSELoss()(content_xy, target_xy) * weight * 0.001
        )

        if name:
            writer.add_scalar(name, lossyash.item(), counter.get_iter(name))

        return lossyash

    return compute


def writable(name: str, f: Callable[[Any], Loss]):
    counter.active[name] = True

    def decorated(*args, **kwargs) -> Loss:
        loss = f(*args, **kwargs)
        writer.add_scalar(name, loss.item(), counter.get_iter(name))
        return loss

    return decorated


def imgs_with_mask(imgs, mask, color=[1.0,1.0,1.0]):
    # mask = torch.cat([mask, mask, mask], dim=1)
    mask = mask[:, 0, :, :]
    res: Tensor = imgs.cpu().detach()
    res = res.permute(0, 2, 3, 1)
    res[mask > 0.00001, :] = torch.tensor(color, dtype=torch.float32)
    res = res.permute(0, 3, 1, 2)

    return res


def train(generator, discriminator, encoder, style_encoder, device, starting_model_number):

    batch = 32
    Celeba.batch_size = batch

    latent_size = 512
    model = CondStyleGanModel(generator, StyleGANLoss(discriminator), (0.001, 0.0015))

    style_opt = optim.Adam(style_encoder.parameters(), lr=5e-4, betas=(0.5, 0.97))

    g_transforms: albumentations.DualTransform = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.ElasticTransform(p=0.8, alpha=150, alpha_affine=1, sigma=10)),
        NumpyBatch(albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10)),
        ToTensor(device)
    ])

    R_s = UnoTransformRegularizer.__call__(
        g_transforms, lambda trans_dict, img, ltnt:
        L1("R_s")(ltnt, style_encoder(trans_dict['image']))
    )

    sample_z = torch.randn(batch, latent_size, device=device)
    test_img = next(LazyLoader.celeba().loader).to(device)
    print(test_img.shape)
    test_cond = encoder(test_img)

    requires_grad(encoder, False)  # REMOVE BEFORE TRAINING

    t_start = time.time()

    for i in range(100000):
        counter.update(i)
        real_img = next(LazyLoader.celeba().loader).to(device)

        img_content = encoder(real_img).detach()

        noise = mixing_noise(batch, latent_size, 0.9, device)
        fake, _ = generator(img_content, noise)

        model.discriminator_train([real_img], [fake.detach()], img_content)

        writable("Generator loss", model.generator_loss)([real_img], [fake], [], img_content)\
            .minimize_step(model.optimizer.opt_min)

        # print("gen train", time.time() - t1)

        if i % 5 == 0 and i > 0:
            noise = mixing_noise(batch, latent_size, 0.9, device)

            img_content = encoder(real_img).detach()
            fake, fake_latent = generator(img_content, noise, return_latents=True)

            fake_latent_test = fake_latent[:, [0, 13], :].detach()
            fake_latent_pred = style_encoder(fake)
            fake_content_pred = encoder(fake)

            restored = generator.module.decode(img_content[:batch//2], style_encoder(real_img[:batch//2]))
            (
                HMLoss("BCE content gan", 5000)(fake_content_pred, img_content) +
                L1("L1 restored")(restored, real_img[:batch//2]) * 50 +
                L1("L1 style gan")(fake_latent_pred, fake_latent_test) * 30 +
                R_s(fake.detach(), fake_latent_pred) * 50
            ).minimize_step(
                model.optimizer.opt_min,
                style_opt
            )

        if i % 100 == 0:
            t_100 = time.time()
            print(i, t_100 - t_start)
            t_start = time.time()
            with torch.no_grad():

                fake_img, _ = generator(test_cond, [sample_z])
                coords, p = heatmap_to_measure(test_cond)
                test_mes = ProbabilityMeasure(p, coords)
                iwm = imgs_with_mask(fake_img, test_mes.toImage(256))
                send_images_to_tensorboard(writer, iwm, "FAKE", i)

                iwm = imgs_with_mask(test_img, test_mes.toImage(256))
                send_images_to_tensorboard(writer, iwm, "REAL", i)

                restored = generator.module.decode(test_cond, style_encoder(test_img))
                send_images_to_tensorboard(writer, restored, "RESTORED", i)

        if i % 10000 == 0 and i > 0:
            torch.save(
                {
                    'g': generator.state_dict(),
                    'd': discriminator.state_dict(),
                    'style': style_encoder.state_dict()
                    # 'enc': cont_style_encoder.state_dict(),
                },
                f'/trinity/home/n.buzun/PycharmProjects/saved/stylegan2_w300_{str(starting_model_number + i).zfill(6)}.pt',
            )


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # second_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)

    encoder_HG = HG_softmax2020(num_classes=68, heatmap_size=64)
    encoder_HG.load_state_dict(torch.load("/trinity/home/n.buzun/PycharmProjects/saved/hg2_e29.pt", map_location="cpu"))
    encoder_HG = encoder_HG.cuda()

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

    starting_model_number = 110000

    generator = generator.cuda()
    discriminator = discriminator.to(device)

    generator = nn.DataParallel(generator, [0, 1, 2, 3])
    discriminator = nn.DataParallel(discriminator, [0, 1, 2, 3])
    encoder_HG = nn.DataParallel(encoder_HG, [0, 1, 2, 3])

    style_encoder = StyleEncoder(style_dim=latent).cuda()

    weights = torch.load(f"/trinity/home/n.buzun/PycharmProjects/saved/stylegan2_w300_{str(starting_model_number).zfill(6)}.pt", map_location="cpu")
    generator.load_state_dict(weights['g'])
    discriminator.load_state_dict(weights['d'])
    style_encoder.load_state_dict(weights['style'])

    print("stylegan")

    train(generator, discriminator, encoder_HG, style_encoder, device, starting_model_number)


