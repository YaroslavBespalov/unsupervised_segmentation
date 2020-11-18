import argparse
import time
from itertools import chain
from typing import Callable, Any, List
import sys
import os

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../dataset'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from loss.tuner import GoldTuner
from loss.regulariser import UnoTransformRegularizer, DualTransformRegularizer
from gan.loss.base import StyleGANLoss, StyleGANLossWithoutPenalty
from model import Generator
from dataset.toheatmap import ToHeatMap, heatmap_to_measure
from gans_pytorch.gan.noise.stylegan import mixing_noise
from loss.hmloss import HMLoss
import albumentations
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch import nn, Tensor
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from dataset.lazy_loader import LazyLoader, W300DatasetLoader, CelebaWithKeyPoints, Celeba
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure, UniformMeasure2DFactory
from gan.gan_model import cont_style_munit_enc, CondGen3, CondDisc3, CondStyleGanModel
from metrics.writers import ItersCounter, send_images_to_tensorboard
from models.munit.enc_dec import MunitEncoder, StyleEncoder
from modules.hg import hg2, final_preds_untransformed, hg8, hg4, HG_softmax2020
from gan.loss_base import Loss
from transforms_utils.transforms import MeasureToMask, ToNumpy, ToTensor, MaskToMeasure, NumpyBatch, MeasureToKeyPoints, \
    ResizeMask, NormalizeMask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

counter = ItersCounter()
writer = SummaryWriter(f"/home/ibespalov/pomoika/KKKKKstylegan{int(time.time())}")


def writable(name: str, f: Callable[[Any], Loss]):
    counter.active[name] = True

    def decorated(*args, **kwargs) -> Loss:
        loss = f(*args, **kwargs)
        it = counter.get_iter(name)
        if it % 10 == 0:
            writer.add_scalar(name, loss.item(), it)
        return loss

    return decorated


def test(encoder):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        content = heatmap_to_measure(encoder(data))[0]
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((content - landmarks).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    return sum_loss / len(LazyLoader.w300().test_dataset)

def imgs_with_mask(imgs, mask, color=[1.0,1.0,1.0]):
    # mask = torch.cat([mask, mask, mask], dim=1)
    mask = mask[:, 0, :, :]
    res: Tensor = imgs.cpu().detach()
    res = res.permute(0, 2, 3, 1)
    res[mask > 0.00001, :] = torch.tensor(color, dtype=torch.float32)
    res = res.permute(0, 3, 1, 2)
    return res

def hm_svoego_roda_loss(pred, target):
    pred_coord = heatmap_to_measure(pred)[0]
    target_coord = heatmap_to_measure(target)[0]

    pred = pred.relu() + 1e-15
    target[target < 1e-7] = 0
    target[target > 1 - 1e-7] = 1

    if torch.isnan(pred).any() or torch.isnan(target).any():
        return Loss.ZERO()

    bce = nn.BCELoss()(pred, target)

    if torch.isnan(bce).any():
        return Loss.ZERO()

    return Loss(
        bce +
        nn.MSELoss()(pred_coord, target_coord) * 0.0005
    )


encoder_HG = HG_softmax2020(num_classes=68, heatmap_size=64)
encoder_HG.load_state_dict(torch.load("/home/ibespalov/pomoika/hg2_e29.pt", map_location="cpu"))
encoder_HG = encoder_HG.cuda()
enc_opt = torch.optim.Adam(encoder_HG.parameters(), lr=3e-5, betas=(0.5, 0.95))

latent = 512
n_mlp = 5
size = 256
latent_size = 512

style_encoder = StyleEncoder(style_dim=latent).cuda()

discriminator = CondDisc3(
    size, channel_multiplier=1
)


generator = CondGen3(Generator(
    size, latent, n_mlp, channel_multiplier=1
))


discriminator = discriminator.to(device)
generator = generator.to(device)

discriminator = nn.DataParallel(discriminator, [0, 1, 3])
generator = nn.DataParallel(generator, [0, 1, 3])
encoder_HG = nn.DataParallel(encoder_HG, [0, 1, 3])

model = CondStyleGanModel(generator, StyleGANLoss(discriminator), (0.0006, 0.001))
loss_without_penalty = StyleGANLossWithoutPenalty(discriminator)
weights = torch.load(f"/home/ibespalov/pomoika/zhores/stylegan2_w300_160000.pt", map_location="cpu")
discriminator.load_state_dict(weights['d'])
generator.load_state_dict(weights['g'])
style_encoder.load_state_dict(weights['style'])
style_opt = torch.optim.Adam(style_encoder.parameters(), lr=5e-4, betas=(0.5, 0.97))

g_transforms: albumentations.DualTransform = albumentations.Compose([
    ToNumpy(),
    NumpyBatch(albumentations.ElasticTransform(p=0.8, alpha=150, alpha_affine=1, sigma=10)),
    NumpyBatch(albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10)),
    ToTensor(device)
])

R_s = UnoTransformRegularizer.__call__(
    g_transforms, lambda trans_dict, img, ltnt:
    Loss(nn.L1Loss()(ltnt, style_encoder(trans_dict['image'])))
)

W300DatasetLoader.batch_size = 24
W300DatasetLoader.test_batch_size = 64
Celeba.batch_size = 24

heatmaper = ToHeatMap(64)

# tuner = GoldTuner([1.0, 1.0], device=device, rule_eps=0.02, radius=0.5, active=True)

w300_test = next(iter(LazyLoader.w300().test_loader))
w300_test_image = w300_test['data'].to(device)[:8]


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
    hm_svoego_roda_loss(encoder_HG(trans_dict['image']), trans_dict['mask'], 1, 0.1)
)

for i in range(100000):

    counter.update(i)

    w300_batch = next(LazyLoader.w300().loader_train_inf)
    w300_image = w300_batch['data'].to(device)
    w300_mes = ProbabilityMeasureFabric(256).from_coord_tensor(w300_batch["meta"]["keypts_normalized"]).cuda()
    w300_target_hm = heatmaper.forward(w300_mes.probability, w300_mes.coord * 63).detach()

    content300 = encoder_HG(w300_image)

    loss_or_none = (
        writable("real_content loss", hm_svoego_roda_loss)(content300, w300_target_hm) +
        writable("R_t", R_t.__call__)(w300_image, content300) * 0.05
    )

    loss_or_none.minimize_step(enc_opt)

    # real_img = next(LazyLoader.celeba().loader).to(device)
    #
    # content_celeba = encoder_HG(real_img)
    # content_celeba_detachted = content_celeba.detach()
    #
    # noise = mixing_noise(W300DatasetLoader.batch_size, latent_size, 0.9, device)
    # fake, _ = generator(content_celeba_detachted, noise)
    #
    # # fake_content = encoder_HG(fake)
    #
    # (
    #     # model.loss.generator_loss(real=None, fake=[real_img, content_celeba]) * 0.1,
    #     writable("fake_content loss", hm_svoego_roda_loss)(fake_content, content_celeba_detachted)
    # ).minimize_step(enc_opt)


    #     loss_without_penalty._discriminator_loss(discriminator(real_img, content_celeba), discriminator(fake.detach(), content_celeba)) * (-5)


        # model.generator_loss([real_img], [fake], [], content_celeba_detachted).minimize_step(model.optimizer.opt_min)

        # if i % 5 == 0 and i > 0:
        #     noise = mixing_noise(W300DatasetLoader.batch_size, latent_size, 0.9, device)
        #
        #     img_content = encoder_HG(real_img)
        #     fake, fake_latent = generator(img_content, noise, return_latents=True)
        #
        #     fake_latent_test = fake_latent[:, [0, 13], :].detach()
        #     fake_latent_pred = style_encoder(fake)
        #     fake_content_pred = encoder_HG(fake)

            # restored = generator.module.decode(img_content[:W300DatasetLoader.batch_size//2], style_encoder(real_img[:W300DatasetLoader.batch_size//2]))
            # (
            #     HMLoss("BCE content gan", 5000)(fake_content_pred, img_content.detach()) +
            #     Loss(nn.L1Loss()(restored, real_img[:W300DatasetLoader.batch_size//2]) * 50) +
            #     Loss(nn.L1Loss()(fake_latent_pred, fake_latent_test) * 25) +
            #     R_s(fake.detach(), fake_latent_pred) * 50
            # ).minimize_step(
            #     model.optimizer.opt_min,
            #     style_opt,
            # )

            # img_content = encoder_HG(real_img)
            # fake, fake_latent = generator(img_content, noise, return_latents=True)
            # fake_content_pred = encoder_HG(fake)
            #
            #
            # disc_influence = model.loss.generator_loss(real=None, fake=[real_img, img_content]) * 2
            # (HMLoss("BCE content gan", 1)(fake_content_pred, img_content.detach()) +
            # disc_influence).minimize_step(enc_opt)

    if i % 50 == 0 and i > 0:
        with torch.no_grad():
            test_loss = test(encoder_HG)
            print(test_loss)
            # tuner.update(test_loss)
            coord, p = heatmap_to_measure(encoder_HG(w300_test_image))
            pred_measure = ProbabilityMeasure(p, coord)
            iwm = imgs_with_mask(w300_test_image, pred_measure.toImage(256))
            send_images_to_tensorboard(writer, iwm, "W300_test_image", i)
            writer.add_scalar("test_loss", test_loss, i)

    # torch.save(enc.state_dict(), f"/home/ibespalov/pomoika/hg2_e{epoch}.pt")


