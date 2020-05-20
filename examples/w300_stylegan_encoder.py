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

from loss.regulariser import UnoTransformRegularizer
from gan.loss.gan_loss import StyleGANLoss
from model import Generator
from dataset.toheatmap import ToHeatMap, heatmap_to_measure
from gans_pytorch.gan.noise.stylegan_noise import mixing_noise

from loss.hmloss import HMLoss

import albumentations
import torch
from torch import optim
from torch import nn, Tensor
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from albumentations.pytorch.transforms import ToTensor as AlbToTensor

from dataset.cardio_dataset import ImageMeasureDataset
from dataset.d300w import ThreeHundredW
from dataset.lazy_loader import LazyLoader, W300DatasetLoader, CelebaWithKeyPoints, Celeba
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from gan.gan_model import cont_style_munit_enc, CondGen3, CondDisc3, CondStyleGanModel
from metrics.writers import ItersCounter
from models.munit.enc_dec import MunitEncoder, StyleEncoder
from modules.hg import hg2, final_preds_untransformed, hg8, hg4, HG_softmax2020
from parameters.dataset import DatasetParameters
from parameters.deformation import DeformationParameters
from parameters.gan import GanParameters, MunitParameters
from gan.loss_base import Loss
from transforms_utils.transforms import MeasureToMask, ToNumpy, ToTensor, MaskToMeasure, NumpyBatch, MeasureToKeyPoints
from useful_utils.save import save_image_with_mask
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

parser = argparse.ArgumentParser(
    parents=[
        DatasetParameters(),
        GanParameters(),
        DeformationParameters(),
        MunitParameters()
    ],
)

munit_args = parser.parse_args()
cont_style_encoder: MunitEncoder = cont_style_munit_enc(
    munit_args,
    None,  # "/home/ibespalov/pomoika/munit_content_encoder15.pt",
    None  # "/home/ibespalov/pomoika/munit_style_encoder_1.pt"
)  # .to(device)


counter = ItersCounter()
writer = SummaryWriter(f"/home/ibespalov/pomoika/stylegan{int(time.time())}")

def writable(name: str, f: Callable[[Any], Loss]):
    counter.active[name] = True
    def decorated(*args, **kwargs) -> Loss:
        loss = f(*args, **kwargs)
        writer.add_scalar(name, loss.item(), counter.get_iter(name))
        return loss
    return decorated

def otdelnaya_function(content: Tensor, measure: ProbabilityMeasure):
    content_cropped = content
    lossyash = Loss((content_cropped - measure.coord).abs().mean())
    return lossyash


def test(encoder):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        mes = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        # HM_test = heatmaper.forward(mes.probability, mes.coord * 63)
        # HM_enc = enc(data)

        # print("L1: ", nn.BCELoss()(HM_enc, HM_test).item())
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        content = heatmap_to_measure(encoder(data))[0]
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((content - mes.coord).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    return sum_loss / len(LazyLoader.w300().test_dataset)

encoder_HG = HG_softmax2020(num_classes=68, heatmap_size=64)
encoder_HG.load_state_dict(torch.load("/home/ibespalov/pomoika/hg2_e29.pt", map_location="cpu"))
encoder_HG = encoder_HG.cuda()
enc_opt = torch.optim.Adam(encoder_HG.parameters(), lr=5e-5, betas=(0.5, 0.97))

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

starting_model_number = 80000


discriminator = discriminator.to(device)
generator = generator.to(device)

discriminator = nn.DataParallel(discriminator, [0, 1, 2])
generator = nn.DataParallel(generator, [0, 1, 2])
encoder_HG = nn.DataParallel(encoder_HG, [0, 1, 2])

model = CondStyleGanModel(generator, StyleGANLoss(discriminator), (0.0006, 0.001))

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
W300DatasetLoader.test_batch_size = 32
Celeba.batch_size = 24

heatmaper = ToHeatMap(64)

for epoch in range(200):
    for i, batch in enumerate(LazyLoader.w300().loader_train):
        counter.update(i + epoch*len(LazyLoader.w300().loader_train))
        data = batch['data'].to(device)
        mes = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        content = encoder_HG(data)
        content_xy, _ = heatmap_to_measure(content)
        target_hm = heatmaper.forward(mes.probability, mes.coord * 63)
        lossyash = Loss(
            nn.BCELoss()(content, target_hm) +
            nn.MSELoss()(content_xy, mes.coord) * 0.0005 +
            (content - target_hm).abs().mean() * 0.1
        )

        lossyash.minimize_step(enc_opt)
        real_img = next(LazyLoader.celeba().loader).to(device)
        content_celeba = encoder_HG(real_img)
        content_celeba_detachted = content_celeba.detach()

        noise = mixing_noise(W300DatasetLoader.batch_size, latent_size, 0.9, device)
        fake, _ = generator(content_celeba_detachted, noise)
        model.disc_train([real_img], [fake.detach()], content_celeba_detachted)

        model.generator_loss([real_img], [fake], [], content_celeba_detachted).minimize_step(model.optimizer.opt_min)

        if i % 5 == 0 and i > 0:
            noise = mixing_noise(W300DatasetLoader.batch_size, latent_size, 0.9, device)

            img_content = encoder_HG(real_img)
            fake, fake_latent = generator(img_content, noise, return_latents=True)

            fake_latent_test = fake_latent[:, [0, 13], :].detach()
            fake_latent_pred = style_encoder(fake)
            fake_content_pred = encoder_HG(fake)

            restored = generator.module.decode(img_content[:W300DatasetLoader.batch_size//2], style_encoder(real_img[:W300DatasetLoader.batch_size//2]))
            (
                HMLoss("BCE content gan", 5000)(fake_content_pred, img_content.detach()) +
                Loss(nn.L1Loss()(restored, real_img[:W300DatasetLoader.batch_size//2]) * 50) +
                Loss(nn.L1Loss()(fake_latent_pred, fake_latent_test) * 25) +
                R_s(fake.detach(), fake_latent_pred) * 50
            ).minimize_step(
                model.optimizer.opt_min,
                style_opt,
            )

            img_content = encoder_HG(real_img)
            fake, fake_latent = generator(img_content, noise, return_latents=True)
            fake_content_pred = encoder_HG(fake)
            # fake, _ = generator(img_content, noise, return_latents=True)
            # model.generator_loss([real_img], [fake], [], img_content).minimize_step(
            #     enc_opt)

            disc_influence = model.loss.generator_loss(real=None, fake=[real_img, img_content]) * 2
            (HMLoss("BCE content gan", 1)(fake_content_pred, img_content.detach()) +
            disc_influence).minimize_step(enc_opt)

        # writer.add_scalar("L1", lossyash.item(), i + epoch*len(LazyLoader.w300().loader_train))
        if i % 100 == 0:
            with torch.no_grad():
                plt.imshow(content[0].sum(0).cpu().detach().numpy())
                plt.show()
                test_loss = test(encoder_HG)
                print(test_loss)
                writer.add_scalar("test_loss", test_loss, i + epoch*len(LazyLoader.w300().loader_train))

    # torch.save(enc.state_dict(), f"/home/ibespalov/pomoika/hg2_e{epoch}.pt")


