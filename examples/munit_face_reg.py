import sys
import os

from torch.distributions import Normal
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from models.munit.enc_dec import MunitEncoder
import argparse
import time
from itertools import chain
from typing import List, Callable, Optional
import numpy as np
import albumentations
import torch
import torchvision
from metrics.writers import send_to_tensorboard, ItersCounter, tensorboard_scatter
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn
from torch import optim
from torchvision import utils
from dataset.cardio_dataset import SegmentationDataset, MRIImages, ImageMeasureDataset
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from gans_pytorch.gan.gan_model import GANModel, ganmodel_munit, cont_style_munit_enc, ConditionalGANModel, \
    cond_ganmodel_munit
from gans_pytorch.gan.loss.wasserstein import WassersteinLoss
from gans_pytorch.gan.loss.hinge import HingeLoss
from gans_pytorch.gan.noise.normal import NormalNoise
from gans_pytorch.gan.gan_model import stylegan2
from gans_pytorch.optim.min_max import MinMaxParameters, MinMaxOptimizer
from gans_pytorch.stylegan2.model import ConvLayer, EqualLinear, PixelNorm
from gan.loss.gan_loss import GANLossObject
from loss.losses import Samples_Loss
from loss.regulariser import BarycenterRegularizer, DualTransformRegularizer
from loss_base import Loss
from modules.image2measure import ResImageToMeasure
from modules.lambdaf import LambdaF
from modules.cat import Concat
from torchvision import transforms, utils

from modules.uptosize import Uptosize
from parameters.dataset import DatasetParameters
from parameters.deformation import DeformationParameters
from parameters.gan import GanParameters, MunitParameters
from transforms_utils.transforms import MeasureToMask, ToNumpy, ToTensor, MaskToMeasure, NumpyBatch
from useful_utils.save import save_image_with_mask


def imgs_with_mask(imgs, mask):
    mask = torch.cat([mask, mask, mask], dim=1)
    res = imgs.cpu().detach()
    res[mask > 0.00001] = 1
    return res


def send_images_to_tensorboard(data: Tensor, name: str, iter: int, count=8):
    with torch.no_grad():
        grid = make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=True, range=(-1, 1),
            scale_each=False)
        writer.add_image(name, grid, iter)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    parents=[
        DatasetParameters(),
        GanParameters(),
        DeformationParameters(),
        MunitParameters()
    ]
)
args = parser.parse_args()
for k in vars(args):
    print(f"{k}: {vars(args)[k]}")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

image_size = 256

full_dataset = ImageMeasureDataset(
    "/raid/data/celeba",
    "/raid/data/celeba_masks",
    img_transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.05, 0.05)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
)


def content_to_measure(content):
    pred_measures: ProbabilityMeasure = ProbabilityMeasure(
            torch.ones(args.batch_size, 70, device=device) / 70,
            content.reshape(args.batch_size, 70, 2)
        )
    return pred_measures

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [len(full_dataset) - 1000, 1000])

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)

noise_gen = NormalNoise(args.noise_size, device)

gan_model: ConditionalGANModel = cond_ganmodel_munit("hinge", (0.0005, 0.001), args, None)

cont_style_encoder: MunitEncoder = cont_style_munit_enc(
    args,
    "/home/ibespalov/pomoika/munit_content_encoder15.pt",
    "/home/ibespalov/pomoika/munit_style_encoder_1.pt"
)

style_opt = optim.Adam(cont_style_encoder.enc_style.parameters(), lr=5e-4, betas=(0.5, 0.999))

scheduler_1 = StepLR(gan_model.optimizer.opt_min, step_size=1, gamma=0.5)
scheduler_2 = StepLR(gan_model.optimizer.opt_max, step_size=1, gamma=0.5)
scheduler_3 = StepLR(style_opt, step_size=1, gamma=0.5)

counter = ItersCounter()
writer = SummaryWriter(f"/home/ibespalov/pomoika/munit{int(time.time())}")
gan_model.loss_pair = send_to_tensorboard("G", "D", counter=counter, writer=writer)(gan_model.loss_pair)


fabric = ProbabilityMeasureFabric(args.image_size)
barycenter = fabric.load("/home/ibespalov/unsupervised_pattern_segmentation/examples/face_barycenter").cuda().padding(args.measure_size).batch_repeat(args.batch_size)

g_transforms: albumentations.DualTransform = albumentations.Compose([
    MeasureToMask(size=256),
    ToNumpy(),
    NumpyBatch(albumentations.ElasticTransform(p=0.5, alpha=100, alpha_affine=1, sigma=10)),
    NumpyBatch(albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10)),
    ToTensor(device),
    MaskToMeasure(size=256, padding=args.measure_size),

])

R_b = BarycenterRegularizer.__call__(barycenter)
R_t = DualTransformRegularizer.__call__(
    g_transforms, lambda trans_dict:
    Samples_Loss(scaling=0.8, p=1)(content_to_measure(cont_style_encoder(trans_dict['image'])[0]), trans_dict['mask'])
)

R_b.forward = send_to_tensorboard("R_b", counter=counter, writer=writer)(R_b.forward)
R_t.forward = send_to_tensorboard("R_t", counter=counter, writer=writer)(R_t.forward)

deform_array = list(np.linspace(0, 1, 1500))
Whole_Reg = R_t @ deform_array + R_b
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


test_images, _ = next(iter(dataloader))
test_images = test_images.cuda()
test_noise = noise_gen.sample(args.batch_size)

decoder = gan_model.generator.dec
noise_to_latent = gan_model.generator.style

for epoch in range(500):

    print("epoch", epoch)
    if epoch > 0:
        gan_model.model_save(f"/home/ibespalov/pomoika/gan_{epoch}n.pt")
        torch.save(cont_style_encoder.enc_style.state_dict(), f"/home/ibespalov/pomoika/munit_style_encoder_{epoch}n.pt")
        scheduler_1.step(epoch)
        scheduler_2.step(epoch)
        scheduler_3.step(epoch)

    for i, (imgs, masks) in enumerate(dataloader, 0):
        counter.update(i)
        if imgs.shape[0] != args.batch_size:
            continue

        imgs = imgs.cuda()
        content, latent = cont_style_encoder(imgs)

        noise = noise_gen.sample(args.batch_size)
        fake = gan_model.generator(content.detach(), noise)

        fake_latent = cont_style_encoder.enc_style(fake.detach())
        L1("L1 style gan")(fake_latent, noise_to_latent(noise).detach()).__mul__(20).minimize_step(style_opt)

        fake_content, fake_latent = cont_style_encoder(fake)

        gan_model.loss_pair([imgs], [fake], content).add_min_loss(
            L1("L1 content gan")(fake_content, content.detach()) * 5 +
            L1("L1 style gan")(fake_latent, noise_to_latent(noise).detach()) * 10
        ).minimize_step(gan_model.optimizer)

        # restore L1

        restored = decoder(content, latent)
        restored_content, restored_latent = cont_style_encoder(restored)

        (
            L1("L1 image")(restored, imgs) * 20 +
            L1("L1 content")(restored_content, content.detach()) * 2 +
            L1("L1 style")(restored_latent, latent.detach()) * 2
         ).minimize_step(gan_model.optimizer.opt_min, style_opt)

        if i % 100 == 0:
            with torch.no_grad():

                content, latent = cont_style_encoder(test_images)
                pred_measures: ProbabilityMeasure = content_to_measure(content)
                iwm = imgs_with_mask(test_images, pred_measures.toImage(256))
                send_images_to_tensorboard(iwm, "IMGS WITH MASK", i)

                fake = gan_model.generator(content.detach(), test_noise)
                fwm = imgs_with_mask(fake, pred_measures.toImage(256))
                send_images_to_tensorboard(fwm, "FAKE", i)

                restored = decoder(content, latent)
                send_images_to_tensorboard(restored.detach(), "RESTORED", i)













