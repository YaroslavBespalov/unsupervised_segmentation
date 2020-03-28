import sys
import os

from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# from munit.networks import StyleEncoder, ContentEncoder

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/munit/'))

import argparse
import time
from itertools import chain
from typing import List, Callable
import numpy as np
import albumentations
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
from metrics.writers import send_to_tensorboard, ItersCounter, tensorboard_scatter
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn
from torch import optim
from torchvision import utils
from dataset.cardio_dataset import SegmentationDataset, MRIImages, ImageMeasureDataset
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from gans_pytorch.gan.gan_model import GANModel, ganmodel_munit, cont_style_munit_enc
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

noise = NormalNoise(args.noise_size, device)

gan_model: GANModel = ganmodel_munit("hinge", (0.0004, 0.0004), args)

# enc_style = StyleEncoder(n_downsample=4, input_dim=args.input_dim, dim=args.dim, style_dim=args.style_dim,
#                          norm=args.norm, activ=args.activ, pad_type=args.pad_type).cuda()
# enc_content: ContentEncoder = ContentEncoder(args.n_downsample, args.n_res, args.input_dim, args.dim, 'in', args.activ,
#                                              args.pad_type).cuda()

cont_style_encoder: nn.Module = cont_style_munit_enc(args)#, "/home/ibespalov/pomoika/munit_encoder.pt")
cont_style_opt = torch.optim.Adam(cont_style_encoder.parameters(), lr=1e-5)
# gan_model.optimizer.add_param_group((cont_style_encoder.parameters(), None), (1e-5, None))

counter = ItersCounter()
writer = SummaryWriter(f"/home/ibespalov/pomoika/munit{int(time.time())}")
gan_model.loss_pair = send_to_tensorboard("G", "D", counter=counter, writer=writer)(gan_model.loss_pair)
# gan_model.generator.forward = send_to_tensorboard("Fake", counter=counter, skip=10, writer=writer)(
#     gan_model.generator.forward
# )

fabric = ProbabilityMeasureFabric(args.image_size)
barycenter = fabric.load("/home/ibespalov/unsupervised_pattern_segmentation/examples/face_barycenter").cuda().padding(args.measure_size).batch_repeat(args.batch_size)

g_transforms: albumentations.DualTransform = albumentations.Compose([
    MeasureToMask(size=256),
    ToNumpy(),
    NumpyBatch(albumentations.ElasticTransform(p=1, alpha=100, alpha_affine=1, sigma=10)),
    NumpyBatch(albumentations.ShiftScaleRotate(p=1, rotate_limit=10)),
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
# images, labels = next(iter(dataloader))
# content, style = cont_style_encoder(images.cuda())
# writer.add_graph(gan_model.generator, [content.cuda(), style.cuda()])
# writer.add_graph(gan_model.loss.discriminator, images.cuda())
# writer.add_graph(cont_style_encoder, images.cuda())

l1_loss = nn.L1Loss()

def L1(name: str, writer: SummaryWriter = writer) -> Callable[[Tensor, Tensor], Loss]:

    counter.active[name] = True

    def compute(t1: Tensor, t2: Tensor):
        loss = l1_loss(t1, t2)
        writer.add_scalar(name, loss, counter.get_iter(name))
        return Loss(loss)

    return compute


for epoch in range(500):

    print("epoch", epoch)

    for i, (imgs, masks) in enumerate(dataloader, 0):
        counter.update(i)
        if imgs.shape[0] != args.batch_size:
            continue

        imgs = imgs.cuda()
        content, style = cont_style_encoder(imgs)

        pred_measures: ProbabilityMeasure = ProbabilityMeasure(
            torch.ones(args.batch_size, 70, device=device) / 70,
            content.reshape(args.batch_size, 70, 2) #.sigmoid()
        )

        # if i == 1499:
        #     print("saving model to /home/ibespalov/pomoika/munit_encoder.pt")
        #     torch.save(cont_style_encoder.state_dict(), "/home/ibespalov/pomoika/munit_encoder.pt")
        #
        # if i < 1500:
        #     (Whole_Reg.apply(i)(imgs, pred_measures) * 50).minimize_step(cont_style_opt)
        #
        #     if i % 100 == 0:
        #         tensorboard_scatter(pred_measures.coord.cpu().detach(), writer, i)
        #     continue

        restored = gan_model.generator(content, style)
        sohranennii_restored = restored.detach()
        restored_content, _ = cont_style_encoder(restored)

        (L1("L1 content")(restored_content, content.detach()) * 1).minimize_step(gan_model.optimizer.opt_min, retain_graph=True)

        writer.add_scalar("style norm", (style ** 2).view(args.batch_size, -1).sum(dim=1).pow(0.5).max(), i)

        (
                L1("L1 image")(restored, imgs) * 10 +
                (R_b + R_t)(imgs, pred_measures) * 10
        ).minimize_step(gan_model.optimizer.opt_min, cont_style_opt)

        content, _ = cont_style_encoder(imgs)
        style_noise = noise.sample(args.batch_size)[:, :, None, None].tanh()
        restored = gan_model.generator(content.detach(), style_noise)
        restored_content, restored_style = cont_style_encoder(restored)

        (L1("gan L1 style")(restored_style, style_noise.detach()) * 2).minimize_step(cont_style_opt, retain_graph=True)

        if i % 100 == 0:
            with torch.no_grad():
                grid = make_grid(restored[0:4], nrow=4, padding=2, pad_value=0, normalize=True, range=(-1, 1),
                                 scale_each=False)
                writer.add_image("RESTORED_QWADRAKOPTER", grid, i)

                grid = make_grid(sohranennii_restored[0:4], nrow=4, padding=2, pad_value=0, normalize=True, range=(-1, 1),
                                 scale_each=False)
                writer.add_image("TRUE_QWADRAKOPTER", grid, i)

                grid = make_grid(imgs_with_mask(imgs, pred_measures.toImage(256))[0:4], nrow=4, padding=2, \
                                 pad_value=0, normalize=True, range=(-1, 1),
                                 scale_each=False)
                writer.add_image("IMGS WITH MASK", grid, i)

                tensorboard_scatter(pred_measures.coord.cpu().detach(), writer, i)

        (gan_model.loss_pair([imgs], [restored]).add_min_loss(
            L1("gan L1 content")(restored_content, content.detach()) * 2 +
            L1("gan L1 style")(restored_style, style_noise.detach()) * 2
        )).minimize_step(gan_model.optimizer)

        (gan_model.generator_loss([imgs], content, style_noise) * 0.2).minimize_step(cont_style_opt)








