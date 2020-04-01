import sys
import os

from torch.utils.tensorboard import SummaryWriter


sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/munit/'))

import argparse
import time
import numpy as np
import albumentations
import torch
import torchvision
from metrics.writers import send_to_tensorboard, ItersCounter, tensorboard_scatter
from torch import nn
from dataset.cardio_dataset import ImageMeasureDataset
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from gans_pytorch.gan.gan_model import cont_style_munit_enc

from loss.losses import Samples_Loss
from loss.regulariser import BarycenterRegularizer, DualTransformRegularizer

from parameters.dataset import DatasetParameters
from parameters.deformation import DeformationParameters
from parameters.gan import GanParameters, MunitParameters
from transforms_utils.transforms import MeasureToMask, ToNumpy, ToTensor, MaskToMeasure, NumpyBatch


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

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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


cont_style_encoder: nn.Module = cont_style_munit_enc(args, "/home/ibespalov/pomoika/munit_content_encoder15.pt")
cont_style_opt = torch.optim.Adam(cont_style_encoder.parameters(), lr=3e-5)

# cont_encoder = cont_style_encoder.module[0]
# torch.save(cont_encoder.state_dict(), "/home/ibespalov/pomoika/munit_content_encoder15.pt")

counter = ItersCounter()
writer = SummaryWriter(f"/home/ibespalov/pomoika/munit{int(time.time())}")


fabric = ProbabilityMeasureFabric(args.image_size)
barycenter = fabric.load("/home/ibespalov/unsupervised_pattern_segmentation/examples/face_barycenter").cuda().padding(args.measure_size).batch_repeat(args.batch_size)

g_transforms: albumentations.DualTransform = albumentations.Compose([
    MeasureToMask(size=256),
    ToNumpy(),
    NumpyBatch(albumentations.ElasticTransform(p=0.5, alpha=150, alpha_affine=1, sigma=10)),
    NumpyBatch(albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10)),
    ToTensor(device),
    MaskToMeasure(size=256, padding=args.measure_size),

])

R_b = BarycenterRegularizer.__call__(barycenter)
R_t = DualTransformRegularizer.__call__(
    g_transforms, lambda trans_dict:
    Samples_Loss(scaling=0.85, p=1)(content_to_measure(cont_style_encoder(trans_dict['image'])[0]), trans_dict['mask'])
)

R_b.forward = send_to_tensorboard("R_b", counter=counter, writer=writer)(R_b.forward)
R_t.forward = send_to_tensorboard("R_t", counter=counter, writer=writer)(R_t.forward)

# deform_array = list(np.linspace(0, 1, 1500))
# Whole_Reg = R_t @ deform_array + R_b


for i, (imgs, masks) in enumerate(dataloader, 15001):
    counter.update(i)
    if imgs.shape[0] != args.batch_size:
        continue

    imgs = imgs.cuda()
    content, style = cont_style_encoder(imgs)

    pred_measures: ProbabilityMeasure = ProbabilityMeasure(
        torch.ones(args.batch_size, 70, device=device) / 70,
        content.reshape(args.batch_size, 70, 2)
    )

    if i % 1000 == 0 and i > 0:
        print("saving model to /home/ibespalov/pomoika/munit_encoder.pt")
        torch.save(cont_style_encoder.state_dict(), f"/home/ibespalov/pomoika/munit_encoder{i//1000}.pt")

    ((R_b + R_t * 0.3)(imgs, pred_measures) * 10).minimize_step(cont_style_opt)
    print(i)
    if i % 100 == 0:
        print("ZAPUSTILOS 100")
        tensorboard_scatter(pred_measures.coord.cpu().detach(), writer, i)
