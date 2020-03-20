import sys
import os

from loss.losses import Samples_Loss
from loss.regulariser import DualTransformRegularizer, BarycenterRegularizer
from loss_base import Loss
from metrics.writers import send_to_tensorboard, ItersCounter
from modules.image2measure import ResImageToMeasure
import matplotlib.pyplot as plt
# sys.path.append(os.path.join(sys.path[0], '../'))
# sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))

import argparse
import albumentations
import torch
import torchvision
from torch import Tensor, nn
from torch import optim
from dataset.cardio_dataset import SegmentationDataset, MRIImages, ImageMeasureDataset
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from gans_pytorch.gan.gan_model import GANModel, ConditionalGANModel, stylegan2_cond_transfer, stylegan2_transfer
from gans_pytorch.gan.noise.normal import NormalNoise

from parameters.dataset import DatasetParameters
from parameters.deformation import DeformationParameters
from parameters.gan import GanParameters
from transforms_utils.transforms import MeasureToMask, ToNumpy, ToTensor, MaskToMeasure, NumpyBatch
from useful_utils.save import save_image_with_mask

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    parents=[
        DatasetParameters(),
        GanParameters(),
        DeformationParameters()
    ]
)
args = parser.parse_args()
for k in vars(args):
    print(f"{k}: {vars(args)[k]}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

image_size = 256
measure_size = 70

full_dataset = ImageMeasureDataset(
    "/raid/data/celeba",
    "/raid/data/celeba_masks",
    img_transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
)


g_transforms = albumentations.Compose([
    MeasureToMask(size=256),
    ToNumpy(),
    NumpyBatch(albumentations.ShiftScaleRotate(p=1, rotate_limit=20)),
    ToTensor(device),
    MaskToMeasure(size=256, padding=70),
])


train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [len(full_dataset) - 1000, 1000])

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)

noise = NormalNoise(args.noise_size, device)

cond_gan_model: GANModel = stylegan2_transfer(
    "/home/ibespalov/stylegan2/stylegan2-pytorch/checkpoint/790000.pt",
    "hinge",
    0.001,
    measure_size * 3,
    measure_size * 3 + noise.size()
)

counter = ItersCounter()

cond_gan_model.train = send_to_tensorboard("generator loss", "discriminator loss", counter=counter)(cond_gan_model.train)
cond_gan_model.generator.forward = send_to_tensorboard("Fake", counter=counter, skip=10)(
    cond_gan_model.generator.forward
)

fabric = ProbabilityMeasureFabric(args.image_size)
barycenter = fabric.load("face_barycenter").cuda().padding(args.measure_size).batch_repeat(args.batch_size)
# barycenter = fabric.cat([barycenter for b in range(args.batch_size)])
# print(barycenter.coord.shape)

image2measure = ResImageToMeasure(args.measure_size).cuda()
image2measure_opt = optim.Adam(image2measure.parameters(), lr=0.00005)

# cond_gan_model.loss += GANLossObject(
#                 lambda dx, dy: Loss.ZERO(),
#                 lambda dgz, real, fake: Loss(
#                     nn.L1Loss()(image2measure(fake[0]).coord, fabric.from_channels(real[1]).coord.detach())
#                 ) * 10,
#                 None
# )

#
# def test():
#     dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=20)
#
#     sum_loss = 0
#     with torch.no_grad():
#         for i, (imgs, masks) in enumerate(dataloader_test, 0):
#             imgs = imgs.cuda().type(torch.float32)
#             pred_measures: ProbabilityMeasure = image2measure(imgs)
#             ref_measures: ProbabilityMeasure = fabric.from_coord_tensor(masks).cuda().padding(args.measure_size)
#             ref_loss = Samples_Loss()(pred_measures, ref_measures)
#             sum_loss += ref_loss.item()
#
#     print(sum_loss / len(test_dataset))
R_t = DualTransformRegularizer.__call__(
    g_transforms, lambda trans_dict:
    Samples_Loss()(image2measure(trans_dict['image']), trans_dict['mask'])
)
R_b = BarycenterRegularizer.__call__(barycenter)

for epoch in range(500):

    # if epoch > 0:
    #     cond_gan_model.optimizer.update_lr(0.5)
        # for i in image2measure_opt.param_groups:
        #     i['lr'] *= 0.5
    print("epoch", epoch)

    for i, (imgs, masks) in enumerate(dataloader, 0):
        if imgs.shape[0] != args.batch_size:
            continue

        counter.update(i)

        imgs = imgs.cuda()
        pred_measures: ProbabilityMeasure = image2measure(imgs)
        # ref_measures: ProbabilityMeasure = fabric.from_coord_tensor(masks).cuda().padding(args.measure_size)
        cond = pred_measures.toChannels()
        z = noise.sample(cond.shape[0])

        loss_lossovichi = cond_gan_model.train([imgs], cond.detach(), z)

        fake = cond_gan_model.forward(cond.detach(), z)
        fake_measure = image2measure(fake)
        Loss(nn.L1Loss()(fake_measure.coord, pred_measures.coord.detach()) * 0.02).minimize_step(
            cond_gan_model.optimizer.opt_min
        )

        gan_mes_loss = (cond_gan_model.generator_loss([imgs], cond, z) * 0.02)
        gan_mes_loss += (R_t * 10 + R_b * 3)(imgs, pred_measures)
        gan_mes_loss.minimize_step(image2measure_opt)


        # print("loss_lossovichi:", loss_lossovichi)

        # data = g_transforms(image=imgs, mask=ref_measures)
        # new_img, new_mask = data["image"], data["mask"]


        # save_image_with_mask(new_img, new_mask.ToImage(256), "test.png")
        # cond = torch.cat((cond, z), dim=1)
        # gan_model.train(imgs, cond.detach())

        # fake = measure2image(cond)
        # g_loss = gan_model.generator_loss(imgs, fake)
        # (g_loss * 0.01).minimize_step(image2measure_opt)
        #
        # pred_measures: ProbabilityMeasure = image2measure(imgs)
        # with torch.no_grad():
        #     A, T = LinearTransformOT.forward(pred_measures, barycenter)
        # bc_loss = Samples_Loss()(pred_measures, pred_measures.detach() + T) * 0.0001 + \
        #           Samples_Loss()(pred_measures.centered(), pred_measures.centered().multiply(A).detach()) * 0.0002 + \
        #           Samples_Loss()(pred_measures.centered().multiply(A), barycenter.centered().detach()) * 0.003
        #
        #
        # bc_loss.minimize_step(image2measure_opt)

        # print(i)
        # if i % 100 == 0:
        #     print("test")
        #     test()
        #
        # if i % 10 == 0:
        #     with torch.no_grad():
        #         pred: ProbabilityMeasure = image2measure(imgs)
        #         plt.scatter(pred.coord[0, :, 0].cpu().numpy(), pred.coord[0, :, 1].cpu().numpy())
        #         plt.show()

        if i % 100 == 0:
            print(i)
            with torch.no_grad():
                # fake = measure2image(cond)
                pred: ProbabilityMeasure = image2measure(imgs)
                # fake = cond_gan_model.forward(cond, z)
                pred_mask = pred.toImage(256)
                save_image_with_mask(imgs, pred_mask, f'../sample_cond/{str(i).zfill(6)}.png')





