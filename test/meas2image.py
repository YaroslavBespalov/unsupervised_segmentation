from typing import List
import numpy as np
import albumentations
import matplotlib
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
#from albumentations.torch import ToTensor as ToTensorV2
from torch import Tensor, nn
from torch import optim
from torchvision import utils

from dataset.cardio_dataset import SegmentationDataset, MRIImages
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from framework.loss import Loss
from framework.gan.dcgan.discriminator import DCDiscriminator, ResDCDiscriminator
from framework.gan.gan_model import GANModel, ConditionalGANModel
from framework.gan.image2image.discriminator import Discriminator
from framework.gan.loss.hinge import HingeLoss
from framework.gan.loss.wasserstein import WassersteinLoss
from framework.gan.noise.normal import NormalNoise
from modules.image2measure import ResImageToMeasure
from modules.linear_ot import LinearTransformOT
from modules.measure2image import MeasureToImage, ResMeasureToImage
from loss.losses import Samples_Loss, linear_deformation

import os
import cv2


import matplotlib
import matplotlib.pyplot as plt


all_layers = []
def remove_sequential(network):
    for layer in network.children():
        if type(layer) == nn.Sequential: # if sequential layer, apply recursively to layers in sequential layer
            remove_sequential(layer)
        if list(layer.children()) == []: # if leaf node, add it to list
            all_layers.append(layer)



image_size = 256
batch_size = 4
padding = 200

device = torch.device("cuda")

if not os.path.exists(os.path.join("../examples/samples4/")):
            os.makedirs(os.path.join("../examples/samples4/"))

#/home/nazar/PycharmProjects/mrt
dataset = SegmentationDataset("/home/nazar/PycharmProjects/mrt",
                              transform=albumentations.Compose([
                          albumentations.Resize(image_size, image_size),
                          albumentations.CenterCrop(image_size, image_size),
                          ToTensorV2()
                      ]),
                              )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

fabric = ProbabilityMeasureFabric(image_size)

measure2image = ResMeasureToImage(padding * 3 + 2, image_size, 64).cuda()
netD = DCDiscriminator(ndf=160).cuda()
gan_model = GANModel(measure2image,
                     HingeLoss(netD).add_generator_loss(nn.L1Loss())*batch_size,
                     lr=0.0002)
noise = NormalNoise(2, device)

model_main = measure2image.main
dis_main = netD.main

first_image, first_mask = dataset[0]
first_image, first_mask = first_image[None,], first_mask[None,]

second_image, second_mask = dataset[1]
second_image, second_mask = second_image[None,], second_mask[None,]

first_imgs = first_image.cuda().type(torch.float32)
first_measures: ProbabilityMeasure = fabric.from_mask(first_mask).cuda().padding(padding)
cond1 = first_measures.toChannels()

second_imgs = second_image.cuda().type(torch.float32)
second_measures: ProbabilityMeasure = fabric.from_mask(second_mask).cuda().padding(padding)
cond2 = second_measures.toChannels()

batch_imgs = torch.cat((first_imgs, second_imgs), dim=0)
batch_imgs = batch_imgs.cuda().type(torch.float32)
batch_measures = torch.cat((first_mask, second_mask), dim=0)
batch_measures = fabric.from_mask(batch_measures).cuda().padding(padding)
batch_cond = batch_measures.toChannels()




z = noise.sample(cond1.shape[0])
print(z.shape)
print(cond1.shape)
batch_z = torch.cat((z,z), dim=0)

cond1 = torch.cat((cond1, z), dim=1)
cond2 = torch.cat((cond2, z), dim=1)
batch_cond = torch.cat((batch_cond, batch_z), dim=1)


remove_sequential(measure2image)
# print(len(all_layers))
# print(type(all_layers[9]))
with torch.no_grad():
    for i, layer in enumerate(model_main):
        # if i==9:
        #     print("cond1 shape", cond1.shape)
        #     print("cond2 shape", cond2.shape)
        cond1 = layer(cond1)
        cond2 = layer(cond2)
        fake_stack = torch.cat((cond1, cond2), dim=0)
        batch_cond = layer(batch_cond)
        print(f'layer №{i} max: ', (batch_cond.cpu() - fake_stack.cpu()).max())

    print("DISCRIMINATOR: ")

    for i, layer in enumerate(dis_main):
        # if i ==0:
        #     first_imgs = layer(first_imgs, cond1)
        #     second_imgs = layer(second_imgs, cond2)
        #     batch_imgs = layer(batch_imgs, batch_cond)
        # else:
        first_imgs = layer(first_imgs)
        second_imgs = layer(second_imgs)
        batch_imgs = layer(batch_imgs)

        fake_images = torch.cat((first_imgs.cpu(), second_imgs.cpu()), dim=0)
        print(f'layer №{i} max: ', (batch_imgs.cpu() - fake_images.cpu()).max())

        # print(f'layer №{i} mean: ', (batch_cond.cpu() - fake_stack.cpu()).mean())



# with torch.no_grad():
#         fake1 = measure2image(cond1)
#         fake2 = measure2image(cond2)
#         fake_stack = torch.cat((fake1,fake2), dim=0)
#         batch_fake = measure2image(batch_cond)
#         print('max: ', (batch_fake.cpu() - fake_stack.cpu()).max())
#         print('mean: ', (batch_fake.cpu() - fake_stack.cpu()).mean())
#         a = ((batch_fake.cpu().numpy()) - (fake_stack.cpu().numpy()))
    # plt.imshow(np.transpose(fake[0].cpu().numpy(), (1, 2, 0)))
    # plt.show()
    # f_images = np.transpose(fake[j].cpu().numpy(), (1, 2, 0))
    # cv2.imwrite(f'./samples/fimg_{str(epoch + 1).zfill(6)}.tiff', f_images)




