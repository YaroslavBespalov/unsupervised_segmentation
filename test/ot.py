import time
from typing import List
import numpy as np
import albumentations
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn
from torch import optim
from torchvision import utils
from dataset.cardio_dataset import SegmentationDataset, MRIImages, ImageMeasureDataset
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from loss.losses import Samples_Loss
from modules.linear_ot import LinearTransformOT, SOT


image_size = 256
batch_size = 16
measure_size = 68

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

dataset = ImageMeasureDataset(
    "/raid/data/celeba",
    "/raid/data/celeba_masks",
    img_transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)


fabric = ProbabilityMeasureFabric(image_size)
barycenter = fabric.load("../examples/face_barycenter").cuda().crop(measure_size)
barycenter = fabric.cat([barycenter for b in range(batch_size)])


for i, (imgs, masks) in enumerate(dataloader, 0):
    imgs = imgs.cuda()
    measures: ProbabilityMeasure = fabric.from_coord_tensor(masks).cuda().padding(measure_size)

    t1 = time.time()
    with torch.no_grad():
        A, T = LinearTransformOT.forward(measures, barycenter)
    t2 = time.time()
    dist = Samples_Loss().forward(measures, barycenter)
    t3 = time.time()

    print(dist, t2 - t1, t3-t2)

    # m_lin = measures.centered().multiply(A) + barycenter.mean()
    # plt.scatter(m_lin.coord[0, :, 1].cpu().numpy(), m_lin.coord[0, :, 0].cpu().numpy())
    # plt.scatter(barycenter.coord[0, :, 1].cpu().numpy(), barycenter.coord[0, :, 0].cpu().numpy())
    # plt.show()

    Atest = torch.tensor([[3, 0.2],
                          [0, 1]], device=device, dtype=torch.float32)
    Atest = torch.cat([Atest[None,]] * batch_size)
    bc_tr = barycenter.random_permute().multiply(Atest) + 0.1

    A, T = LinearTransformOT.forward(bc_tr, barycenter)

    m_lin = bc_tr.centered().multiply(A) + barycenter.mean()
    plt.scatter(m_lin.coord[0, :, 1].cpu().numpy(), m_lin.coord[0, :, 0].cpu().numpy())
    plt.scatter(barycenter.coord[0, :, 1].cpu().numpy(), barycenter.coord[0, :, 0].cpu().numpy())
    plt.show()
