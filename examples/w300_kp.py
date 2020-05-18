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

from dataset.toheatmap import ToHeatMap, heatmap_to_measure

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
from dataset.lazy_loader import LazyLoader, W300DatasetLoader
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from gan.gan_model import cont_style_munit_enc
from metrics.writers import ItersCounter
from models.munit.enc_dec import MunitEncoder, MunitEncoder2
from modules.hg import hg2, final_preds_untransformed, hg8, hg4
from parameters.dataset import DatasetParameters
from parameters.deformation import DeformationParameters
from parameters.gan import GanParameters, MunitParameters
from gan.loss_base import Loss
from transforms_utils.transforms import MeasureToMask, ToNumpy, ToTensor, MaskToMeasure, NumpyBatch, MeasureToKeyPoints
from useful_utils.save import save_image_with_mask
from matplotlib import pyplot as plt

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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

def test():
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        content = enc.enc_content(data)
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((content - landmarks).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    return sum_loss / len(LazyLoader.w300().test_dataset)


enc = MunitEncoder2().cuda()
enc.load_state_dict(torch.load("/home/ibespalov/pomoika/munit_e39.pt"))
# style_opt = optim.Adam(cont_style_encoder.enc_style.parameters(), lr=1e-3, betas=(0.5, 0.97))
cont_opt = optim.Adam(enc.parameters(), lr=1e-5, betas=(0.5, 0.97))

W300DatasetLoader.batch_size = 32
W300DatasetLoader.test_batch_size = 64


for epoch in range(40):
    for i, batch in enumerate(LazyLoader.w300().loader_train):
        print(i)
        counter.update(i + epoch*len(LazyLoader.w300().loader_train))
        data = batch['data'].to(device)
        mes = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        content = enc.enc_content(data)

        pred_mes = ProbabilityMeasure(mes.probability, content)

        lossyash = Loss(
            nn.MSELoss()(content, mes.coord) +
            (content - mes.coord).abs().mean() * 0.5
        )

        lossyash.minimize_step(cont_opt)

        writer.add_scalar("L1", lossyash.item(), i + epoch*len(LazyLoader.w300().loader_train))
        if i % 100 == 0:
            with torch.no_grad():
                plt.scatter(content[0, :, 0].detach().cpu().numpy(), content[0, :, 1].detach().cpu().numpy())
                plt.show()
                test_loss = test()
                writer.add_scalar("test_loss", test_loss, i + epoch*len(LazyLoader.w300().loader_train))

    torch.save(enc.state_dict(), f"/home/ibespalov/pomoika/munit_e{epoch + 40}.pt")


