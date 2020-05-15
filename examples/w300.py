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
from models.munit.enc_dec import MunitEncoder
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

def otdelnaya_function(content: Tensor, measure: ProbabilityMeasure):
    content_cropped = content
    lossyash = Loss((content_cropped - measure.coord).abs().mean())
    return lossyash


def test():
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        mes = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        # HM_test = heatmaper.forward(mes.probability, mes.coord * 63)
        # HM_enc = enc(data)

        # print("L1: ", nn.BCELoss()(HM_enc, HM_test).item())
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        content = enc.return_coords(data)
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((content - mes.coord).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    return sum_loss / len(LazyLoader.w300().test_dataset)

class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder: MunitEncoder = encoder
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        self.layer1 = nn.Sequential(
            nn.Linear(140, 136),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layer1(self.encoder.get_content(input)).view(-1, 68, 2)

    def parameters(self, recurse=True):
        return chain(self.encoder.enc_content.parameters(), self.layer1.parameters())


class HG(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = hg4(num_classes=68, num_blocks=1)

    def forward(self, image: Tensor):
        B, C, D, D = image.shape

        heatmaps: List[Tensor] = self.model.forward(image)

        # coords = final_preds_untransformed(heatmaps[-1], (64, 64))
        return heatmaps[-1].view(B, 68, -1).softmax(dim=2).view(B, 68, 64, 64) / 68

    def return_coords(self, image: Tensor):
        heatmaps = self.forward(image)
        # coords = final_preds_untransformed(heatmaps, (64, 64)) / 64
        coords, p = heatmap_to_measure(heatmaps)
        return coords



starting_model_number = 230000
# weights = torch.load(f"/home/ibespalov/pomoika/stylegan2_invertable_{str(starting_model_number).zfill(6)}.pt", map_location="cpu")
# generator.load_state_dict(weights['g'])
# discriminator.load_state_dict(weights['d'])
# cont_style_encoder.load_state_dict(weights['enc'])

# generator = generator.to(device)
# discriminator = discriminator.to(device)

# enc = EncoderWrapper(cont_style_encoder).cuda()
enc = HG().to(device)
# style_opt = optim.Adam(cont_style_encoder.enc_style.parameters(), lr=1e-3, betas=(0.5, 0.97))
cont_opt = optim.Adam(enc.parameters(), lr=1e-4, betas=(0.5, 0.97))

W300DatasetLoader.batch_size = 8
W300DatasetLoader.test_batch_size = 32

heatmaper = ToHeatMap(64)

for epoch in range(30):
    for i, batch in enumerate(LazyLoader.w300().loader_train):
        print(i)
        counter.update(i + epoch*len(LazyLoader.w300().loader_train))
        data = batch['data'].to(device)
        mes = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        content = enc(data)
        content_xy, _ = heatmap_to_measure(content)
        target_hm = heatmaper.forward(mes.probability, mes.coord * 63)
        # lossyash = writable("L2", otdelnaya_function)(content, heatmaper.forward(mes.probability, mes.coord))
        # lossyash = Loss(nn.L1Loss()(content, heatmaper.forward(mes.probability, mes.coord * 63)))
        lossyash = Loss(
            nn.BCELoss()(content, target_hm) +
            nn.MSELoss()(content_xy, mes.coord) * 0.0005 +
            (content - target_hm).abs().mean() * 0.3
        )

        lossyash.minimize_step(cont_opt)
        writer.add_scalar("L1", lossyash.item(), i + epoch*len(LazyLoader.w300().loader_train))
        if i % 100 == 0:
            with torch.no_grad():
                plt.imshow(content[0].sum(0).cpu().detach().numpy())
                plt.show()
                test_loss = test()
                writer.add_scalar("test_loss", test_loss, i + epoch*len(LazyLoader.w300().loader_train))

    torch.save(enc.state_dict(), f"/home/ibespalov/pomoika/hg_e{epoch}.pt")


