import argparse
import time
from itertools import chain
from typing import Callable, Any

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
from dataset.lazy_loader import LazyLoader
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure
from gan.gan_model import cont_style_munit_enc
from metrics.writers import ItersCounter
from models.munit.enc_dec import MunitEncoder
from parameters.dataset import DatasetParameters
from parameters.deformation import DeformationParameters
from parameters.gan import GanParameters, MunitParameters
from gan.loss_base import Loss
from transforms_utils.transforms import MeasureToMask, ToNumpy, ToTensor, MaskToMeasure, NumpyBatch, MeasureToKeyPoints
from useful_utils.save import save_image_with_mask

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
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        content = encoder_content_wrapper(data)
        content_cropped = content
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((content_cropped - mes.coord).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
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

starting_model_number = 230000
# weights = torch.load(f"/home/ibespalov/pomoika/stylegan2_invertable_{str(starting_model_number).zfill(6)}.pt", map_location="cpu")
# generator.load_state_dict(weights['g'])
# discriminator.load_state_dict(weights['d'])
# cont_style_encoder.load_state_dict(weights['enc'])

# generator = generator.to(device)
# discriminator = discriminator.to(device)

encoder_content_wrapper = EncoderWrapper(cont_style_encoder).cuda()
style_opt = optim.Adam(cont_style_encoder.enc_style.parameters(), lr=1e-3, betas=(0.5, 0.97))
cont_opt = optim.Adam(encoder_content_wrapper.parameters(), lr=1e-4, betas=(0.5, 0.97))

for epoch in range(15):
    for i, batch in enumerate(LazyLoader.w300().loader_train):
        print(i)
        counter.update(i + epoch*len(LazyLoader.w300().loader_train))
        data = batch['data'].to(device)
        mes = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        content = encoder_content_wrapper(data)
        lossyash = writable("L2", otdelnaya_function)(content, mes)
        lossyash.minimize_step(cont_opt)
        if i % 100 == 0:
            test_loss = test()
            writer.add_scalar("test_loss", test_loss, i + epoch*len(LazyLoader.w300().loader_train))




