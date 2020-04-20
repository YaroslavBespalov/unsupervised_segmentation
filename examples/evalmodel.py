# %%
import argparse
import sys, os

from dataset.probmeasure import ProbabilityMeasure, ProbabilityMeasureFabric
from gan.gan_model import cont_style_munit_enc
from loss.losses import Samples_Loss
from models.munit.enc_dec import MunitEncoder
from parameters.dataset import DatasetParameters
from parameters.deformation import DeformationParameters
from parameters.gan import GanParameters, MunitParameters

sys.path.append(os.path.join(sys.path[0], '/home/ibespalov/unsupervised_pattern_segmentation/'))
sys.path.append(os.path.join(sys.path[0], '/home/ibespalov/unsupervised_pattern_segmentation/gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '/home/ibespalov/unsupervised_pattern_segmentation/gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '/home/ibespalov/unsupervised_pattern_segmentation/gans_pytorch/gan/'))
import albumentations
from albumentations.pytorch.transforms import ToTensor as AlbToTensor
from dataset.cardio_dataset import ImageMeasureDataset
from torch.utils import data
from transforms_utils.transforms import MaskToMeasure
import torch
# %%

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# %%

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


# %%

transform = albumentations.Compose(
    [
        albumentations.HorizontalFlip(),
        albumentations.Resize(256, 256),
        albumentations.ElasticTransform(p=0.5, alpha=100, alpha_affine=1, sigma=10),
        albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        AlbToTensor()
    ]
)

# %%

dataset = ImageMeasureDataset(
    "/raid/data/celeba",
    "/raid/data/celeba_masks",
    img_transform=transform
)

loader = data.DataLoader(
    dataset,
    batch_size=8,
    sampler=data_sampler(dataset, shuffle=True, distributed=False),
    drop_last=True,
)

# %%

loader = sample_data(loader)

# %%

parser = argparse.ArgumentParser(
    parents=[
        DatasetParameters(),
        GanParameters(),
        DeformationParameters(),
        MunitParameters()
    ],
    # formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
munit_args = parser.parse_args()

cont_style_encoder: MunitEncoder = cont_style_munit_enc(
    munit_args,
    None,  # "/home/ibespalov/pomoika/munit_content_encoder15.pt",
    None  # "/home/ibespalov/pomoika/munit_style_encoder_1.pt"
)
cont_style_encoder2: MunitEncoder = cont_style_munit_enc(
    munit_args,
    None,  # "/home/ibespalov/pomoika/munit_content_encoder15.pt",
    None  # "/home/ibespalov/pomoika/munit_style_encoder_1.pt"
)

import torch

weights = torch.load(f"/home/ibespalov/pomoika/stylegan2_invertable_320000.pt", map_location='cpu')
cont_style_encoder.load_state_dict(weights['enc'])
cont_style_encoder = cont_style_encoder.cuda()

weights2 = torch.load(f"/home/ibespalov/pomoika/stylegan2_invertable_330000.pt", map_location='cpu')
cont_style_encoder2.load_state_dict(weights2['enc'])
cont_style_encoder2 = cont_style_encoder2.cuda()


def content_to_measure(content):
    batch_size = content.shape[0]
    pred_measures: ProbabilityMeasure = ProbabilityMeasure(
            torch.ones(batch_size, 70, device=content.device) / 70,
            content.reshape(batch_size, 70, 2)
        )
    return pred_measures


fabric = ProbabilityMeasureFabric(256)
barycenter = fabric.load("/raid/data/saved_models/barycenter/face_barycenter").cuda().padding(70).batch_repeat(8)

err_pred_list = []
err_pred_list_2 = []
err_bc_list = []

for i in range(30):

    test_img, test_mask = next(loader)
    test_img = test_img.cuda()

    content = cont_style_encoder.enc_content(test_img)
    pred_measures: ProbabilityMeasure = content_to_measure(content)
    content2 = cont_style_encoder2.enc_content(test_img)
    pred_measures2: ProbabilityMeasure = content_to_measure(content2)

    ref_measure = MaskToMeasure(size=256, padding=140, clusterize=True)(image=test_img, mask=test_mask)["mask"].cuda()

    err_pred = Samples_Loss(p=1)(pred_measures, ref_measure).item()
    err_pred_2 = Samples_Loss(p=1)(pred_measures2, ref_measure).item()
    err_bc = Samples_Loss(p=1)(barycenter, ref_measure).item()
    print("pred:", err_pred)
    print("bc:", err_bc)
    err_pred_list.append(err_pred)
    err_pred_list_2.append(err_pred_2)
    err_bc_list.append(err_bc)

# %%


print("pred mean:", sum(err_pred_list) / len(err_pred_list))
print("pred mean 2:", sum(err_pred_list_2) / len(err_pred_list_2))
print("bc mean:", sum(err_bc_list) / len(err_bc_list))


