import torch
from torch import nn, Tensor
from torchvision import utils


def save_image_with_mask(img: Tensor, mask: Tensor, path):

    mask = torch.cat([mask, mask, mask], dim=1)

    img[mask > 0.00001] = 1

    utils.save_image(
        img[:9],
        path,
        nrow=3,
        normalize=True,
        range=(-1, 1),
    )
