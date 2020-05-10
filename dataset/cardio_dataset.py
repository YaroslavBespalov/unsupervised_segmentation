import os
import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import re


class SegmentationDataset(Dataset):
    def __init__(self,
                 images_path,
                 masks_path,
                 transform_joint,
                 img_transform,
                 mask_transform=None):

        image_folders = [x for x in os.listdir(images_path)]

        self.imgs = []
        self.masks = []

        for folder in image_folders:
            for img in os.listdir(os.path.join(images_path, folder)):
                img_path = os.path.join(images_path, folder, img)
                mask_path = os.path.join(masks_path, folder, re.sub("(jpg|tiff|png|jpeg)", "npy", img))
                if os.path.isfile(mask_path):
                    self.imgs += [img_path]
                    self.masks += [mask_path]

        self.transform = transform_joint
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        image = np.array(Image.open(self.imgs[index]).convert('RGB'))
        mask = np.load(self.masks[index]).astype(np.int32)

        # print(self.masks[index],  np.where(mask > 0.001))
        dict_transfors = self.transform(image=image, mask=mask)
        image, mask = dict_transfors['image'].type(torch.float32) / 255, dict_transfors['mask'].type(torch.float32)
        # image = torchvision.transforms.Normalize(mean=[57.0, 57.0, 57.0], std=[57.0, 57.0, 57.0])(image)
        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask[None, :])[0]
        return image, mask

    def __len__(self):
        return len(self.masks)


class MRIImages(Dataset):
    def __init__(self, path, transform):
        self.init = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.tiff')]
        self.transform = transform

    def __getitem__(self, index):
        image = np.array(Image.open(self.init[index]))
        dict_transfors = self.transform(image=image)
        image: Tensor = dict_transfors['image'].type(torch.float32)
        m = [57.0, 57.0, 57.0]
        sd = [57.0, 57.0, 57.0]
        image = torchvision.transforms.Normalize(mean=m, std=sd)(image)

        return image

    def __len__(self):
        return len(self.init)


class ImageMeasureDataset(Dataset):
    def __init__(self,
                 images_path,
                 masks_path,
                 img_transform):

        image_folders = [x for x in os.listdir(images_path)]

        self.imgs = []
        self.masks = []

        for folder in image_folders:
            for img in os.listdir(os.path.join(images_path, folder)):
                img_path = os.path.join(images_path, folder, img)
                mask_path = os.path.join(masks_path, folder, re.sub("(jpg|tiff|png|jpeg)", "npy", img))
                if os.path.isfile(mask_path):
                    self.imgs += [img_path]
                    self.masks += [mask_path]

        self.img_transform = img_transform

    def __getitem__(self, index):
        image = np.array(Image.open(self.imgs[index]).convert('RGB'))
        mask = np.load(self.masks[index]).astype(np.int32)

        dict_transfors = self.img_transform(image=image, mask=mask)
        image = dict_transfors['image']
        mask = dict_transfors['mask']

        return image, mask

    def __len__(self):
        return len(self.masks)
