import random

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch import nn, Tensor
import numpy as np
import os
from PIL import Image
import glob
import torch
from os.path import join as pjoin
from scipy.io import loadmat
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset




def center_by_face(image: torch.Tensor, landmarks: torch.Tensor):
    image, landmarks = np.transpose(image.numpy(), (1,2,0)), landmarks.numpy()
    # y_center = int(landmarks[36][0] + landmarks[45][0]) // 2
    # x_center = int(landmarks[:,1].mean().item())
    y, x = landmarks[:,0], landmarks[:,1]
    keypoints_landmarks = [x, y, 0, 1]
    # H, W, C = image.shape
    # W_max = min(x_center, W - x_center)
    # H_max = min(y_center, H - y_center)
    # radius = min(W_max, H_max)
    # y_max, y_min = min(H, y_center + H//2), max(0, y_center - H//2)
    # x_max, x_min = min(W, x_center + W//2), max(0, x_center - W//2)
    H, W, C = image.shape
    H09 = int(H * 0.9)
    rh = max(int(270 * H09 / W), 270)
    rw = max(int(270 * W / H09), 270)
    transforms = albumentations.Compose([
        albumentations.Crop(x_min=0, y_min=0, x_max=W, y_max=H09),
        albumentations.Resize(rh, rw),
        albumentations.CenterCrop(256, 256),
        # albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    data_dict = transforms(image=image, keypoints=[keypoints_landmarks])
    image_new = torch.tensor(np.transpose(data_dict['image'], (2,0,1)))
    kp_x, kp_y = data_dict['keypoints'][0][0], data_dict['keypoints'][0][1]
    keypoints_new = torch.cat([torch.tensor(kp_x)[..., None], torch.tensor(kp_y)[..., None]], dim=1)
    return image_new, keypoints_new

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)


def norm_range(t, range=None):
    t = t.clone()
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))
    return t

def kp_normalize(H, W, kp):
    kp = kp.clone()
    kp[..., 0] = kp[..., 0] / (W-1)
    kp[..., 1] = kp[..., 1] / (H-1)
    return kp

def pad_and_crop(im, rr):
    """Return im[rr[0]:rr[1],rr[2]:rr[3]]
    Pads if necessary to allow out of bounds indexing
    """

    meanval = np.array(np.dstack((0, 0, 0)), dtype=im.dtype)

    if rr[0] < 0:
        top = -rr[0]
        P = np.tile(meanval, [top, im.shape[1], 1])
        im = np.vstack([P, im])
        rr[0] = rr[0] + top
        rr[1] = rr[1] + top

    if rr[2] < 0:
        left = -rr[2]
        P = np.tile(meanval, [im.shape[0], left, 1])
        im = np.hstack([P, im])
        rr[2] = rr[2] + left
        rr[3] = rr[3] + left

    if rr[1] > im.shape[0]:
        bottom = rr[1] - im.shape[0]
        P = np.tile(meanval, [bottom, im.shape[1], 1])
        im = np.vstack([im, P])

    if rr[3] > im.shape[1]:
        right = rr[3] - im.shape[1]
        P = np.tile(meanval, [im.shape[0], right, 1])
        im = np.hstack([im, P])

    im = im[rr[0]:rr[1], rr[2]:rr[3]]

    return im

class ThreeHundredW(Dataset):
    """The 300W dataset, which is an amalgamation of several other datasets
    We use the split from "Face alignment at 3000 fps via regressing local binary features"
    Where they state:
    "Our training set consists of AFW, the training sets of LFPW,
    and the training sets of Helen,  with 3148 images in total.
    Our testing set consists of IBUG, the testing sets of LFPW,
    and the testing sets of Helen, with 689 images in total.
    We do not use images from XM2VTS as it is taken under a
    controlled environment and is too simple"
    """
    eye_kp_idxs = [36, 45]

    def __init__(self,
                 root,
                 train=True,
                 imwidth=100,
                 crop=15):
        from scipy.io import loadmat

        self.root = root
        self.imwidth = imwidth
        self.train = train
        self.crop = crop

        afw = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_afw.mat'))
        helentr = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_helen_trainset.mat'))
        helente = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_helen_testset.mat'))
        lfpwtr = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_lfpw_trainset.mat'))
        lfpwte = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_lfpw_testset.mat'))
        ibug = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_ibug.mat'))

        self.filenames = []
        self.bounding_boxes = []
        self.keypoints = []

        if train:
            datasets = [(afw, 'afw'), (helentr, 'helen/trainset'), (lfpwtr, 'lfpw/trainset')]
        else:
            datasets = [(helente, 'helen/testset'), (lfpwte, 'lfpw/testset'), (ibug, 'ibug')]

        for dset in datasets:
            ds = dset[0]
            ds_imroot = dset[1]
            imnames = [ds['bounding_boxes'][0, i]['imgName'][0, 0][0] for i in range(ds['bounding_boxes'].shape[1])]
            bbs = [ds['bounding_boxes'][0, i]['bb_ground_truth'][0, 0][0] for i in range(ds['bounding_boxes'].shape[1])]

            for i, imn in enumerate(imnames):
                # only some of the images given in ibug boxes exist (those that start with 'image')
                if ds is not ibug or imn.startswith('image'):
                    self.filenames.append(os.path.join(ds_imroot, imn))
                    self.bounding_boxes.append(bbs[i])

                    kpfile = os.path.join(root, ds_imroot, imn[:-3] + 'pts')
                    with open(kpfile) as kpf:
                        kp = kpf.read()
                    kp = kp.split()[5:-1]
                    kp = [float(k) for k in kp]
                    assert len(kp) == 68 * 2
                    kp = np.array(kp).astype(np.float32).reshape(-1, 2)
                    self.keypoints.append(kp)

        if train:
            assert len(self.filenames) == 3148
        else:
            assert len(self.filenames) == 689

        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # augmentations = [transforms.ToTensor()] if train else [transforms.ToTensor()]
        #
        # self.initial_transforms = transforms.Compose([transforms.Resize((self.imwidth, self.imwidth))])
        # self.transforms = transforms.Compose(augmentations)
        # self.transforms = transforms.Compose(augmentations + [normalize])
        self.transforms = albumentations.Compose([
            albumentations.Resize(self.imwidth, self.imwidth),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        im = Image.open(os.path.join(self.root, self.filenames[index])).convert("RGB")
        # Crop bounding box
        xmin, ymin, xmax, ymax = self.bounding_boxes[index]

        pm = random.randint(0, 20)

        if self.train:
            xmax = min(xmax + pm, im.width)
            xmin = max(xmin - pm, 0)
            ymax = min(ymax + pm, im.height)
            ymin = max(ymin - pm, 0)

        keypts = self.keypoints[index]

        # This is basically copied from matlab code and assumes matlab indexing
        bw = xmax - xmin + 1
        bh = ymax - ymin + 1
        bcy = ymin + (bh + 1) / 2
        bcx = xmin + (bw + 1) / 2

        # To simplify the preprocessing, we do two image resizes (can fix later if speed
        # is an issue)
        preresize_sz = 100

        bw_ = 52  # make the (tightly cropped) face 52px
        fac = bw_ / bw

        imr = im.resize((int(im.width * fac), int(im.height * fac)))

        bcx_ = int(np.floor(fac * bcx))
        bcy_ = int(np.floor(fac * bcy))
        bx = bcx_ - bw_ / 2 + 1
        bX = bcx_ + bw_ / 2
        by = bcy_ - bw_ / 2 + 1
        bY = bcy_ + bw_ / 2
        pp = (preresize_sz - bw_) / 2
        bx = int(bx - pp)
        bX = int(bX + pp)
        by = int(by - pp - 2)
        bY = int(bY + pp - 2)

        imr = pad_and_crop(np.array(imr), [(by - 1), bY, (bx - 1), bX])
        im = Image.fromarray(imr)

        cutl = bx - 1
        keypts = keypts.copy() * fac
        keypts[:, 0] = keypts[:, 0] - cutl
        cutt = by - 1
        keypts[:, 1] = keypts[:, 1] - cutt

        kp = keypts - 1  # from matlab to python style
        kp = kp * self.imwidth / preresize_sz
        kp = torch.tensor(kp)

        data = self.transforms(image=np.array(im))["image"]
        data, kp = center_by_face(data, kp[:, [1, 0]]) # kp[:, [0, 1]])
        C, H, W = data.shape
        meta = {'keypts': kp, 'keypts_normalized': kp_normalize(W, H, kp), 'index': index}

        return {"data": data, "meta": meta}
