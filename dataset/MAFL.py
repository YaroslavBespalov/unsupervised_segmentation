import torch
import os
import torchvision
import numpy as np
import albumentations
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CelebA

from dataset.d300w import center_by_face
from dataset.d300w import kp_normalize


def true_center_by_face(image: torch.Tensor, landmarks: torch.Tensor):
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


class MAFLDataset(CelebA):

    transforms = albumentations.Compose([
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()]
    )

    def _check_integrity(self):
        return True

    def __getitem__(self, index):
        data, kp = super().__getitem__(index)
        data = MAFLDataset.transforms(image=np.asarray(data))["image"]

        # data = torch.from_numpy(np.asarray(data)).permute(2, 0, 1)

        kp = torch.tensor([(kp[i].item(), kp[i + 1].item()) for i in range(0, len(kp), 2)])

        data, kp = true_center_by_face(data, kp[:, [1, 0]])
        C, H, W = data.shape
        meta = {'keypts': kp, 'keypts_normalized': kp_normalize(W, H, kp), 'index': index}

        return {"data": data, "meta": meta}


