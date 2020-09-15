import torch
import os
import torchvision
import numpy as np
import albumentations
from torchvision.datasets import CelebA

from dataset.d300w import center_by_face
from dataset.d300w import kp_normalize


class MAFLDataset(CelebA):
    def _check_integrity(self):
        return True

    def __getitem__(self, index):
        data, kp = super().__getitem__(index)
        data = torch.from_numpy(np.asarray(data)).permute(2, 0, 1)
        print(data.shape)
        kp = torch.tensor([(kp[i].item(), kp[i + 1].item()) for i in range(0, len(kp), 2)])

        data, kp = center_by_face(data, kp[:, [1, 0]])
        C, H, W = data.shape
        meta = {'keypts': kp, 'keypts_normalized': kp_normalize(W, H, kp), 'index': index}

        return {"data": data, "meta": meta}


