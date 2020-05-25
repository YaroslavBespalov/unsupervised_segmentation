import time
from typing import Dict

import albumentations
import torch
import numpy as np
from albumentations import DualTransform, BasicTransform
from dataset.probmeasure import ProbabilityMeasure, ProbabilityMeasureFabric
from scipy.ndimage import label, generate_binary_structure
from joblib import Parallel, delayed, cpu_count


class MeasureToMask(DualTransform):
    def __init__(self, size=256):
        super(MeasureToMask, self).__init__(1)
        self.size = size

    def apply(self, img: torch.Tensor, **params):
        return img

    def apply_to_mask(self, img: ProbabilityMeasure, **params):
        return img.toImage(self.size)


class MeasureToKeyPoints(DualTransform):
    def __init__(self):
        super(MeasureToKeyPoints, self).__init__(1)

    def apply(self, img: torch.Tensor, **params):
        return img

    def apply_to_keypoint(self, mes: ProbabilityMeasure, **params):
        params["prob"] = mes.probability
        return [mes.coord[:, 1], mes.coord[:, 0], 0, 1]


class KeyPointsToMeasure(DualTransform):
    def __init__(self):
        super(KeyPointsToMeasure, self).__init__(1)

    def apply(self, img: torch.Tensor, **params):
        return img

    def apply_to_keypoint(self, kp, **params):
        x, y, a, s = kp
        return ProbabilityMeasure(params["prob"], torch.cat([y[..., None], x[..., None]], dim=-1))


class ResizeMask(DualTransform):
    def __init__(self, h: int, w: int):
        super(ResizeMask, self).__init__(1)
        self.resize = albumentations.Resize(h, w)

    def apply(self, img: np.ndarray, **params):
        return img

    def apply_to_mask(self, mask: torch.Tensor, **params):
        return self.resize.apply_to_mask(mask)

class ResizeImage(DualTransform):
    def __init__(self, h: int, w: int):
        super(ResizeImage, self).__init__(1)
        self.resize = albumentations.Resize(h, w)

    def apply(self, img: np.ndarray, **params):
        return self.resize.apply(img)

    def apply_to_mask(self, mask: torch.Tensor, **params):
        return mask


class ToNumpy(DualTransform):
    def __init__(self):
        super(ToNumpy, self).__init__(1)

    def apply(self, img: torch.Tensor, **params):
        return np.transpose(img.detach().cpu().numpy(), [0, 2, 3, 1])

    def apply_to_mask(self, mask: torch.Tensor, **params):
        return np.transpose(mask.detach().cpu().numpy(), [0, 2, 3, 1])

    def apply_to_keypoint(self, keypoint, **params):
        x, y, a, s = keypoint
        return [x.detach().cpu().numpy(), y.detach().cpu().numpy(), a, s]


class ToTensor(DualTransform):
    def __init__(self, device):
        super(ToTensor, self).__init__(1)
        self.device = device

    def apply(self, img: np.array, **params):
        return torch.tensor(np.transpose(img, [0, 3, 1, 2]), device=self.device)

    def apply_to_mask(self, img: np.array, **params):
        return torch.tensor(np.transpose(img, [0, 3, 1, 2]), device=self.device)


class NormalizeMask(albumentations.DualTransform):
    def __init__(self, dim):
        super(NormalizeMask, self).__init__(1)
        self.dim = dim

    def apply(self, img: torch.Tensor, **params):
        return img

    def apply_to_mask(self, img: torch.Tensor, **params):
        img = np.array(img)
        img[img<0]=0
        img = img / (img.sum(axis=self.dim, keepdims=True) + 1e-10)
        return img


class ParTr(albumentations.DualTransform):

    def __init__(self, transform: DualTransform):
        super(ParTr, self).__init__(1)
        self.transform = transform
        self.par_img = Parallel(n_jobs=3)
        self.par_mask = Parallel(n_jobs=8)

    def apply(self, img: np.ndarray, **params):
        processed_list = self.par_img(
            delayed(self.transform.apply)(img[:, :, i:i+1]) for i in range(img.shape[-1])
        )
        return np.concatenate(processed_list, axis=2)

    def apply_to_mask(self, img: np.ndarray, **params):
        processed_list = self.par_mask(
            delayed(self.transform.apply_to_mask)(img[:, :, i:i + 1]) for i in range(img.shape[-1])
        )
        return np.concatenate(processed_list, axis=2)


class NumpyBatch(BasicTransform):

    def __init__(self, transform: BasicTransform):
        super(NumpyBatch, self).__init__(1)
        self.transform = transform
        self.par = Parallel(n_jobs=24)

    def __call__(self, force_apply=False, **kwargs):

        keys = ["image"]
        if "mask" in kwargs:
            keys.append("mask")

        def compute(transform, tdata: Dict[str, np.ndarray]):

            data_i = transform(**tdata)
            # print(data_i)

            if data_i["image"].sum() is None:
                print("None in transform!!! Transform cancelled.")
                data_i = tdata

            return data_i

        processed_list = self.par(delayed(compute)(
            self.transform, {k: kwargs[k][i] for k in keys}) for i in range(kwargs["image"].shape[0])
        )

        batch = {key: [] for key in keys}

        for data in processed_list:
            for key in keys:
                # if key == "mask":
                    # data["mask"] = np.array(data["mask"])
                    # data["mask"][data["mask"] < 0] = 0
                    # data["mask"] = data["mask"] / (data["mask"].sum() + 1e-8)
                batch[key].append(data[key][np.newaxis, ...])

        return {key: np.concatenate(batch[key], axis=0) for key in keys}


class MaskToMeasure(DualTransform):
    def __init__(self, size=256, padding=140, p=1.0, clusterize=True):
        super(MaskToMeasure, self).__init__(p)
        self.size = size
        self.padding = padding
        self.clusterize = clusterize

    def apply(self, img: torch.Tensor, **params):
        return img

    def apply_to_mask(self, img: torch.Tensor, **params):
        if self.clusterize:
            res = clusterization(img,
            size=self.size, padding=self.padding)
            return res
        else:
            return ProbabilityMeasureFabric(self.size).from_mask(img).padding(self.padding)


def clusterization(images: torch.Tensor, size=256, padding=70):

    imgs = images.cpu().numpy().squeeze()
    pattern = generate_binary_structure(2, 2)
    coord_result, prob_result = [], []

    # print("img sum:", images.sum(dim=[1,2,3]).max())
    # t1 = time.time()

    # for sample in range(imgs.shape[0]):
    def compute(sample):
        x, y = np.where((imgs[sample] > 1e-6))
        measure_mask = np.zeros((2, size, size))
        measure_mask[0, x, y] = 1
        measure_mask[1, x, y] = imgs[sample, x, y]
        labeled_array, num_features = label(measure_mask[0], structure=pattern)
        # if num_features > 75:
        #     print(num_features)

        x_coords, y_coords, prob_value = [], [], []
        sample_centroids_coords, sample_probs_value = [], []

        for i in range(1, num_features + 1):
            x_clust, y_clust = np.where(labeled_array == i)
            x_coords.append(np.average(x_clust) / size)
            y_coords.append(np.average(y_clust) / size)
            prob_value.append(np.sum(measure_mask[1, x_clust, y_clust]))
            assert(measure_mask[1, x_clust, y_clust].all() != 0)
            # print("PROB_VALUE ", prob_value)

        [x_coords.append(0) for i in range(padding - len(x_coords))]
        [y_coords.append(0) for i in range(padding - len(y_coords))]
        [prob_value.append(0) for i in range(padding - len(prob_value))]

        sample_centroids_coords.append([x_coords, y_coords])
        sample_probs_value.append(prob_value)

        sample_centroids_coords = np.transpose(np.array(sample_centroids_coords), axes=(0, 2, 1))
        sample_probs_value = np.array(sample_probs_value)

        # coord_result.append(sample_centroids_coords)
        # assert(sample_probs_value.sum() != 0)
        # assert(sample_probs_value.all() / sample_probs_value.sum() >= 0)
        # prob_result.append(sample_probs_value / sample_probs_value.sum())
        return x_coords, y_coords, sample_probs_value / (sample_probs_value.sum() + 1e-8)
        #return sample_centroids_coords,  sample_probs_value / (sample_probs_value.sum() + 1e-8)

    processed_list = Parallel(n_jobs=16)(delayed(compute)(i) for i in range(imgs.shape[0]))

    for x, y, p in processed_list:
        coord_result.append(torch.cat((torch.tensor(y)[:, None], torch.tensor(x)[:,None]),dim=1)[None, ...])
        prob_result.append(p)
    # print(time.time() - t1)

    return ProbabilityMeasure(torch.tensor(np.concatenate(prob_result, axis=0)).type(torch.float32),
                              torch.cat(coord_result).type(torch.float32)).cuda()





