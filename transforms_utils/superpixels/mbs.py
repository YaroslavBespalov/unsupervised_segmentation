import numpy as np
import cv2
from matplotlib import pyplot as plt
from torch import Tensor
from typing import List

from transforms_utils.superpixels import mbspy
from multiprocessing import Pool
import torch


def superpixels(img: np.ndarray) -> np.ndarray:

    assert len(img.shape) == 3
    assert img.shape[2] <= 3

    mat = mbspy.Mat.from_array(img.astype(np.uint8))
    sp = mbspy.superpixels(mat, int(img.shape[1] // 2), 0.1)
    sp_nd = np.asarray(sp)
    # sp_nd += 1

    assert sp_nd.shape[0] == img.shape[0]
    assert sp_nd.shape[1] == img.shape[1]

    return sp_nd


if __name__ == "__main__":

    img = cv2.imread('/home/nazar/PycharmProjects/segmentation_data/leftImg8bit/train/strasbourg/strasbourg_000000_000065_leftImg8bit.png')
    sp_nd = superpixels(img)

    print(sp_nd)

    plt.imshow(sp_nd)
    plt.show()

