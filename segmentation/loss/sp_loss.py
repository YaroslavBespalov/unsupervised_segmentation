from torch import Tensor
import torch
import numpy as np
from gan.loss.loss_base import Loss
from transforms_utils.superpixels import mbspy
from transforms_utils.superpixels.pool import SPPoolMean


def superpixels(img: np.ndarray) -> np.ndarray:

    assert len(img.shape) == 3
    assert img.shape[2] <= 3

    mat = mbspy.Mat.from_array(img.astype(np.uint8))
    sp = mbspy.superpixels(mat, int(img.shape[1] // 2), 0.1/2)
    sp_nd = np.asarray(sp)
    # sp_nd += 1

    assert sp_nd.shape[0] == img.shape[0]
    assert sp_nd.shape[1] == img.shape[1]

    return sp_nd


def torch_sp(img: Tensor):
    return torch.stack(
        [torch.from_numpy(superpixels(img[i].detach().permute(1,2,0).cpu().numpy()))[None, ] for i in range(img.shape[0])],
        dim=0
    ).type(torch.int64).cuda()


class SuperPixelsLoss:

    def __init__(self, weight: float = 1.0):
        self.pooling = SPPoolMean()
        self.weight = weight
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, image: Tensor, segm: Tensor) -> Loss:

        sp = torch_sp(image)
        # print(sp.device)

        nc = segm.shape[1]
        sp = torch.cat([sp] * nc, dim=1).detach()

        sp_argmax = self.pooling.forward(
            segm.detach(),
            sp
        ).detach().max(dim=1)[1]

        return Loss(self.loss(segm, sp_argmax)) * self.weight


if __name__ == "__main__":


    SuperPixelsLoss().forward(torch.randn(4, 3, 128, 128).sigmoid(), torch.randn(4, 3, 128, 128).sigmoid())