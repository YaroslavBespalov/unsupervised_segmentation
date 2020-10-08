import json
import time
from typing import Callable, Any
import sys
import os

import albumentations

from loss.losses import Samples_Loss
from loss.regulariser import DualTransformRegularizer
from parameters.path import Paths
from transforms_utils.transforms import MeasureToMask, ToNumpy, NumpyBatch, ToTensor, MaskToMeasure, ResizeMask, \
    NormalizeMask

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../dataset'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from dataset.toheatmap import ToHeatMap, heatmap_to_measure

import torch
from torch import optim
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter

from dataset.lazy_loader import LazyLoader, W300DatasetLoader, Celeba
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure, UniformMeasure2DFactory
from metrics.writers import ItersCounter, send_images_to_tensorboard
from modules.hg import HG_softmax2020
from gan.loss_base import Loss
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

counter = ItersCounter()
writer = SummaryWriter(f"{Paths.default.board()}/w300{int(time.time())}")

print(f"{Paths.default.board()}/w300{int(time.time())}")

def writable(name: str, f: Callable[[Any], Loss]):
    counter.active[name] = True
    def decorated(*args, **kwargs) -> Loss:
        loss = f(*args, **kwargs)
        writer.add_scalar(name, loss.item(), counter.get_iter(name))
        return loss
    return decorated

def imgs_with_mask(imgs, mask, color=[1.0,1.0,1.0]):
    # mask = torch.cat([mask, mask, mask], dim=1)
    mask = mask[:, 0, :, :]
    res: Tensor = imgs.cpu().detach()
    res = res.permute(0, 2, 3, 1)
    res[mask > 0.00001, :] = torch.tensor(color, dtype=torch.float32)
    res = res.permute(0, 3, 1, 2)
    return res

def otdelnaya_function(content: Tensor, measure: ProbabilityMeasure):
    content_cropped = content
    lossyash = Loss((content_cropped - measure.coord).abs().mean())
    return lossyash


def test(enc):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        mes = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        content = enc(data)
        content_xy, _ = heatmap_to_measure(content)
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((content_xy - mes.coord).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    print("test loss: ", sum_loss / len(LazyLoader.w300().test_dataset))
    return sum_loss / len(LazyLoader.w300().test_dataset)


encoder_HG = HG_softmax2020(num_classes=68, heatmap_size=64)
# encoder_HG.load_state_dict(torch.load(f"{Paths.default.models()}/hg2_e29.pt", map_location="cpu"))
encoder_HG = encoder_HG.cuda()
encoder_HG = nn.DataParallel(encoder_HG, [0, 1, 3])

cont_opt = optim.Adam(encoder_HG.parameters(), lr=5e-5, betas=(0.5, 0.97))

W300DatasetLoader.batch_size = 36
W300DatasetLoader.test_batch_size = 36
Celeba.batch_size = 36

heatmaper = ToHeatMap(64)

# g_transforms: albumentations.DualTransform = albumentations.Compose([
#     ToNumpy(),
#     NumpyBatch(albumentations.Compose([
#            ResizeMask(256, 256),
#            # albumentations.ElasticTransform(p=1, alpha=100, alpha_affine=1, sigma=10),
#
#     ])),
#     NumpyBatch(albumentations.Compose([
#         albumentations.ShiftScaleRotate(p=1, rotate_limit=10),
#         ResizeMask(64, 64)
#     ])),
#     ToTensor(device)
# ])

g_transforms: albumentations.DualTransform = albumentations.Compose([
    ToNumpy(),
    NumpyBatch(albumentations.Compose([
        ResizeMask(h=256, w=256),
        albumentations.ElasticTransform(p=0.7, alpha=150, alpha_affine=1, sigma=10),
        albumentations.ShiftScaleRotate(p=0.7, rotate_limit=15),
        ResizeMask(h=64, w=64),
        NormalizeMask(dim=(0, 1, 2))
    ])),
    ToTensor(device),
])


def hm_svoego_roda_loss(pred, target):

    pred_xy, _ = heatmap_to_measure(pred)
    t_xy, _ = heatmap_to_measure(target)

    return Loss(
        nn.BCELoss()(pred, target) +
        nn.MSELoss()(pred_xy, t_xy) * 0.0005 +
        (pred - target).abs().mean() * 0.3
    )


R_t = DualTransformRegularizer.__call__(
    g_transforms, lambda trans_dict, img:
    hm_svoego_roda_loss(encoder_HG(trans_dict['image']), trans_dict['mask'])
)


for epoch in range(130):
    for i, batch in enumerate(LazyLoader.w300().loader_train):
        # print(i)
        counter.update(i + epoch*len(LazyLoader.w300().loader_train))

        data = batch['data'].to(device)
        mes = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        target_hm = heatmaper.forward(mes.probability, mes.coord * 63)

        content = encoder_HG(data)
        hm_svoego_roda_loss(content, target_hm).minimize_step(cont_opt)

        if i % 5 == 0:
            real_img = next(LazyLoader.celeba().loader).to(device)
            content = encoder_HG(data)
            coefs = json.load(open("../parameters/content_loss_sup.json"))
            R_t(real_img, content).__mul__(coefs["R_t"]).minimize_step(cont_opt)

        # writer.add_scalar("L1", lossyash.item(), i + epoch*len(LazyLoader.w300().loader_train))
        if i % 100 == 0:
        #     print(i)
        #     with torch.no_grad():
        #         trans = albumentations.Resize(256, 256)
        #         # plt.imshow(trans(image=trans_content[0].sum(0).cpu().detach().numpy())["image"] * 100 + trans_image[0][0].cpu().detach().numpy())
        #         mask = ProbabilityMeasure(prob, content_xy).toImage(256)
        #         # transformed_res = imgs_with_mask(trans_image, mask)
        #         send_images_to_tensorboard(writer, transformed_res, "W300_test_image", i + epoch*len(LazyLoader.w300().loader_train))
        #         # plt.show()
                test_loss = test(encoder_HG)
                writer.add_scalar("test_loss", test_loss, i + epoch*len(LazyLoader.w300().loader_train))

    # torch.save(enc.state_dict(), f"/home/ibespalov/pomoika/hg2_e{epoch}.pt")


