import torch
from torch import nn, Tensor

import json
import sys, os

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import CondStyleGanModel
from gan.nn.stylegan.discriminator import ConditionalDiscriminator
from gan.nn.stylegan.generator import CondGen7, ConditionalDecoder
from gan.nn.stylegan.style_encoder import StyleEncoder
from train_procedure import gan_trainer, content_trainer_with_gan, content_trainer_supervised, requires_grad, \
    train_content


from modules.accumulator import Accumulator
import albumentations
# from matplotlib import pyplot as plt
from loss.hmloss import coord_hm_loss
from metrics.measure import liuboff
from viz.image_with_mask import imgs_with_mask

from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader
from dataset.toheatmap import ToGaussHeatMap, CoordToGaussSkeleton
# from modules.hg import HG_softmax2020
from modules.nashhg import HG_skeleton
from parameters.path import Paths

from loss.tuner import GoldTuner
from loss.regulariser import DualTransformRegularizer, BarycenterRegularizer, UnoTransformRegularizer
from transforms_utils.transforms import ToNumpy, NumpyBatch, ToTensor

import torch
from torch import nn, optim

from dataset.probmeasure import UniformMeasure2DFactory, \
    UniformMeasure2D01

from metrics.writers import send_images_to_tensorboard, WR
from stylegan2.model import Generator


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)

    encoder_HG = HG_skeleton(CoordToGaussSkeleton(256, 4))
    encoder_ema = Accumulator(HG_skeleton(CoordToGaussSkeleton(256, 4)))

    print("HG")

    latent = 512
    n_mlp = 5
    size = 256

    generator = CondGen7(Generator(
        size, latent, n_mlp, channel_multiplier=1
    ), heatmap_channels=1, cond_mult=1.0)

    style_encoder = StyleEncoder(style_dim=latent)

    starting_model_number = 440000
    weights = torch.load(
        f'{Paths.default.models()}/stylegan2_new_{str(starting_model_number).zfill(6)}.pt',
        # f'{Paths.default.nn()}/stylegan2_w300_{str(starting_model_number).zfill(6)}.pt',
        map_location="cpu"
    )

    generator.load_state_dict(weights['g'])
    style_encoder.load_state_dict(weights['s'])
    encoder_HG.load_state_dict(weights['e'])

    encoder_ema.storage_model.load_state_dict(weights['e'])

    generator = generator.cuda()
    encoder_HG = encoder_HG.cuda()
    style_encoder = style_encoder.cuda()
    decoder = ConditionalDecoder(generator)

    test_img = next(LazyLoader.w300().loader_train_inf)["data"][:8].cuda()

    with torch.no_grad():
        # pred_measures_test, sparse_hm_test = encoder_HG(test_img)
        encoded_test = encoder_HG(test_img)
        pred_measures_test: UniformMeasure2D01 = UniformMeasure2D01(encoded_test["coords"])
        heatmaper_256 = ToGaussHeatMap(256, 1.0)
        sparse_hm_test_1 = heatmaper_256.forward(pred_measures_test.coord)

        latent_test = style_encoder(test_img)

        sparce_mask = sparse_hm_test_1.sum(dim=1, keepdim=True)
        sparce_mask[sparce_mask < 0.0003] = 0
        iwm = imgs_with_mask(test_img, sparce_mask)
        send_images_to_tensorboard(WR.writer, iwm, "REAL", i)

