import torch

from torch import nn, Tensor

from framework.Loss import Loss
from framework.gan.GANModel import ConditionalGANModel
from framework.gan.image2image.discriminator import Discriminator
from framework.gan.image2image.unet_generator import UNetGenerator
from framework.gan.noise.normal import NormalNoise
from framework.gan.wgan.WassersteinLoss import WassersteinLoss
from framework.segmentation.Mask import Mask
from framework.optim.min_max import MinMaxOptimizer, MinMaxLoss
from framework.parallel import ParallelConfig
from framework.segmentation.split_and_fill import weights_init


class MaskToImage:
    def __init__(self,
                 image_size: int,
                 mask_channels_count: int,
                 image_channels_count: int = 3,
                 noise=NormalNoise(50, ParallelConfig.MAIN_DEVICE),
                 generator_size: int = 32,
                 discriminator_size: int = 32):

        netG = UNetGenerator(noise, image_size, mask_channels_count, image_channels_count, generator_size) \
            .to(ParallelConfig.MAIN_DEVICE)
        netD = Discriminator(discriminator_size, image_channels_count + mask_channels_count, image_size) \
            .to(ParallelConfig.MAIN_DEVICE)

        netG.apply(weights_init)
        netD.apply(weights_init)

        if torch.cuda.device_count() > 1:
            netD = nn.DataParallel(netD, ParallelConfig.GPU_IDS)
            netG = nn.DataParallel(netG, ParallelConfig.GPU_IDS)

        self.gan_model = ConditionalGANModel(
            netG,
            netD,
            WassersteinLoss(10.0)
                # .add_penalty(AdaptiveLipschitzPenalty(1, 0.05))
                # .add_penalty(L2Penalty(0.01))
        )

        lrG = 0.0001
        lrD = 0.0004
        self.optimizer = MinMaxOptimizer(self.gan_model.parameters(), lrG, lrD)

    def train(self, images: Tensor, masks: Mask):

        loss: MinMaxLoss = self.gan_model.loss_pair(images, masks.tensor)
        self.optimizer.train_step(loss)

    def generator_loss(self, images: Tensor, masks: Mask) -> Loss:

        fake = self.gan_model.generator.forward(masks.tensor)
        return self.gan_model.generator_loss(images, fake, masks.tensor)

