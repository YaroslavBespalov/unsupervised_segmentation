from functools import reduce

import torch
from typing import List

from torch import nn, Tensor

from framework.Loss import Loss
from framework.gan.GANModel import ConditionalGANModel
from framework.gan.conditional import ConditionalGenerator
from framework.gan.image2image.discriminator import Discriminator
from framework.gan.image2image.unet_generator import UNetGenerator
from framework.gan.noise.Noise import Noise
from framework.gan.noise.normal import NormalNoise
from framework.gan.loss.penalties.AdaptiveLipschitzPenalty import AdaptiveLipschitzPenalty
from framework.gan.loss.penalties.l2_penalty import L2Penalty
from framework.gan.vgg.gan_loss import VggGeneratorLoss
from framework.gan.wgan.WassersteinLoss import WassersteinLoss
from framework.segmentation.Mask import Mask
from framework.optim.min_max import MinMaxOptimizer, MinMaxLoss
from framework.parallel import ParallelConfig
from framework.segmentation.split_and_fill import weights_init


class CompositeGenerator(ConditionalGenerator):

    def __init__(self, noise: Noise, gen_list: nn.ModuleList):
        super(CompositeGenerator, self).__init__(noise)
        self.gen_list = gen_list

    def _forward_impl(self, noise: Tensor, condition: Tensor, *additional_input: Tensor) -> Tensor:
        assert self.gen_list.__len__() == condition.size(1)

        segments: List[Tensor] = condition.split(1, dim=1)
        fakes = [self.gen_list[i].forward(condition=s, noise=noise) * s for i, s in enumerate(segments)]

        return reduce(lambda a, b: a + b, fakes)


class MaskToImageComposite:
    def __init__(self,
                 image_size: int,
                 labels_list: List[int],
                 image_channels_count: int = 3,
                 noise=NormalNoise(100, ParallelConfig.MAIN_DEVICE),
                 generator_size: int = 32,
                 discriminator_size: int = 32):

        mask_nc = len(labels_list)

        gen_list = nn.ModuleList(
            [UNetGenerator(noise, image_size, 1, image_channels_count, int(generator_size/2), nc_max=256) for i in range(mask_nc)]
        )

        netG = CompositeGenerator(noise, gen_list) \
            .to(ParallelConfig.MAIN_DEVICE)
        netD = Discriminator(discriminator_size, image_channels_count + mask_nc, image_size) \
            .to(ParallelConfig.MAIN_DEVICE)

        netG.apply(weights_init)
        netD.apply(weights_init)

        if torch.cuda.device_count() > 1:
            netD = nn.DataParallel(netD, ParallelConfig.GPU_IDS)
            netG = nn.DataParallel(netG, ParallelConfig.GPU_IDS)

        self.gan_model = ConditionalGANModel(
            netG,
            netD,
            WassersteinLoss(2)
                .add_penalty(AdaptiveLipschitzPenalty(0.1, 0.01))
                .add_penalty(L2Penalty(0.1)) + VggGeneratorLoss(15, 1)
        )

        # vgg_loss_fn = VggGeneratorLoss(ParallelConfig.MAIN_DEVICE)

        lrG = 0.0002
        lrD = 0.0002
        self.optimizer = MinMaxOptimizer(self.gan_model.parameters(), lrG, lrD)

    def train(self, images: Tensor, masks: Mask):

        loss: MinMaxLoss = self.gan_model.loss_pair(images, masks.tensor)
        self.optimizer.train_step(loss)

    def generator_loss(self, images: Tensor, masks: Mask) -> Loss:

        fake = self.gan_model.generator.forward(masks.tensor)
        return self.gan_model.generator_loss(images, fake, masks.tensor)

