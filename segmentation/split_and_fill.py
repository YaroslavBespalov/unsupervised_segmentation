from typing import Tuple

from torch import nn, Tensor

from framework.Loss import Loss
from framework.gan.image2image.discriminator import Discriminator
from framework.nn.modules.common.vgg import Vgg16
from framework.gan.GANModel import ConditionalGANModel
from framework.gan.image2image.unet_generator import UNetGenerator
from framework.gan.noise.normal import NormalNoise
from framework.gan.loss.penalties.AdaptiveLipschitzPenalty import AdaptiveLipschitzPenalty
from framework.gan.loss.penalties.l2_penalty import L2Penalty
from framework.gan.wgan.WassersteinLoss import WassersteinLoss
from framework.segmentation.Mask import Mask
from framework.optim.min_max import MinMaxOptimizer, MinMaxLoss
from framework.parallel import ParallelConfig


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            {}
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class VggDotLoss(nn.Module):

    def __init__(self, depth: int = 15, weight: float = 1):
        super(VggDotLoss, self).__init__()
        self.vgg = Vgg16(depth).to(ParallelConfig.MAIN_DEVICE)
        if ParallelConfig.GPU_IDS.__len__() > 1:
            self.vgg = nn.DataParallel(self.vgg, ParallelConfig.GPU_IDS)

        self.weight = weight

    def forward(self, x: Tensor, y: Tensor) -> Loss:

        x_vgg, y_vgg = self.vgg(x).view(x.size(0), -1), self.vgg(y).view(x.size(0), -1)
        x_vgg = x_vgg - x_vgg.mean(dim=0, keepdim=True).detach()
        y_vgg = y_vgg - y_vgg.mean(dim=0, keepdim=True).detach()

        return Loss(
            (x_vgg * y_vgg).mean()
        )


class FillGenerator(UNetGenerator):

    def _forward_impl(self, noise: Tensor, condition: Tensor, *additional_input: Tensor) -> Tensor:

        fake = self.unet.forward(condition, noise)
        segment = additional_input[0]
        return fake * (1 - segment) + segment * condition


class FillImageModel:

    def __init__(self,
                 image_size: int,
                 generator_size: int = 32,
                 discriminator_size: int = 32,
                 channels_count: int = 3):
        self.noise = NormalNoise(100, ParallelConfig.MAIN_DEVICE)

        self.G = FillGenerator(self.noise, image_size, channels_count, channels_count, generator_size) \
            .to(ParallelConfig.MAIN_DEVICE)
        self.D = Discriminator(discriminator_size, 2 * channels_count, image_size) \
            .to(ParallelConfig.MAIN_DEVICE)

        self.G.apply(weights_init)
        self.D.apply(weights_init)

        if ParallelConfig.GPU_IDS.__len__() > 1:
            self.G = nn.DataParallel(self.G, ParallelConfig.GPU_IDS)
            self.D = nn.DataParallel(self.D, ParallelConfig.GPU_IDS)

        was_loss = WassersteinLoss(2) \
            .add_penalty(AdaptiveLipschitzPenalty(0.1, 0.01)) \
            .add_penalty(L2Penalty(0.1))

        self.gan_model = ConditionalGANModel(self.G, self.D, was_loss)

        lr = 0.0002
        self.optimizer = MinMaxOptimizer(self.gan_model.parameters(), lr, lr)

    def train(self, images: Tensor, segments: Mask):

        front: Tensor = images * segments.tensor
        loss: MinMaxLoss = self.gan_model.loss_pair(images, front, segments.tensor)
        self.optimizer.train_step(loss)

    def test(self, images: Tensor, segments: Mask) -> Tensor:
        front: Tensor = images * segments.tensor
        return self.G(front, segments.tensor)

    def generator_loss(self, images: Tensor, segments: Mask) -> Loss:
        front: Tensor = images * segments.tensor
        fake = self.G(front, segments.tensor)
        loss = self.gan_model.generator_loss(images, fake, front)

        return loss


class SplitAndFill:

    def __init__(self,
                 image_size: int,
                 generator_size: int = 32,
                 discriminator_size: int = 32,
                 channels_count: int = 3):

        self.front_model = FillImageModel(image_size, generator_size, discriminator_size, channels_count)
        self.bk_model = FillImageModel(image_size, generator_size, discriminator_size, channels_count)

    def train(self, images: Tensor, segments: Mask):

        self.front_model.train(images, segments)
        self.bk_model.train(images, Mask(1 - segments.tensor))

    def test(self, images: Tensor, segments: Mask) -> Tuple[Tensor, Tensor]:

        return self.front_model.test(images, segments), \
               self.bk_model.test(images, Mask(1 - segments.tensor))

    def generator_loss(self, images: Tensor, segments: Mask) -> Loss:

        return self.front_model.generator_loss(images, segments) + \
            self.bk_model.generator_loss(images, Mask(1 - segments.tensor))

