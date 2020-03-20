import torch
from torch import Tensor, nn

from framework.gan.gan_model import gan_weights_init
from framework.module import NamedModule
from style_gan.models.builder import Identity, ModuleBuilder
from style_gan.models.discriminator import AlphaMixWithProgression, ToLinear, Out
from style_gan.models.generator import GeneratorBuilder
from style_gan.models.discriminator import DiscriminatorBuilder
from style_gan.models.other import ConvBlock, EqualLinear, EqualConv2d
from style_gan.models.style import StyleNoise, Noise


class StepScale(nn.Module):
    def forward(self, img, step):
        k = img.shape[-1]//(4 * (2 ** step))
        if k < 2:
            return img
        return nn.functional.avg_pool2d(img, k).detach()


class GeneratorWithStyle(nn.Module):
    def __init__(self, alpha, in_dim):
        super(GeneratorWithStyle, self).__init__()
        self.generator_builder = GeneratorBuilder(alpha=alpha)
        # self.generator_builder.builder.init_weights(gan_weights_init)
        self.style: StyleNoise = StyleNoise(in_dim=in_dim, out_dim=128, n_mlp=2).cuda()
        self.style.apply(gan_weights_init)
        self.noise = Noise()
        self.generator = None
        self.step = None

    def set_step(self, step):
        self.generator = self.generator_builder.build(step)
        self.step = step
        return self

    def forward(self, input, cond=None, noise=None, mean_style=None, style_weight=0, mixing_range=(-1, -1)):

        noise = self.noise.forward(input, self.step, noise)
        style = self.style.forward(input, self.step, mean_style, style_weight, mixing_range)

        gen_input = {"input": input}

        for i in range(self.step+1):
            gen_input[f"style{i+1}"] = style[i]
            gen_input[f"noise{i+1}"] = noise[i]

        return self.generator(gen_input)[f"out_rgb{self.step+1}"]


class DiscriminatorWithStyle(nn.Module):
    def __init__(self, alpha: float):
        super(DiscriminatorWithStyle, self).__init__()
        self.discriminatorBuilder = DiscriminatorBuilder(alpha)
        # self.discriminatorBuilder.builder.init_weights(gan_weights_init)
        self.discriminator = None
        self.step = None

    def set_step(self, step):
        self.discriminator = self.discriminatorBuilder.build(step)
        self.step = step

    def forward(self, images: Tensor, cond: Tensor):

        input = torch.cat([images, cond], dim=1)

        return self.discriminator({f"rgb{self.step + 1}": input})['out_pr0']



