from typing import Callable

import torch
from torch import nn, Tensor

from gan.nn.stylegan.generator import Generator1

torch.cuda.set_device("cuda:1")

gen = nn.DataParallel(Generator1(128, 512, 5, channel_multiplier=1).cuda(), [1,3])

styles = torch.randn(8, 512).cuda()
cond = torch.randn(8, 512, 4, 4).cuda()

noise = [torch.randn(8, 512, 4, 4).cuda()]
for i in range(0, gen.module.log_size - 2):
    noise.append(torch.randn(8, gen.module.channels[2 ** (i+3)], 8 * (2 ** i), 8 * (2 ** i)).cuda())
    noise.append(torch.randn(8, gen.module.channels[2 ** (i+3)], 8 * (2 ** i), 8 * (2 ** i)).cuda())

res = gen.forward([styles], cond, noise=noise)

