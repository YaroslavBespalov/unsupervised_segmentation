import math
import unittest
from typing import List, Dict

import torch
from torch import nn, Tensor

from model import StyledConv, ToRGB
from models.unet.la_divina_progressiya import Progressive,  ProgressiveWithStateInit, \
    ProgressiveSequential, InjectLast, InjectByName

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

class TestProgMod(nn.Module):
    def forward(self, input: Tensor, state: Tensor):
        return state + (input * 2)[..., None]


class TestProgModWithoutInput(nn.Module):
    def forward(self, state: Tensor):
        return state / 7


class TestProgModDict(nn.Module):
    def forward(self, t1: Tensor, state: Tensor, t2: Tensor):
        return t1 + state * 2 + t2 * 3


ml = [
            TestProgMod(),
            TestProgMod(),
            TestProgMod(),
        ]

dict_ml = [
            TestProgModDict(),
            TestProgModDict(),
            TestProgModDict()
        ]

input = [
    torch.randn(2, 3),
    torch.randn(2, 3),
    torch.randn(2, 3)
]

state_init = torch.randn(2, 3, 2)

dict_state_init = {'state': torch.randn(2, 3)}

t1_input = [
    torch.randn(2, 3),
    torch.randn(2, 3),
    torch.randn(2, 3),
]
t2_input = [
    torch.randn(2, 3),
    torch.randn(2, 3),
    torch.randn(2, 3),
]

class TestSimpleProgression(unittest.TestCase):
    def test_outputs(self):

        state = state_init

        res_output_first = []
        for i in range(3):
            state = ml[i](input[i], state)
            res_output_first.append(state)

        pr = Progressive[List[Tensor]](ml, InjectLast())

        res_list = pr.forward(state_init, input)

        i = 0
        for rr in res_list:
            self.assertAlmostEqual((rr - res_output_first[i]).abs().max().item(), 0, delta=1e-5)
            i += 1


    def test_dict(self):
        state = dict_state_init['state']
        res_output_first = []
        for i in range(3):
            state = dict_ml[i](t1_input[i], state, t2_input[i])
            res_output_first.append(state)

        pr = Progressive[List[Tensor]](
            dict_ml,
            InjectByName("state")
        )

        res_list = pr.forward(state=dict_state_init['state'], t1=t1_input, t2=t2_input)

        i = 0
        for rr in res_list:
            self.assertAlmostEqual((rr - res_output_first[i]).abs().max().item(), 0, delta=1e-5)
            i += 1

    def test_styleganich(self):
        print("AKULELE")
        channels = {
            4: 64,
            8: 64,
            16: 32,
            32: 32,
            64: 16,
            128: 16,
            256: 8,
            512: 8,
            1024: 4
        }
        log_size = int(math.log(64, 2))
        n_latent = log_size * 2 - 2
        blur_kernel = [1, 3, 3, 1]
        style_dim = 100
        convs = []
        conv1 = StyledConv(
            channels[4], channels[4], 3, style_dim, blur_kernel=blur_kernel
        ).cuda()
        in_channel = channels[4]

        to_rgb1 = ToRGB(channels[4], style_dim, upsample=False).cuda()
        to_rgbs = []
        for i in range(3, log_size + 1):
            out_channel = channels[2 ** i]
            convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                ).cuda()
            )

            convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                ).cuda()
            )
            in_channel = out_channel
            to_rgbs.append(ToRGB(out_channel, style_dim).cuda())

        #forward
        out = torch.randn(8, channels[4], 4, 4).cuda()
        init_state = out
        latent = torch.randn(8, n_latent, style_dim).cuda()
        noise = [torch.randn(8, 1, 4, 4).cuda()]
        for i in range(0, len(convs)//2):
            noise.append(torch.randn(8, 1, 8 * (2 ** i), 8 * (2 ** i)).cuda())
            noise.append(torch.randn(8, 1, 8 * (2 ** i), 8 * (2 ** i)).cuda())

        main_out = []
        out = conv1(out, latent[:, 0], noise=noise[0])
        skip = to_rgb1(out, latent[:, 1])
        skip1 = skip
        main_out.append(out)
        i = 1
        for conv3, conv4, noise1, noise2, to_rgb in zip(
                convs[::2], convs[1::2], noise[1::2], noise[2::2], to_rgbs
        ):
            out = conv3(out, latent[:, i], noise=noise1)
            main_out.append(out)
            out = conv4(out, latent[:, i + 1], noise=noise2)
            main_out.append(out)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2


        dict_ml = [conv1] + convs
        # input_dict = [{'noise': noise[i], 'style': latent[:,i,:]} for i in range(len(convs)+1)]
        style_list = [latent[:, i] for i in range(len(convs)+1)]

        pr = Progressive[List[Tensor]](dict_ml, InjectByName("input"))
        progessive_out = pr.forward(init_state, noise=noise, style=style_list)
        for i in range(len(progessive_out)):
            self.assertAlmostEqual((main_out[i] - progessive_out[i]).abs().max().item(), 0, delta=1e-5)

        pr2_with_init = ProgressiveWithStateInit(
            to_rgb1,
            Progressive[List[Tensor]](to_rgbs, InjectByName("skip"))
        )

        style_list_1 = [latent[:, i, :] for i in range(1, len(convs) + 2, 2)]
        input_list_1 = [progessive_out[i-1] for i in range(1, len(convs) + 2, 2)]
        progessive_skip_1 = pr2_with_init.forward(input=input_list_1, style=style_list_1)

        self.assertAlmostEqual((skip - progessive_skip_1[-1]).abs().max().item(), 0, delta=1e-5)

        # seq = ProgressiveSequential(
        #     (pr, "input", {'noise', 'style'}, list(range(len(input_dict)))),
        #     (pr2_with_init, "res", {'input', 'style'}, list(range(1, len(convs) + 2, 2)))
        # )
        #
        # seq.forward(input_dict, state)
