from typing import List, Callable, Union, Tuple

import torch
from torch import nn, Tensor

import copy
from gan.loss_base import Loss


class CoefTuner:

    def __init__(self, coefs: List[float], device):
        self.coefs = torch.tensor(coefs, dtype=torch.float32, device=device).requires_grad_(True)
        self.opt = torch.optim.Adam([self.coefs], lr=0.002)

    def opt_with_grads(self,  grads: List[Tensor], coef_loss: Callable[[Tensor], Loss]):

        grads_sum = 0

        for i, g in enumerate(grads):
            grads_sum = grads_sum + self.coefs[i] * g

        norm = grads_sum.pow(2).view(grads_sum.shape[0], -1).sum(1).sqrt() + 1e-5
        grads_sum = grads_sum / norm.view(-1, 1)
        coef_loss(grads_sum).minimize_step(self.opt)
        self.coefs.data = self.coefs.data.clamp_(0, 100)


    def tune(self,
             argument: Tensor,
             coef_loss: Callable[[Tensor, Tensor], Loss],
             losses: List[Union[Loss, Tuple[Loss, Tensor]]]) -> None:
        grads = []
        for loss in losses:
            tmp_arg = argument
            if isinstance(loss, tuple):
                tmp_arg = loss[1]
                loss = loss[0]
            g = torch.autograd.grad(loss.to_tensor(), [tmp_arg], only_inputs=True)[0].detach()
            grads.append(g)

        grads_sum = 0
        for i, g in enumerate(grads):
            grads_sum = grads_sum + self.coefs[i] * g

        norm = grads_sum.pow(2).view(argument.shape[0], -1).sum(1).sqrt() + 1e-5
        grads_sum = grads_sum / norm.view(-1, 1)
        coef_loss(argument, grads_sum).minimize_step(self.opt)
        self.coefs.data = self.coefs.data.clamp_(0, 100)

    def tune_module(self, input: Tensor, module: nn.Module, losses: List[Loss], module_loss: Callable[[Tensor], Loss],
                    lr: float):
        outputs = []

        out = module(input).detach()
        print("out", module_loss(out).item())

        for loss in losses:
            module.zero_grad()
            loss.to_tensor().backward(retain_graph=True)
            module_new = copy.deepcopy(module)
            for p1, p2 in zip(module.parameters(), module_new.parameters()):
                p2._grad = p1._grad
                p2.data = p2.data - lr * p1._grad
            # module_new_opt = torch.optim.Adam(module_new.parameters(), lr=lr)
            # module_new_opt.step()
            q = module_new(input).detach()
            outputs.append(q)

        # self.opt_with_grads(grads, lambda g: module_loss(out + g))

        sum_outputs = 0
        norm = self.coefs.sum()
        for ind in range(len(outputs)):
            sum_outputs = sum_outputs + outputs[ind] * self.coefs[ind] / norm

        print("before", module_loss(sum_outputs).item())

        module_loss(sum_outputs).minimize_step(self.opt)
        self.coefs.data = self.coefs.data.clamp_(0, 100)

        sum_outputs = 0
        norm = self.coefs.sum()
        for ind in range(len(outputs)):
            sum_outputs = sum_outputs + outputs[ind] * self.coefs[ind] / norm

        print("after", module_loss(sum_outputs).item())

    def sum_losses(self, losses: List[Loss]) -> Loss:
        res = Loss.ZERO()
        for i, l in enumerate(losses):
            res = res + l * self.coefs[i].detach()

        return res
